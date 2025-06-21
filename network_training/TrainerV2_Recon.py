#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author      : Han Liu
# Date Created: 02/10/2023


import torch
import numpy as np
from time import time
import os.path as osp
from network_training.TrainerV2 import TrainerV2
from utils.util import poly_lr, update_log, get_time, get_lr

import random
from utils.ops import *
from utils import view_ops
from utils import view_transforms
from utils.losses import *
from time import time
from monai.losses import DiceLoss
from monai.metrics import compute_dice
from monai.transforms import AsDiscrete
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from torch.cuda.amp import GradScaler, autocast
from monai.networks.utils import one_hot
from loss_functions.deep_supervision import MultipleOutputLoss


#--------------------------------------
# Trainer for image reconstruction task
#--------------------------------------
# Loss: reconstruction loss, L2


class TrainerV2_Recon(TrainerV2):
    
    def __init__(self, data, model, opt):
        super().__init__(data, model, opt)
        self.loss_fn = torch.nn.MSELoss(reduction='mean')
        self.best_metric = 1e9

    def fit(self, data):
        # Swin-UNETR
        if self.opt.nid == 13:
            # print("TRUE!!!")
            loss_function = Loss(self.opt.batch_size, self.opt)

            image, target = data['data_image'], data['data_image']
            image, target = image.cuda(), target.cuda()
            target = self.prep_label(target)

            x1, rot1 = rot_rand(self.opt, image)
            x2, rot2 = rot_rand(self.opt, image)
            x1_augment = aug_rand(self.opt, x1)
            x2_augment = aug_rand(self.opt, x2)
            x1_augment = x1_augment
            x2_augment = x2_augment
            with autocast(enabled=False):
                rot1_p, contrastive1_p, rec_x1 = self.model(x1_augment)
                rot2_p, contrastive2_p, rec_x2 = self.model(x2_augment)
                rot_p = torch.cat([rot1_p, rot2_p], dim=0)
                rots = torch.cat([rot1, rot2], dim=0)
                imgs_recon = torch.cat([rec_x1, rec_x2], dim=0)
                imgs = torch.cat([x1, x2], dim=0)
                self.loss, _ = loss_function(rot_p, rots, contrastive1_p, contrastive2_p, imgs_recon, imgs)
        
        # SwinMM
        elif self.opt.nid == 14:
            mutual_loss_function = MutualLoss(self.opt)
            loss_function = Loss_V2(self.opt.batch_size, self.opt)

            image, target = data['data_image'], data['data_image']
            image, target = image.cuda(), target.cuda()

            x1, rot1 = view_ops.rot_rand(image)
            x2, rot2 = view_ops.rot_rand(image)

            window_sizes = tuple(self.opt.window_size for _ in range(3))
            input_sizes = self.opt.crop_size

            x1_masked, mask1 = mask_rand_patch(window_sizes, input_sizes, self.opt.mask_ratio, x1)
            x2_masked, mask2 = mask_rand_patch(window_sizes, input_sizes, self.opt.mask_ratio, x2)

            permutations_candidates = set(
                view_transforms.permutation_transforms.keys()) - {0}
            permutations = [
                random.choice(list(permutations_candidates)) for _ in range(2)
            ]
            x1_masked_permuted, x2_masked_permuted = [
                view_transforms.permutation_transforms[vn](val)
                for vn, val in zip(permutations, [x1_masked, x2_masked])
            ]

            with autocast(enabled=False):
                rot1_p, contrastive1_p, rec_x1 = self.model(x1_masked)
                rot2_p, contrastive2_p, rec_x2 = self.model(x2_masked)
                _, contrastive3_p, rec_x3 = self.model(x1_masked_permuted)
                _, contrastive4_p, rec_x4 = self.model(x2_masked_permuted)

                # masked voxels: [2, H, W, D]
                mask = torch.stack([mask1, mask2], dim=0)
                rec_x3, rec_x4 = [
                    view_transforms.permutation_inverse_transforms[vn](val)
                    for vn, val in zip(permutations, [rec_x3, rec_x4])
                ]

                rot_p = torch.cat([rot1_p, rot2_p], dim=0)
                rots = torch.cat([rot1, rot2], dim=0)
                # [B, 2, H, W, D]
                imgs_recon = torch.cat([rec_x1, rec_x2], dim=1)
                imgs = torch.cat([x1, x2], dim=1)

                # loss1, losses_tasks1 = loss_function(rot_p, rots,
                #                                     contrastive1_p,
                #                                     contrastive2_p, imgs_recon,
                #                                     imgs, mask, only_mae=True)
                
                loss1, _ = loss_function(rot_p, rots,
                                      contrastive1_p,
                                      contrastive2_p, imgs_recon,
                                      imgs, mask, only_mae=False)

                mutual_loss1 = mutual_loss_function(rec_x3, rec_x1, mask1)

                imgs_recon = torch.cat([rec_x3, rec_x4], dim=1)
                loss2, _ = loss_function(rot_p,
                                      rots,
                                      contrastive3_p,
                                      contrastive4_p,
                                      imgs_recon,
                                      imgs,
                                      mask,
                                      only_mae=False)

                loss = loss1 + loss2 + mutual_loss1
                # loss = loss1 + loss2

                mutual_loss2 = None
                if self.opt.mutual_learning_on_more_view:
                    def ensure_five_dims(tensor):
                        """
                        Ensure the input MetaTensor has five dimensions. If it has four dimensions,
                        add an additional dimension at the beginning.

                        Args:
                            tensor (MetaTensor): The input MetaTensor to be checked.

                        Returns:
                            MetaTensor: The MetaTensor with five dimensions.
                        """
                        if tensor.dim() == 4:
                            tensor = tensor.unsqueeze(0)  # Add batch dimension
                        return tensor

                    def _align_rot(x, src_rot, dst_rot):
                        x = ensure_five_dims(x)
                        return view_transforms.rotation_transforms[dst_rot](
                            view_transforms.rotation_inverse_transforms[src_rot]
                            (x)).contiguous()

                    # [B, C, H, W, D]
                    rec_x4_aligned = torch.stack([
                        _align_rot(val, src_rot.item(), dst_rot.item())
                        for val, src_rot, dst_rot in zip(rec_x4, rot2, rot1)
                    ])
                    # [B, 1, H, W, D]
                    mask2_aligned = torch.concat([
                        _align_rot(mask2[None, None], src_rot.item(),
                                   dst_rot.item())
                        for src_rot, dst_rot in zip(rot2, rot1)
                    ])
                    mask_intersection = torch.logical_and(mask2_aligned, mask1)
                    # Rescale to the same scale of mutual_loss1
                    rescaler = (mask1.sum() * mask2_aligned.size(0) /
                                (mask2_aligned.sum() + 1e-6))
                    mutual_loss2 = mutual_loss_function(
                        rec_x4_aligned, rec_x1, mask_intersection) * rescaler

                    loss = loss + mutual_loss2
                
            self.loss = loss

        else:
            image, target = data["data_image"], data["data_image"]
            image, target = image.cuda(), target.cuda()
            pred = self.model(image)            
            self.loss = self.loss_fn(pred, target)

        self.optim.zero_grad()
        self.loss.backward()
        self.gradient_clipping()
        self.optim.step()
        self.train_loss += self.loss.item()

    def valid(self):
        torch.cuda.empty_cache()
        self.set_evaluation_mode()
        self.val_loss = []
        
        with torch.no_grad():
            for i, data in enumerate(self.valid_ds):
                start = time()
                image, target = data["data_image"], data["data_image"]
                image, target = image.cuda(), target.cuda()
                sub = data["subject"]
                sub = osp.splitext(osp.basename(sub[0]))[0]
                pred = self.model(image)[-1]
                # pred = self.predict(image)
                assert pred.shape == target.shape
                mse = torch.mean((pred - target) ** 2).cpu().numpy()
                self.val_loss.append(mse)

                if self.opt.display_per_iter:
                    update_log((f"{get_time():%Y-%m-%d %H:%M:%S}: "
                        f"epoch={self.epoch}, id={i+1}/{len(self.valid_ds.dataset)}, "
                        f"subject={sub}, time={time()-start:.4f}, "
                        f"mse={mse:.4f}"), self.run_log)
            
            self.val_loss = np.nanmean(self.val_loss, axis=0)  # report mse without nans
            update_log(f"{self.val_loss}", self.val_log, verbose=False)
            self.save_ckpt()
            update_log((f"{get_time():%Y-%m-%d %H:%M:%S}: current mean mse:"
                f" {self.val_loss:.4f}, best mean mse: {self.best_metric:.4f}"
                f" at epoch {self.best_metric_epoch}"), self.run_log)
        

    def save_ckpt(self):
        if self.val_loss < self.best_metric:
            self.epochs_no_improve = 0
            self.best_metric = self.val_loss
            self.best_metric_epoch = self.epoch
            torch.save({'epoch': self.epoch,
                        'state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optim.state_dict(),
                        'best_metric': self.best_metric,
                        'best_metric_epoch': self.best_metric_epoch},
                        osp.join(self.opt.expr_dir, f'best_model.pth'))
        else:
            self.epochs_no_improve += 1

    def run(self):
        super().run()
    
