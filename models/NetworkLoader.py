#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author      : Han Liu
# Date Created: 06/02/2022


import os.path as osp
from typing import Sequence
import torch
import torch.nn as nn
from monai.networks.nets.swin_unetr import SwinTransformer as SwinViT
from monai.utils import ensure_tuple_rep
from monai.networks.layers import Norm
from monai.networks.nets import *
from models.DI2IN import *
from models.nnFormer import nnFormer
from models.generic_UNet import Generic_UNet
from models.DoDNet import DoDNet
from models.DeeplabV3_plus import DeepLabV3_3D
from utils.util import update_log


class SSLHead(nn.Module):
    def __init__(self, args, upsample="vae", dim=768):
        super(SSLHead, self).__init__()
        patch_size = ensure_tuple_rep(2, 3)
        window_size = ensure_tuple_rep(7, 3)
        self.swinViT = SwinViT(
            in_chans=args.input_nc,
            embed_dim=args.embedding_dim,
            window_size=window_size,
            patch_size=patch_size,
            depths=[2, 2, 2, 2],
            num_heads=[3, 6, 12, 24],
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=args.dropout_path_rate,
            norm_layer=torch.nn.LayerNorm,
            use_checkpoint=False,
            spatial_dims=3,
        )
        self.rotation_pre = nn.Identity()
        self.rotation_head = nn.Linear(dim, 4)
        self.contrastive_pre = nn.Identity()
        self.contrastive_head = nn.Linear(dim, 512)
        if upsample == "large_kernel_deconv":
            self.conv = nn.ConvTranspose3d(dim, args.in_channels, kernel_size=(32, 32, 32), stride=(32, 32, 32))
        elif upsample == "deconv":
            self.conv = nn.Sequential(
                nn.ConvTranspose3d(dim, dim // 2, kernel_size=(2, 2, 2), stride=(2, 2, 2)),
                nn.ConvTranspose3d(dim // 2, dim // 4, kernel_size=(2, 2, 2), stride=(2, 2, 2)),
                nn.ConvTranspose3d(dim // 4, dim // 8, kernel_size=(2, 2, 2), stride=(2, 2, 2)),
                nn.ConvTranspose3d(dim // 8, dim // 16, kernel_size=(2, 2, 2), stride=(2, 2, 2)),
                nn.ConvTranspose3d(dim // 16, args.in_channels, kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            )
        elif upsample == "vae":
            self.conv = nn.Sequential(
                nn.Conv3d(dim, dim // 2, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm3d(dim // 2),
                nn.LeakyReLU(),
                nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
                nn.Conv3d(dim // 2, dim // 4, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm3d(dim // 4),
                nn.LeakyReLU(),
                nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
                nn.Conv3d(dim // 4, dim // 8, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm3d(dim // 8),
                nn.LeakyReLU(),
                nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
                nn.Conv3d(dim // 8, dim // 16, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm3d(dim // 16),
                nn.LeakyReLU(),
                nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
                nn.Conv3d(dim // 16, dim // 16, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm3d(dim // 16),
                nn.LeakyReLU(),
                nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
                nn.Conv3d(dim // 16, args.input_nc, kernel_size=1, stride=1),
            )
            
            # VQVAE Embedding
            # self.n_embedding = 256
            # self.vq_embedding = nn.Embedding(self.n_embedding, embedding_dim=dim)
            # self.vq_embedding.weight.data.uniform_(-1.0 / self.n_embedding, 1.0 / self.n_embedding)

    def forward(self, x):
        x_out = self.swinViT(x.contiguous())[4]
        b, c, h, w, d = x_out.shape
        x4_reshape = x_out.flatten(start_dim=2, end_dim=4)
        x4_reshape = x4_reshape.transpose(1, 2)
        x_rot = self.rotation_pre(x4_reshape[:, 0])
        x_rot = self.rotation_head(x_rot)
        x_contrastive = self.contrastive_pre(x4_reshape[:, 1])
        x_contrastive = self.contrastive_head(x_contrastive)
        x_rec = x_out.flatten(start_dim=2, end_dim=4)
        x_rec = x_rec.view(-1, c, h, w, d)
        x_rec = self.conv(x_rec)
        return x_rot, x_contrastive, x_rec
        # ze = self.swinViT(x.contiguous())[4]
        # embedding = self.vq_embedding.weight.data
        # b, c, h, w, d = ze.shape
        # K, _ = embedding.shape
        # embedding_broadcast = embedding.reshape(1, K, c, 1, 1, 1)
        # ze_broadcast = ze.reshape(b, 1, c, h, w, d)
        # distance = torch.sum((embedding_broadcast - ze_broadcast) ** 2, 2)
        # nearest_neighbour = torch.argmin(distance, 1)
        
        # zq = self.vq_embedding(nearest_neighbour).permute(0, 4, 1, 2, 3)
        # x_out = ze + (zq - ze).detach()
        
        # x4_reshape = x_out.flatten(start_dim=2, end_dim=4)
        # x4_reshape = x4_reshape.transpose(1, 2)
        # x_rot = self.rotation_pre(x4_reshape[:, 0])
        # x_rot = self.rotation_head(x_rot)
        # x_contrastive = self.contrastive_pre(x4_reshape[:, 1])
        # x_contrastive = self.contrastive_head(x_contrastive)
        # x_rec = x_out.flatten(start_dim=2, end_dim=4)
        # x_rec = x_rec.view(-1, c, h, w, d)
        # x_rec = self.conv(x_rec)
        # return x_rot, x_contrastive, x_rec, ze, zq
        
        


class NetworkLoader(object):
    def __init__(self, opt):
        self.opt = opt
        self.fname = osp.join(self.opt.expr_dir, 'run.log')

    def load(self)-> nn.Module:
        if self.opt.nid == 0:
            update_log(f'model architecture: DI2IN (nbase={self.opt.nbase})', self.fname)
            m = DI2IN(num_classes=self.opt.num_classes, nbase=self.opt.nbase)

        elif self.opt.nid == 1:
            update_log('model architecture: UNet (MONAI)', self.fname)
            m = UNet(
                spatial_dims=self.opt.dim,
                in_channels=self.opt.input_nc,
                out_channels=self.opt.num_classes,
                channels=(16, 32, 48, 64, 128),
                strides=(2, 2, 2, 2),
                num_res_units=2,
                norm=Norm.BATCH,
                dropout=self.opt.dropout)

        elif self.opt.nid == 2:
            assert self.opt.dim==3, f'nnFormer requires input dim to be 3, but get {self.opt.dim}'
            update_log('model architecture: nnFormer', self.fname)
            m = nnFormer(
                num_classes=self.opt.num_classes, 
                crop_size=self.opt.crop_size,
                embedding_dim=self.opt.embedding_dim,
                input_channels=self.opt.input_nc, 
                conv_op=nn.Conv3d, 
                depths=self.opt.depths,
                num_heads=self.opt.num_heads,
                patch_size=self.opt.patch_size,
                window_size=self.opt.window_size,
                deep_supervision=self.opt.do_ds)

        elif self.opt.nid == 3:
            update_log('model architecture: generic UNet from nnUNet', self.fname)
            m = Generic_UNet(
                input_channels=self.opt.input_nc, 
                base_num_features=self.opt.nbase, 
                num_classes=self.opt.num_classes, 
                num_pool=self.opt.num_pool,
                deep_supervision=self.opt.do_ds)

        elif self.opt.nid == 4:
            update_log(f'model architecture: DI2IN_DS (nbase={self.opt.nbase})', self.fname)
            m = DI2IN_DS(num_classes=self.opt.num_classes, nbase=self.opt.nbase)

        elif self.opt.nid == 5:
            update_log(f'model architecture: DoDNet (partial label learning))', self.fname)
            m = DoDNet(num_classes=self.opt.num_classes)

        elif self.opt.nid == 6:
            update_log('model architecture: generic UNet customized', self.fname)

            # Pancreas model
            # m = Generic_UNet(
            #     input_channels=1, 
            #     base_num_features=32, 
            #     num_classes=2, 
            #     num_pool=5,
            #     pool_op_kernel_sizes=[[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [1, 2, 2]],
            #     conv_kernel_sizes=[[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
            #     deep_supervision=self.opt.do_ds)
            
            # Whole_Bowel/Constrictor_Muscles model
            m = Generic_UNet(
                input_channels=1, 
                base_num_features=32, 
                num_classes=4, 
                num_pool=5,
                pool_op_kernel_sizes=[[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 1, 2]],
                conv_kernel_sizes=[[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
                deep_supervision=self.opt.do_ds)

        elif self.opt.nid == 7:
            update_log(f'model architecture: DI2IN_KD (nbase={self.opt.nbase})', self.fname)
            m = DI2IN_KD(num_classes=self.opt.num_classes, nbase=self.opt.nbase)

        elif self.opt.nid == 8:
            update_log(f'model architecture: DI2IN_BNF_ConvTrans (nbase={self.opt.nbase})', self.fname)
            m = DI2IN_BNF_ConvTrans(num_classes=self.opt.num_classes, nbase=self.opt.nbase)
            torch.backends.cudnn.deterministic = True

        elif self.opt.nid == 9:
            update_log(f'model architecture: DI2IN_BNF_ConvTrans_KD (nbase={self.opt.nbase})', self.fname)
            m = DI2IN_BNF_ConvTrans_KD(num_classes=self.opt.num_classes, nbase=self.opt.nbase)
            torch.backends.cudnn.deterministic = True

        elif self.opt.nid == 10:
            update_log(f'model architecture: DI2IN_BNF_ConvTrans_KD_V2 (nbase={self.opt.nbase})', self.fname)
            m = DI2IN_BNF_ConvTrans_KD_V2(num_classes=self.opt.num_classes, nbase=self.opt.nbase)
            torch.backends.cudnn.deterministic = True

        elif self.opt.nid == 11:
            update_log(f'model architecture: DeeplabV3+ (3D)', self.fname)
            update_log('Adapted from: https://github.com/ChoiDM/pytorch-deeplabv3plus-3D', self.fname)
            m = DeepLabV3_3D(
                num_classes=self.opt.num_classes,
                input_channels=self.opt.input_nc,
                resnet='resnet18_os16',
                last_activation=None)

        if self.opt.nid == 12:
            update_log(f'model architecture: DI2IN_L (nbase={self.opt.nbase})', self.fname)
            m = DI2IN_L(num_classes=self.opt.num_classes, nbase=self.opt.nbase)

        if self.opt.nid == 13:
            update_log(f'model architecture: Swin-UNETR (nbase={self.opt.nbase})', self.fname)
            m = SSLHead(self.opt)

        if self.opt.nid == 14:
            update_log(f'model architecture: SwinMM (nbase={self.opt.nbase})', self.fname)
            m = SSLHead(self.opt)

        elif self.opt.nid == 999:
            update_log('model architecture: DI2IN-32 (pre-training)', self.fname)
            m = DI2IN(num_classes=82, nbase=32)       
            checkpoint = torch.load(
                '/model_pool/selfSupervised/ctmultiorgan/pretrained_models_2022-03-15/DI2IN_32/best-loss_valid=0.2610-epoch=0087.ckpt')
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in checkpoint['state_dict'].items():
                name = k.replace("net.", "")
                new_state_dict[name] = v
            m.load_state_dict(new_state_dict, strict=False)
            print('successfully loaded the pretrained model weights')
            m.output = nn.Conv3d(m.output.in_channels, self.opt.num_classes, kernel_size=1, stride=1, padding=0, bias=True)

            if self.opt.ft_id == 0:
                update_log('we wil fine-tune all the model parameters', self.fname)
            elif self.opt.ft_id == 1:
                update_log('we wil fine-tune the parameters in the decoder', self.fname)
                for i, (name, param) in enumerate(m.named_parameters()):
                    if i < 40:
                        param.requires_grad = False
            elif self.opt.ft_id == 2:
                update_log('we wil fine-tune the parameters in the output layer', self.fname)
                for name, param in m.named_parameters():
                    if 'output' not in name:
                        param.requires_grad = False
            return m
                
        if not self.opt.load_ckpt:
            if self.opt.init == 'scratch':
                update_log('model initialization: Scratch', self.fname)
            elif self.opt.init == "kaiming":
                update_log('model initialization: Kaiming', self.fname)
                m.apply(InitWeights_He(1e-2))
            elif self.opt.init == 'xavier':
                update_log('model initialization: Xavier uniform', self.fname)
                m.apply(InitWeights_XavierUniform(1))
            else:
                raise NotImplementedError("This initialization method has not been implemented...")
        return m


class InitWeights_He(object):
    def __init__(self, neg_slope=1e-2):
        self.neg_slope = neg_slope

    def __call__(self, module):
        if isinstance(module, nn.Conv3d) or isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d) or isinstance(module, nn.ConvTranspose3d):
            module.weight = nn.init.kaiming_normal_(module.weight, a=self.neg_slope)
            if module.bias is not None:
                module.bias = nn.init.constant_(module.bias, 0)


class InitWeights_XavierUniform(object):
    def __init__(self, gain=1):
        self.gain = gain

    def __call__(self, module):
        if isinstance(module, nn.Conv3d) or isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d) or isinstance(module, nn.ConvTranspose3d):
            module.weight = nn.init.xavier_uniform_(module.weight, self.gain)
            if module.bias is not None:
                module.bias = nn.init.constant_(module.bias, 0)
