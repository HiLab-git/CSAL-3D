import os
import os.path as osp
from monai.transforms import *
import SimpleITK as sitk
from glob import glob
from tqdm import tqdm
import nibabel as nib
import numpy as np
from monai.transforms import *


class DebugTransform(Transform):
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, data):
        for key in self.keys:
            if key in data:
                print(f"{key} shape: {np.array(data[key]).shape}")
        return data

class AddChannel(Transform):
    """
    Adds a 1-length channel dimension to the input image.

    Most of the image transformations in ``monai.transforms``
    assumes the input image is in the channel-first format, which has the shape
    (num_channels, spatial_dim_1[, spatial_dim_2, ...]).

    This transform could be used, for example, to convert a (spatial_dim_1[, spatial_dim_2, ...])
    spatial image into the channel-first format so that the
    multidimensional image array can be correctly interpreted by the other
    transforms.
    """

    def __call__(self, img):
        """
        Apply the transform to `img`.
        """
        return img[None]

class AddChanneld(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.AddChannel`.
    """

    backend = AddChannel.backend

    def __init__(self, keys, allow_missing_keys: bool = False) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            allow_missing_keys: don't raise exception if key is missing.
        """
        super().__init__(keys, allow_missing_keys)
        self.adder = AddChannel()

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.adder(d[key])
        return d


def resample(data_dict, out_spacing, save_dir):
    return Compose([
        LoadImaged(keys=['image', 'label']),
        
        DebugTransform(keys=["image", "label"]),

        AddChanneld(keys=["image", "label"]),
        
        DebugTransform(keys=["image", "label"]),
        
        Spacingd(
            keys=['image', 'label'],
            pixdim=out_spacing,
            mode=("bilinear", "nearest")),
        
        DebugTransform(keys=["image", "label"]),

        SaveImaged(
            keys=['image'],
            output_dir=osp.join(save_dir, 'imagesTr_res'),
            output_postfix='',
            output_ext='.nii.gz',
            resample=False,
            separate_folder=False,
            print_log=False),

        SaveImaged(
            keys=['label'],
            output_dir=osp.join(save_dir, 'labelsTr_res'),
            output_postfix='',
            output_ext='.nii.gz',
            resample=False,
            separate_folder=False,
            print_log=False),

    ])(data_dict)


def resample_for_Brats(data_dict, out_spacing, save_dir):
    transforms = Compose([
        LoadImaged(keys=['image', 'label']),
    ])
    data_dict = transforms(data_dict)
    
    image = data_dict['image']  #  [240, 240, 155, 4]
    label = data_dict['label']  #  [240, 240, 155]

    assert image.shape[-1] == 4

    processed_modalities = []

    for i in range(4):
        single_modality_image = image[..., i]  #  [240, 240, 155]
        
        single_modality_dict = {'image': single_modality_image, 'label': label}
        
        single_modality_processed = Compose([
            AddChanneld(keys=["image", "label"]),
            Spacingd(
                keys=['image', 'label'],
                pixdim=out_spacing,
                mode=("bilinear", "nearest")),
        ])(single_modality_dict)
        
        processed_image = single_modality_processed['image'][0]  # [240, 240, 155]
        processed_modalities.append(processed_image)

    processed_image = np.stack(processed_modalities, axis=0)  # [4, 240, 240, 155]

    SaveImaged(
        keys=['image'],
        output_dir=osp.join(save_dir, 'imagesTr_res'),
        output_postfix='',
        output_ext='.nii.gz',
        resample=False,
        separate_folder=False,
        print_log=False)({'image': processed_image, 'label': label})

    SaveImaged(
        keys=['label'],
        output_dir=osp.join(save_dir, 'labelsTr_res'),
        output_postfix='',
        output_ext='.nii.gz',
        resample=False,
        separate_folder=False,
        print_log=False)({'image': processed_image, 'label': label})
    
# 
# data_dirs = ['Task01_BrainTumour']


# for data_dir in data_dirs:
#     img_paths = glob(osp.join(data_dir, 'imagesTr') + '/*.*')

#     # with tqdm(total=len(img_paths)) as pbar:
#     #     for path in img_paths:
#     #         msk_path = osp.join(data_dir, 'labelsTr', osp.basename(path))
#     #         data_dict = {'image': path, 'label': msk_path}
#     #         resample(data_dict, (1.5, 1.5, 2.0), data_dir)
#     #         pbar.update(1)
#     with tqdm(total=len(img_paths)) as pbar:
#         for path in img_paths:
#             print(path)
#             msk_path = osp.join(data_dir, 'labelsTr', osp.basename(path))
#             print(msk_path)
#             data_dict = {'image': path, 'label': msk_path}
#             resample_for_Brats(data_dict, (1.0, 1.0, 1.0), data_dir)
#             pbar.update(1)


data_dirs = ['Task01_BrainTumour']

save_dirs = ['../BrainTumour_Multi-Modality/data']

for i, data_dir in enumerate(data_dirs):
    if not osp.exists(save_dirs[i]):
        os.mkdir(save_dirs[i])

    img_paths = glob(osp.join(data_dir, 'imagesTr') + '/*.*')

    with tqdm(total=len(img_paths)) as pbar:
        for img_path in img_paths:
            msk_path = osp.join(data_dir, 'labelsTr', osp.basename(img_path))
            img = nib.load(img_path)
            specs = img.affine
            img = img.get_fdata()
            msk = nib.load(msk_path).get_fdata()
            npz_save_path = osp.join(save_dirs[i], f'{osp.basename(img_path)[:-7]}.npz')
            np.savez(npz_save_path, data_image=img, data_mask=msk, data_specs=specs)

            pbar.update(1)