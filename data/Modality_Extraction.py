import os
import numpy as np
import nibabel as nib
from glob import glob

def extract_flair_modality(image):
    """
    Extract FLAIR
    Args:
        image (numpy array):  (240, 240, 155, 4)。
    Returns:
        numpy array: FLAIR  (240, 240, 155)。
    """
    return image[..., 0]

def process_images(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 查找输入目录下所有 .nii.gz 文件
    image_paths = glob(os.path.join(input_dir, "*.nii.gz"))

    for image_path in image_paths:
        # 加载图像文件
        image_nii = nib.load(image_path)
        image_data = image_nii.get_fdata()

        # 提取 FLAIR 模态
        flair_image = extract_flair_modality(image_data)

        # 创建新的 NIfTI 文件对象
        flair_nii = nib.Nifti1Image(flair_image, image_nii.affine, image_nii.header)

        # 保存到输出目录
        output_path = os.path.join(output_dir, os.path.basename(image_path))
        nib.save(flair_nii, output_path)

        print(f"Processed and saved: {output_path}")

if __name__ == "__main__":
    input_dir = "./Task01_BrainTumour/imagesTr"
    output_dir = "./Task01_BrainTumour/imagesTr_flair"

    process_images(input_dir, output_dir)
