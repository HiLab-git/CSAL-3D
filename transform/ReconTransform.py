#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author      : Han Liu
# Date Created: 02/10/2022


import numpy as np
from typing import Tuple, Union, Callable, List
from monai.transforms import *
from .transform_zoo import IntensityNormd


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


class ReadImageOnlyd(MapTransform):
    def __init__(self, keys) -> None:
        MapTransform.__init__(self, keys)

    def __call__(self, data):
        fname = data['npz']
        data = dict(np.load(data['npz'], allow_pickle=True))
        del data['data_specs'], data['data_mask']
        data['subject'] = fname
        return data


class ReconTransform(object):
    def __init__(self, crop_size, num_samples, modality):
        self.__version__ = "0.1.1"
        self.crop_size = crop_size
        self.num_samples = num_samples
        self.modality = modality

        self.train = Compose([
            ReadImageOnlyd(keys=['npz']), 
            AddChanneld(keys=["data_image"]),
            CastToTyped(keys=["data_image"], dtype=np.int16),
            
            SpatialPadd(
                keys=["data_image"],
                spatial_size=self.crop_size,
                mode='constant'),

            IntensityNormd(
                keys=["data_image"],
                modality=self.modality),
            
            # cropping into patches
            RandSpatialCropd(
                keys=["data_image"],
                roi_size=self.crop_size,
                random_center=True,
                random_size=False),

            CastToTyped(
                keys=["data_image"], 
                dtype=np.float32),

            ToTensord(keys=["data_image"]),])

        self.infer = Compose([
            ReadImageOnlyd(keys=['npz']), 
            AddChanneld(
                keys=["data_image"],
                allow_missing_keys=True), 

            IntensityNormd(
                keys=["data_image"],
                modality=self.modality),

            CastToTyped(
                keys=["data_image"], 
                dtype=np.float32),

            ToTensord(
                keys=["data_image"]),])
