
import numpy as np
import torch
from torch import nn
from torch.nn.functional import avg_pool3d
from monai.transforms import AsDiscrete
from monai.data import decollate_batch


class MultipleOutputLoss(nn.Module):
    def __init__(self, loss, num_class, weight_factors=None):
        super(MultipleOutputLoss, self).__init__()
        self.weight_factors = weight_factors
        self.loss = loss
        self.num_class = num_class

    def ds_target(self, y):  # downsampling y
        # kernel_size = stride = (2, 2, 2)
        # pad = tuple((i-1) // 2 for i in kernel_size)
        # y = avg_pool3d(y, kernel_size, stride, pad, count_include_pad=False, ceil_mode=False)
        y = torch.nn.functional.interpolate(y, scale_factor=0.5, mode='nearest')
        return y
        

    def forward(self, x, y):
        assert isinstance(x, (tuple, list)), "x must be either tuple or list"
        if self.weight_factors is None:
            weights = [1] * len(x)
        else:
            weights = self.weight_factors

        l = 0
        for i in range(len(x)):
            # print(i, weights[i], x[i].size(), y.size())
            l += weights[i] * self.loss(x[i], y)
            if i < len(x)-1:
                y = self.ds_target(y)
        return l
