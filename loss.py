import torch
import numpy as np
from kornia.color import rgb_to_lab
from torchmetrics import MultiScaleStructuralSimilarityIndexMeasure as MSSSIM


class deepISPloss():
    def __init__(self, alpha=0.):
        self.alpha = alpha
        self.MSSSIM = MSSSIM()
        self.l1 = torch.nn.L1Loss()

    def __call__(self, x, target):
        lab_x = rgb_to_lab(x).float()
        lab_tar = rgb_to_lab(target).float()

        res = (1 - self.alpha) * self.l1(lab_x, lab_tar)
        # take only first channel to MS-SSIM
        # turned ssim off coz it doesnt work
        # res =     self.alpha  * (self.MSSSIM(lab_x[:, :1, :, :], lab_tar[:, :1, :, :]))

        return res
