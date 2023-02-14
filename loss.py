import torch
from kornia.color import rgb_to_lab
from torchmetrics import MultiScaleStructuralSimilarityIndexMeasure as MSSSIM

from utils import Norm


class deepISPloss():
    def __init__(self, device='cpu', alpha=0.5, norm_mode='simple'):
        self.alpha = alpha
        self.l1 = torch.nn.L1Loss()
        self.MSSSIM = MSSSIM().to(torch.device(device))

        self.norm_mode = norm_mode
        self.norm = Norm(mode='positive')

    def __call__(self, x, target):
        # get (max, min) vals that were used in dataset norm
        b = Norm.get_bounds_by_mode(self.norm_mode)
        x = self.norm(x, bounds_before=b)
        t = self.norm(target, bounds_before=b)

        lab_x = rgb_to_lab(x)
        lab_tar = rgb_to_lab(t)

        res = (1 - self.alpha) * self.l1(lab_x, lab_tar)
        # take only first channel to MS-SSIM
        res +=     self.alpha  * (self.MSSSIM(lab_x[:, :1, :, :],
                                              lab_tar[:, :1, :, :]))

        return res / 100
