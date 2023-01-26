import torch
from kornia.color import rgb_to_lab
from torchmetrics import MultiScaleStructuralSimilarityIndexMeasure as MSSSIM


def my_norm(x):
    # from [-1, 1) to [0, 1)
    return (x + 1) / 2


class deepISPloss():
    def __init__(self, alpha=0.5, norm=my_norm):
        self.alpha = alpha
        self.l1 = torch.nn.L1Loss()
        self.MSSSIM = MSSSIM()

        self.norm = norm

    def __call__(self, x, target):
        lab_x = rgb_to_lab(self.norm(x)).float()
        lab_tar = rgb_to_lab(self.norm(target)).float()

        res = (1 - self.alpha) * self.l1(lab_x, lab_tar)
        # take only first channel to MS-SSIM
        # turned ssim off coz it doesnt work
        # res +=     self.alpha  * (self.MSSSIM(lab_x[:, :1, :, :],
        #                                       lab_tar[:, :1, :, :]))

        return res
