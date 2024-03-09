import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window(window_size, sigma=1.5, channel=1):
    _1D_window = gaussian(window_size, sigma).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(im1, im2, window):
    C, _, H, W = window.shape

    im1_pad = F.pad(im1, (H//2,H//2,W//2,W//2), 'replicate')
    im2_pad = F.pad(im2, (H//2,H//2,W//2,W//2), 'replicate')

    mu1 = F.conv2d(im1_pad, window, padding=0, groups=C)
    mu2 = F.conv2d(im2_pad, window, padding=0, groups=C)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(im1_pad*im1_pad, window, padding=0, groups=C) - mu1_sq
    sigma2_sq = F.conv2d(im2_pad*im2_pad, window, padding=0, groups=C) - mu2_sq
    sigma12 = F.conv2d(im1_pad*im2_pad, window, padding=0, groups=C) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    return {'mu1':mu1, 'mu2':mu2, 
            'sigma1_sq':sigma1_sq, 'sigma2_sq':sigma2_sq,
            'ssim':ssim_map}


class SSIM(nn.Module):
    def __init__(self, window_size=11, sigma=1.5):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.sigma = sigma

    def forward(self, im1, im2):
        assert (im1.shape ==im2.shape), f'{im1.shape} and {im2.shape} are not same!'

        self.window = create_window(self.window_size, channel=im1.shape[1]).to(im1.device)
        ssim_dict = _ssim(im1, im2, self.window)

        return ssim_dict

