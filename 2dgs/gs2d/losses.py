import numpy as np
import torch
from pytorch_msssim import ssim as _ssim_pms


def ssim(img1, img2, window_size=11, size_average=True):
    # Expects [H, W, C] images, converts to [1, C, H, W]
    if img1.ndim == 3:
        img1 = img1.permute(2, 0, 1).unsqueeze(0)
        img2 = img2.permute(2, 0, 1).unsqueeze(0)
    return _ssim_pms(
        img1, img2,
        data_range=1.0,
        size_average=size_average,
        win_size=window_size,
    )


def diff_image(a, b):
    d = torch.abs(a - b)
    mag = torch.clamp(torch.norm(d, dim=-1, keepdim=True) / np.sqrt(3.0), 0.0, 1.0)
    return d, mag.repeat(1, 1, 3)

