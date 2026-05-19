import torch

from . import render_2dgs_bindings as _C


def render_gaussians_2d(H, W, mus, sigmas, thetas, opacities, rgbs, n_sigma=3.0):
    arrays = [
        t.detach().contiguous().cpu().float().numpy()
        for t in (mus, sigmas, thetas, opacities, rgbs)
    ]
    img = _C.render_gaussians_2d(H, W, *arrays, n_sigma)
    return torch.from_numpy(img)


__all__ = ["render_gaussians_2d"]
