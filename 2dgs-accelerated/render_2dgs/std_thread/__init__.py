import torch

from . import std_thread_bindings as _C


class RenderGaussians2D(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mus, sigmas, thetas, opacities, rgbs, H, W, n_sigma):
        input_arrays = [
            t.detach().contiguous().cpu().float().numpy()
            for t in (mus, sigmas, thetas, opacities, rgbs)
        ]
        img, out_raw = _C.render_gaussians_2d(H, W, *input_arrays, n_sigma)

        out_raw_tensor = torch.from_numpy(out_raw).clone()
        ctx.save_for_backward(mus, sigmas, thetas, opacities, rgbs, out_raw_tensor)
        ctx.H = H
        ctx.W = W
        ctx.n_sigma = n_sigma

        return torch.from_numpy(img)

    @staticmethod
    def backward(ctx, grad_output):
        mus, sigmas, thetas, opacities, rgbs, out_raw = ctx.saved_tensors
        H, W, n_sigma = ctx.H, ctx.W, ctx.n_sigma

        arrays = [
            t.detach().contiguous().cpu().float().numpy()
            for t in (mus, sigmas, thetas, opacities, rgbs, out_raw)
        ]
        grad_out_np = grad_output.detach().contiguous().cpu().float().numpy()

        grad_mus, grad_sigmas, grad_thetas, grad_opacities, grad_rgbs = (
            _C.render_gaussians_2d_backward(H, W, *arrays, grad_out_np, n_sigma)
        )

        return (
            torch.from_numpy(grad_mus),
            torch.from_numpy(grad_sigmas),
            torch.from_numpy(grad_thetas),
            torch.from_numpy(grad_opacities),
            torch.from_numpy(grad_rgbs),
            None, None, None,
        )


def render_gaussians_2d(H, W, mus, sigmas, thetas, opacities, rgbs, n_sigma=3.0):
    return RenderGaussians2D.apply(mus, sigmas, thetas, opacities, rgbs, H, W, n_sigma)


__all__ = ["render_gaussians_2d"]
