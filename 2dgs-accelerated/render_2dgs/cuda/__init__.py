import torch

from . import render_2dgs_cuda_bindings as _C
# backward는 2026-09까지 CPU fallback — std_thread 바인딩 재사용
from render_2dgs.std_thread import std_thread_bindings as _CPU


class RenderGaussians2DCUDA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mus, sigmas, thetas, opacities, rgbs, H, W, n_sigma):
        dev = torch.device("cuda")
        g = [t.detach().to(device=dev, dtype=torch.float32).contiguous()
             for t in (mus, sigmas, thetas, opacities, rgbs)]
        N = g[0].shape[0]

        img     = torch.empty((H, W, 3), dtype=torch.float32, device=dev)
        out_raw = torch.empty((H, W, 3), dtype=torch.float32, device=dev)

        _C.render_forward(
            H, W, N,
            *[t.data_ptr() for t in g],
            img.data_ptr(), out_raw.data_ptr(),
            float(n_sigma),
        )

        ctx.save_for_backward(mus, sigmas, thetas, opacities, rgbs,
                              out_raw.cpu())
        ctx.H, ctx.W, ctx.n_sigma = H, W, n_sigma
        ctx.in_device = mus.device
        return img

    @staticmethod
    def backward(ctx, grad_output):
        # NOTE: CPU fallback. CUDA backward는 2026-09 예정.
        mus, sigmas, thetas, opacities, rgbs, out_raw = ctx.saved_tensors
        arrays = [t.detach().contiguous().cpu().float().numpy()
                  for t in (mus, sigmas, thetas, opacities, rgbs, out_raw)]
        grad_np = grad_output.detach().contiguous().cpu().float().numpy()

        grads = _CPU.render_gaussians_2d_backward(
            ctx.H, ctx.W, *arrays, grad_np, ctx.n_sigma)

        return (*[torch.from_numpy(x).to(ctx.in_device) for x in grads],
                None, None, None)


def render_gaussians_2d(H, W, mus, sigmas, thetas, opacities, rgbs, n_sigma=3.0):
    return RenderGaussians2DCUDA.apply(
        mus, sigmas, thetas, opacities, rgbs, H, W, n_sigma)


__all__ = ["render_gaussians_2d"]
