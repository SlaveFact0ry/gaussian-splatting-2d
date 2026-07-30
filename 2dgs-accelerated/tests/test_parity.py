#  C++ render 백엔드와 Python 원본 (gs2d.render) 간 numerical parity 검증. - by claude-code
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "2dgs"))
sys.path.insert(0, str(REPO_ROOT / "2dgs-accelerated"))

import torch

from gs2d.render import render_gaussians_2d as ref_render
from render_2dgs.cpp_single import render_gaussians_2d as cpp_render_single
from render_2dgs.std_thread import render_gaussians_2d as cpp_render_thread
from render_2dgs.cuda import render_gaussians_2d as cuda_render

BACKENDS = {
    "cpp_single": cpp_render_single,
    "std_thread": cpp_render_thread,
    "cuda": cuda_render,
}


def _sample_inputs(N=64, seed=0):
    torch.manual_seed(seed)
    mus = torch.rand(N, 2) * 2 - 1                # NDC [-1, 1]
    sigmas = torch.rand(N, 2) * 0.1 + 0.01
    thetas = torch.rand(N) * 6.2832
    opacities = torch.rand(N)
    rgbs = torch.rand(N, 3)
    return mus, sigmas, thetas, opacities, rgbs


def test_forward_parity(H=128, W=128, N=64, rtol=1e-4, atol=1e-4):
    mus, sigmas, thetas, opacities, rgbs = _sample_inputs(N=N)
    img_ref = ref_render(H, W, mus, sigmas, thetas, opacities, rgbs)

    results = {}
    for backend_name, render in BACKENDS.items():
        img = render(H, W, mus, sigmas, thetas, opacities, rgbs)
        img = img.detach().cpu()
        assert img_ref.shape == img.shape, (
            f"[{backend_name}] shape mismatch: {img_ref.shape} vs {img.shape}"
        )
        max_diff = (img_ref - img).abs().max().item()
        results[backend_name] = max_diff
        assert torch.allclose(img_ref, img, rtol=rtol, atol=atol), (
            f"[{backend_name}] forward max_abs_diff = {max_diff:.3e} > atol={atol:.0e}"
        )
    return results


def test_backward_parity(H=64, W=64, N=16, atol=1e-3):
    inputs = _sample_inputs(N=N)

    inputs_ref = [t.clone().detach().requires_grad_(True) for t in inputs]
    out_ref = ref_render(H, W, *inputs_ref)
    grad_ref = torch.autograd.grad(out_ref.sum(), inputs_ref)

    names = ["mus", "sigmas", "thetas", "opacities", "rgbs"]
    results = {}
    for backend_name, render in BACKENDS.items():
        inputs_cpp = [t.clone().detach().requires_grad_(True) for t in inputs]
        out_cpp = render(H, W, *inputs_cpp)
        grad_cpp = torch.autograd.grad(out_cpp.sum(), inputs_cpp)

        max_diffs = {}
        for name, g_ref, g_cpp in zip(names, grad_ref, grad_cpp):
            diff = (g_ref - g_cpp).abs().max().item()
            max_diffs[name] = diff
            assert torch.allclose(g_ref, g_cpp, atol=atol), (
                f"[{backend_name}] {name} grad mismatch: "
                f"max_abs_diff = {diff:.3e} > atol={atol:.0e}"
            )
        results[backend_name] = max_diffs
    return results


if __name__ == "__main__":
    fwd_diffs = test_forward_parity()
    print("forward PARITY OK")
    for backend_name, d in fwd_diffs.items():
        print(f"  {backend_name:12s} max_abs_diff = {d:.3e}")

    print()
    bwd_diffs = test_backward_parity()
    print("backward PARITY OK")
    for backend_name, diffs in bwd_diffs.items():
        print(f"  [{backend_name}]")
        for name, d in diffs.items():
            print(f"    {name:10s} max_abs_diff = {d:.3e}")
