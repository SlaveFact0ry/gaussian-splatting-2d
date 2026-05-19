#  C++ render 백엔드와 Python 원본 (gs2d.render) 간 numerical parity 검증. - by claude-code
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "2dgs"))
sys.path.insert(0, str(REPO_ROOT / "2dgs-accelerated"))

import torch

from gs2d.render import render_gaussians_2d as ref_render
from render_2dgs import render_gaussians_2d as cpp_render


def _sample_inputs(N=64, seed=0):
    torch.manual_seed(seed)
    mus = torch.rand(N, 2) * 2 - 1                # NDC [-1, 1]
    sigmas = torch.rand(N, 2) * 0.1 + 0.01
    thetas = torch.rand(N) * 6.2832
    opacities = torch.rand(N)
    rgbs = torch.rand(N, 3)
    return mus, sigmas, thetas, opacities, rgbs


def test_parity(H=128, W=128, N=64, rtol=1e-4, atol=1e-4):
    mus, sigmas, thetas, opacities, rgbs = _sample_inputs(N=N)

    img_ref = ref_render(H, W, mus, sigmas, thetas, opacities, rgbs)
    img_cpp = cpp_render(H, W, mus, sigmas, thetas, opacities, rgbs)

    assert img_ref.shape == img_cpp.shape, f"shape mismatch: {img_ref.shape} vs {img_cpp.shape}"
    max_diff = (img_ref - img_cpp).abs().max().item()
    assert torch.allclose(img_ref, img_cpp, rtol=rtol, atol=atol), (
        f"max abs diff = {max_diff:.3e} > atol={atol:.0e}"
    )
    return max_diff


if __name__ == "__main__":
    max_diff = test_parity()
    print(f"PARITY OK  max_abs_diff = {max_diff:.3e}")
