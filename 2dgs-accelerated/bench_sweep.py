"""
Backend sweep bench — cpp_single vs std_thread across N values.

bench.py는 단일 config + 프로파일러용. 이건 cross-over 지점 찾기용.

SSIM/L1 같은 도메인 비의존 시간 빼고 **render forward + render backward만** 측정.
backward는 `img.sum().backward()` — 가장 가벼운 형태로 render backward 트리거.

-- claude code --
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "2dgs"))

from gs2d.render import render_gaussians_2d as render_python
from render_2dgs.cpp_single import render_gaussians_2d as render_cpp_single
from render_2dgs.std_thread import render_gaussians_2d as render_std_thread

# ---------- 측정 조건 ----------
H, W = 256, 256
SEED = 42
N_VALUES = [20, 50, 100, 200, 500, 1000, 2000]

WARMUP = 3
MEASURE = 10

BACKENDS = {
    "python": render_python,
    "cpp_single": render_cpp_single,
    "std_thread": render_std_thread,
    # "cuda": render_cuda,
}


def init_random_gaussians(n, seed):
    torch.manual_seed(seed)
    mus = torch.randn(n, 2) * 0.5
    sigmas = torch.rand(n, 2) * 0.2 + 0.1
    thetas = torch.rand(n) * 2 * np.pi
    opacities = torch.rand(n) * 0.5 + 0.3
    rgbs = torch.rand(n, 3) * 0.6 + 0.2
    return mus, sigmas, thetas, opacities, rgbs


def make_params(n, seed):
    mus, sigmas, thetas, op, rgb = init_random_gaussians(n, seed)
    return [
        mus.clone().detach().requires_grad_(True),
        sigmas.clone().detach().requires_grad_(True),
        thetas.clone().detach().requires_grad_(True),
        op.clone().detach().requires_grad_(True),
        rgb.clone().detach().requires_grad_(True),
    ]


def measure(render_fn, N, warmup, measure_iters):
    params = make_params(N, SEED)

    def step():
        for p in params:
            if p.grad is not None:
                p.grad.zero_()
        t0 = time.perf_counter()
        img = render_fn(H, W, *params)
        t1 = time.perf_counter()
        img.sum().backward()
        t2 = time.perf_counter()
        return (t1 - t0), (t2 - t1)

    for _ in range(warmup):
        step()

    fwd_total = 0.0
    bwd_total = 0.0
    for _ in range(measure_iters):
        f, b = step()
        fwd_total += f
        bwd_total += b

    return (fwd_total / measure_iters * 1000,
            bwd_total / measure_iters * 1000)


def main():
    print(f"Render-only sweep (H=W={H}, warmup={WARMUP}, measure={MEASURE} iters)")
    print()

    header = (
        f"{'N':>5} |"
        f"       python fwd /  bwd /  total (ms)  |"
        f"   cpp_single fwd /  bwd /  total (ms)  |"
        f"   std_thread fwd /  bwd /  total (ms)  |"
        f"  speedup vs python  cpp_single / std_thread (total)"
    )
    print(header)
    print("-" * len(header))

    def sp(a, b):
        return a / b if b > 0 else float("inf")

    for N in N_VALUES:
        results = {}
        for name, render in BACKENDS.items():
            fwd, bwd = measure(render, N, WARMUP, MEASURE)
            results[name] = (fwd, bwd, fwd + bwd)

        py = results["python"]
        cs = results["cpp_single"]
        st = results["std_thread"]

        row = (
            f"{N:>5} |"
            f"   {py[0]:>8.3f} / {py[1]:>8.3f} / {py[2]:>8.3f}        |"
            f"   {cs[0]:>8.3f} / {cs[1]:>8.3f} / {cs[2]:>8.3f}        |"
            f"   {st[0]:>8.3f} / {st[1]:>8.3f} / {st[2]:>8.3f}        |"
            f"                     {sp(py[2], cs[2]):>6.2f}x / {sp(py[2], st[2]):>6.2f}x"
        )
        print(row)

    print()
    print("speedup vs python = python_total / backend_total (클수록 빠름)")
    print("cpp_single vs std_thread 세부 비교는 fwd/bwd 열을 직접 비교")


if __name__ == "__main__":
    main()
