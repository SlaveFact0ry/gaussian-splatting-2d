"""
Forward vs Backward 측정 전용 스크립트.

학습 루프의 불필요한 변수 (density control, video capture, optimizer step) 빼고
순수하게 render forward + backward만 N번 반복해서 비교한다.

출력:
- forward / backward 평균 ms
- backward / forward 비율
- 각각의 op 단위 상위 시간 (Self CPU)
- Chrome trace -> outputs/bench_trace.json
"""

import os
import time

import cv2
import numpy as np
import torch
from torch.profiler import profile, record_function, ProfilerActivity, schedule

from gs2d import render_gaussians_2d, ssim

# ---------- 측정 조건 (train.py와 동일하게) ----------
H, W = 256, 256
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_PATH = os.path.join(BASE_DIR, "soondol.jpg")

SEED = 42
NUM_GAUSSIANS = 20
LAMBDA_DSSIM = 0.2

WARMUP = 5
MEASURE = 20

OUTPUT_DIR = "outputs"
TRACE_PATH = os.path.join(OUTPUT_DIR, "bench_trace.json")


def load_target_image(path, H, W):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (H, W))
    return torch.tensor(img, dtype=torch.float32) / 255.0


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
    return (
        mus.clone().detach().requires_grad_(True),
        sigmas.clone().detach().requires_grad_(True),
        thetas.clone().detach().requires_grad_(True),
        op.clone().detach().requires_grad_(True),
        rgb.clone().detach().requires_grad_(True),
    )


def one_step(params, target):
    mus, sigmas, thetas, op, rgb = params
    for p in params:
        if p.grad is not None:
            p.grad.zero_()

    with record_function("forward"):
        rendered = render_gaussians_2d(H, W, mus, sigmas, thetas, op, rgb)
        l1 = torch.abs(rendered - target).mean()
        ssim_loss = 1.0 - ssim(rendered, target)
        loss = (1.0 - LAMBDA_DSSIM) * l1 + LAMBDA_DSSIM * ssim_loss

    with record_function("backward"):
        loss.backward()


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    target = load_target_image(IMAGE_PATH, H, W)
    params = make_params(NUM_GAUSSIANS, SEED)

    # ---------- Wall-clock 측정 (profiler 없는 깨끗한 시간) ----------
    print(f"Warmup {WARMUP} iters...")
    for _ in range(WARMUP):
        one_step(params, target)

    print(f"Measuring {MEASURE} iters (wall-clock)...")
    t_fwd_total = 0.0
    t_bwd_total = 0.0

    for _ in range(MEASURE):
        mus, sigmas, thetas, op, rgb = params
        for p in params:
            if p.grad is not None:
                p.grad.zero_()

        t0 = time.perf_counter()
        rendered = render_gaussians_2d(H, W, mus, sigmas, thetas, op, rgb)
        l1 = torch.abs(rendered - target).mean()
        ssim_loss = 1.0 - ssim(rendered, target)
        loss = (1.0 - LAMBDA_DSSIM) * l1 + LAMBDA_DSSIM * ssim_loss
        t1 = time.perf_counter()

        loss.backward()
        t2 = time.perf_counter()

        t_fwd_total += (t1 - t0)
        t_bwd_total += (t2 - t1)

    fwd_ms = t_fwd_total / MEASURE * 1000
    bwd_ms = t_bwd_total / MEASURE * 1000
    total_ms = fwd_ms + bwd_ms

    print()
    print(f"=== Wall-clock (N={NUM_GAUSSIANS}, H=W={H}) ===")
    print(f"forward  : {fwd_ms:8.3f} ms/iter  ({fwd_ms/total_ms*100:5.1f}%)")
    print(f"backward : {bwd_ms:8.3f} ms/iter  ({bwd_ms/total_ms*100:5.1f}%)")
    print(f"total    : {total_ms:8.3f} ms/iter")
    print(f"bwd/fwd  : {bwd_ms/fwd_ms:.2f}x")

    # ---------- Op-level 측정 (profiler) ----------
    print(f"\nRunning profiler for op-level breakdown...")
    prof_schedule = schedule(wait=1, warmup=2, active=5, repeat=1)

    with profile(
        activities=[ProfilerActivity.CPU],
        schedule=prof_schedule,
        on_trace_ready=lambda p: p.export_chrome_trace(TRACE_PATH),
        record_shapes=True,
        with_stack=False,
    ) as prof:
        for _ in range(1 + 2 + 5):
            one_step(params, target)
            prof.step()

    print(f"\n=== Top 15 ops by Self CPU (forward + backward combined) ===")
    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=15))

    print(f"\nChrome trace saved: {TRACE_PATH}")


if __name__ == "__main__":
    main()
