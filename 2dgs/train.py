import os
from contextlib import nullcontext

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.profiler import profile, record_function, ProfilerActivity, schedule

from gs2d import (
    render_gaussians_2d,
    prune_gaussians,
    remove_from_optimizer,
    clone_gaussians_if_high_grad,
    split_gaussians_if_large,
    extend_optimizer_with_new_points,
    ssim,
    diff_image,
    show_images,
    show_image_grid,
    save_training_video,
    plot_gaussian_positions,
)


# ---------- Hyperparameters ----------
H, W = 256, 256
IMAGE_PATH = "soondol.jpg"

SEED = 42
NUM_GAUSSIANS = 20

MAX_ITERS = 1000
LR = 0.01
LR_DECAY_RATE = 0.995
LR_DECAY_START = 300

DENSIFY_INTERVAL = 30
DENSIFY_END = 300
PRUNE_INTERVAL = 10
GRAD_THRESHOLD = 0.0002
MAX_SIGMA_THRESHOLD = 0.3
MIN_OPACITY_THRESHOLD = 0.05
LAMBDA_DSSIM = 0.2

SNAPSHOT_ITERS = {
    0, 1, 10, 50, 80, 120, 160, 200, 250, 300, 350, 400, 450, 500,
    600, 700, 800, 900, 1000, 1200, 1500, 2000, 2500, 3000, 3500, 4000,
}
VIDEO_SAMPLE_RATE = 1
VIDEO_FPS = 30
 
OUTPUT_DIR = "outputs"
VIDEO_FILENAME = "gaussian_image_training.mp4"

# ---------- Profiling ----------
PROFILE = True
PROFILE_TRACE_PATH = os.path.join(OUTPUT_DIR, "trace.json")
PROFILE_WAIT = 2
PROFILE_WARMUP = 3
PROFILE_ACTIVE = 10


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


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    img_target = load_target_image(IMAGE_PATH, H, W)
    show_images([img_target], ["Target Image (RGB)"])

    init_mus, init_sigmas, init_thetas, init_opacities, init_rgbs = init_random_gaussians(NUM_GAUSSIANS, SEED)

    img_init = render_gaussians_2d(H, W, init_mus, init_sigmas, init_thetas, init_opacities, init_rgbs)
    _, d_mag_img = diff_image(img_init, img_target)
    show_images(
        [img_target, img_init, d_mag_img],
        ["Target Image", f"Initial ({NUM_GAUSSIANS} Gaussians)", "|Target - Initial|"],
        figsize=(12, 4),
    )

    mus_param = init_mus.clone().detach().requires_grad_(True)
    sigmas_param = init_sigmas.clone().detach().requires_grad_(True)
    thetas_param = init_thetas.clone().detach().requires_grad_(True)
    opacities_param = init_opacities.clone().detach().requires_grad_(True)
    rgbs_param = init_rgbs.clone().detach().requires_grad_(True)

    optimizer = torch.optim.Adam(
        [mus_param, sigmas_param, thetas_param, opacities_param, rgbs_param], lr=LR
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=LR_DECAY_RATE)

    history_loss = []
    history_lr = []
    snapshots = [(0,
                  mus_param.detach().clone(),
                  sigmas_param.detach().clone(),
                  thetas_param.detach().clone(),
                  opacities_param.detach().clone(),
                  rgbs_param.detach().clone())]
    video_frames = [img_init.detach().cpu().numpy()]

    if PROFILE:
        prof_ctx = profile(
            activities=[ProfilerActivity.CPU],
            schedule=schedule(wait=PROFILE_WAIT, warmup=PROFILE_WARMUP,
                              active=PROFILE_ACTIVE, repeat=1),
            on_trace_ready=lambda p: p.export_chrome_trace(PROFILE_TRACE_PATH),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        )
    else:
        prof_ctx = nullcontext()

    with prof_ctx as prof:
        for it in range(MAX_ITERS):
            optimizer.zero_grad()

            with torch.no_grad():
                mus_param.clamp_(-1.5, 1.5)
                sigmas_param.clamp_(0.01, 1.0)
                opacities_param.clamp_(0.0, 1.0)
                rgbs_param.clamp_(0.0, 1.0)

            with record_function("forward"):
                rendered = render_gaussians_2d(
                    H, W, mus_param, sigmas_param, thetas_param, opacities_param, rgbs_param
                )

            with record_function("loss"):
                l1_loss = torch.abs(rendered - img_target).mean()
                ssim_loss = 1.0 - ssim(rendered, img_target)
                loss = (1.0 - LAMBDA_DSSIM) * l1_loss + LAMBDA_DSSIM * ssim_loss

            with record_function("backward"):
                loss.backward()
                mus_grad = mus_param.grad.clone() if mus_param.grad is not None else None

            with record_function("optimizer_step"):
                optimizer.step()
                if it >= LR_DECAY_START:
                    scheduler.step()

            with record_function("density_control"):
                if (it + 1) % DENSIFY_INTERVAL == 0 or (it + 1) % PRUNE_INTERVAL == 0:
                    with torch.no_grad():
                        if (it + 1) % DENSIFY_INTERVAL == 0 and it < DENSIFY_END:
                            prev_count = mus_param.shape[0]

                            clone_full = clone_gaussians_if_high_grad(
                                mus_param, sigmas_param, thetas_param, opacities_param, rgbs_param,
                                mus_grad, grad_threshold=GRAD_THRESHOLD, max_sigma=MAX_SIGMA_THRESHOLD,
                            )

                            split_full = split_gaussians_if_large(
                                clone_full[0], clone_full[1], clone_full[2], clone_full[3], clone_full[4],
                                max_sigma=MAX_SIGMA_THRESHOLD,
                            )

                            mus_param, sigmas_param, thetas_param, opacities_param, rgbs_param = extend_optimizer_with_new_points(
                                optimizer,
                                (mus_param, sigmas_param, thetas_param, opacities_param, rgbs_param),
                                split_full,
                                prev_count,
                            )
                            mus_param.requires_grad_(True)
                            sigmas_param.requires_grad_(True)
                            thetas_param.requires_grad_(True)
                            opacities_param.requires_grad_(True)
                            rgbs_param.requires_grad_(True)

                        if (it + 1) % PRUNE_INTERVAL == 0:
                            prune_result = prune_gaussians(
                                mus_param, sigmas_param, thetas_param, opacities_param, rgbs_param,
                                min_opacity=MIN_OPACITY_THRESHOLD, max_sigma=None,
                            )
                            _, _, _, _, _, keep_mask = prune_result

                            if keep_mask.sum() < len(mus_param):
                                mus_param, sigmas_param, thetas_param, opacities_param, rgbs_param = remove_from_optimizer(
                                    optimizer,
                                    (mus_param, sigmas_param, thetas_param, opacities_param, rgbs_param),
                                    keep_mask,
                                )
                                mus_param.requires_grad_(True)
                                sigmas_param.requires_grad_(True)
                                thetas_param.requires_grad_(True)
                                opacities_param.requires_grad_(True)
                                rgbs_param.requires_grad_(True)

            history_loss.append(loss.item())
            history_lr.append(scheduler.get_last_lr()[0])

            if (it + 1) in SNAPSHOT_ITERS:
                snapshots.append((it + 1,
                                  mus_param.detach().clone(),
                                  sigmas_param.detach().clone(),
                                  thetas_param.detach().clone(),
                                  opacities_param.detach().clone(),
                                  rgbs_param.detach().clone()))

            with record_function("video_capture"):
                if (it + 1) % VIDEO_SAMPLE_RATE == 0:
                    frame = render_gaussians_2d(
                        H, W, mus_param, sigmas_param, thetas_param, opacities_param, rgbs_param
                    ).detach().cpu().numpy()
                    video_frames.append(frame)

            if (it + 1) % 50 == 0:
                current_lr = scheduler.get_last_lr()[0]
                print(f"Iteration {it+1}/{MAX_ITERS}, Loss: {loss.item():.6f}, "
                      f"LR: {current_lr:.6f}, Gaussians: {mus_param.shape[0]}")

            if PROFILE:
                prof.step()

    if PROFILE:
        print("\n=== torch.profiler summary (top 20 by CPU time) ===")
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))
        print(f"\nChrome trace saved: {PROFILE_TRACE_PATH}")
        print("View at https://ui.perfetto.dev or chrome://tracing")

    final_frame = render_gaussians_2d(
        H, W, mus_param, sigmas_param, thetas_param, opacities_param, rgbs_param
    ).detach().cpu().numpy()
    video_frames.append(final_frame)

    print(f"Final loss: {history_loss[-1]:.6f}")
    print(f"Final learning rate: {history_lr[-1]:.6f}")
    print(f"Final number of Gaussians: {mus_param.shape[0]} (started with {NUM_GAUSSIANS})")

    # ---------- Final visualization ----------
    img_trained = render_gaussians_2d(H, W, mus_param, sigmas_param, thetas_param, opacities_param, rgbs_param)
    _, d_mag_t = diff_image(img_target, img_trained)
    show_images(
        [img_target, img_init, img_trained, d_mag_t],
        ["Target Image", "Initial", "Trained", "|Target - Trained|"],
        figsize=(16, 4),
    )

    # Snapshot grid
    seen = set()
    snapshots_unique = []
    for snap in snapshots:
        if snap[0] not in seen:
            snapshots_unique.append(snap)
            seen.add(snap[0])
    snapshots_unique.sort(key=lambda t: t[0])

    snap_imgs, snap_titles = [], []
    for it, mus, sigmas, thetas, opacities, rgbs in snapshots_unique:
        snap_imgs.append(render_gaussians_2d(H, W, mus, sigmas, thetas, opacities, rgbs))
        snap_titles.append(f"iter {it}")
    show_image_grid(snap_imgs, snap_titles, cols=4, cell_size=3.0)

    # Loss + LR curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
    ax1.plot(history_loss)
    ax1.set_title("Loss vs Iteration (Image Fitting with Density Control)")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("L1 + D-SSIM Loss")
    ax1.grid(True)
    ax2.plot(history_lr, color="orange")
    ax2.set_title("Learning Rate Decay")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Learning Rate")
    ax2.grid(True)
    plt.tight_layout()
    plt.show()

    print("=== Optimization Results (Image Fitting with Rotation) ===")
    print(f"Number of Gaussians: {NUM_GAUSSIANS}")
    print(f"Initial loss: {history_loss[0]:.6f}")
    print(f"Final loss: {history_loss[-1]:.6f}")
    print(f"Improvement: {(1 - history_loss[-1] / history_loss[0]) * 100:.2f}%")
    print(f"Initial LR: {history_lr[0]:.6f}, Final LR: {history_lr[-1]:.6f}")

    # Video
    video_path = os.path.join(OUTPUT_DIR, VIDEO_FILENAME)
    print(f"Creating video with {len(video_frames)} frames...")
    save_training_video(
        video_frames, video_path, fps=VIDEO_FPS,
        title=f"Image Fitting with {NUM_GAUSSIANS} Gaussians",
    )

    # Gaussian positions: initial vs trained
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    plot_gaussian_positions(ax1, init_mus, init_sigmas, init_thetas, img_init,
                            color="red", title="Initial Gaussian Positions (with rotation)")
    plot_gaussian_positions(ax2, mus_param, sigmas_param, thetas_param, img_trained,
                            color="blue", title="Trained Gaussian Positions (with rotation)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
