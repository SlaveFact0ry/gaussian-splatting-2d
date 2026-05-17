# 2DGS — 2D Gaussian Splatting Image Fitting

회전·anisotropic 2D 가우시안들을 학습시켜 하나의 이미지를 근사하는 미니 프로젝트.
Adam + L1 + D-SSIM 손실 + density control (clone / split / prune)을 포함한다.

## 구조

```
2dgs/
├── 2dgs.ipynb          # 탐색용 노트북 (원본)
├── train.py            # 학습 진입점 (하이퍼파라미터는 상단 상수)
├── gs2d/               # 분리한 패키지
│   ├── render.py       # make_pixel_grid, render_gaussians_2d
│   ├── density.py      # prune / clone / split + optimizer state 수술
│   ├── losses.py       # ssim, diff_image
│   └── viz.py          # show_images, show_image_grid, save_training_video,
│                       # plot_gaussian_positions
└── outputs/            # 학습 산출물 (mp4 등). 런타임에 생성
```

노트북과 스크립트는 같은 로직을 공유한다. 코드 수정은 `gs2d/`에서, 실험은
`train.py` 상단 상수만 만져서 한다.

## 요구사항

- Python 3.10+
- `torch`, `numpy`, `opencv-python`, `matplotlib`, `pytorch-msssim`
- 비디오 저장용 `ffmpeg` (system binary)

## 실행

```bash
python train.py
```

기본적으로 `./soondol.jpg`를 타겟 이미지로 사용한다. 다른 이미지를 쓰려면
`train.py`의 `IMAGE_PATH`를 바꾼다.

## 주요 하이퍼파라미터 (`train.py`)

| 이름 | 기본값 | 설명 |
|---|---|---|
| `H`, `W` | 256 | 렌더 해상도 |
| `NUM_GAUSSIANS` | 20 | 초기 가우시안 개수 |
| `MAX_ITERS` | 1000 | 학습 iteration 수 |
| `LR` | 0.01 | Adam 학습률 |
| `LR_DECAY_RATE` | 0.995 | iteration당 LR 감쇠율 |
| `LR_DECAY_START` | 300 | LR 감쇠 시작 iteration |
| `DENSIFY_INTERVAL` | 30 | clone+split 주기 |
| `DENSIFY_END` | 300 | densification 종료 iteration |
| `PRUNE_INTERVAL` | 10 | pruning 주기 |
| `GRAD_THRESHOLD` | 0.0002 | clone 트리거 grad norm |
| `MAX_SIGMA_THRESHOLD` | 0.3 | split 트리거 sigma |
| `MIN_OPACITY_THRESHOLD` | 0.05 | prune 트리거 opacity |
| `LAMBDA_DSSIM` | 0.2 | L1 vs D-SSIM 가중치 |

## Performance notes

`bench.py`로 forward / backward + op 단위 측정 가능 (`python bench.py`).

### 초기 측정 → SSIM 교체 (N=20, 256×256, CPU)

학습 시간의 약 85%가 SSIM 내부의 `_slow_conv2d_backward` /
`_slow_conv2d_forward`에 쓰이고 있었음. backward가 forward의 2.9배로 컸던 진짜
원인은 `render_gaussians_2d`가 아니라 SSIM의 grouped depthwise conv2d가 CPU
MKL-DNN 경로를 못 받고 slow path로 떨어진 것.

custom SSIM 구현을 `pytorch-msssim`(separable 11×1 + 1×11)으로 교체:

| | 총 시간/iter | forward | backward | bwd/fwd | conv2d 비중 |
|---|---|---|---|---|---|
| Before | 372 ms | 95 ms | 277 ms | 2.92× | 85% |
| After  | 131 ms | 43 ms | 88 ms  | 2.03× | 50% |
| | **2.84× 빠름** | | | | |

### 다음 단계 (render 커널화의 정량 목표)

SSIM 교체 후 남은 `render_gaussians_2d` 비중이 ~42%로 가시화. bench.py 측정에서
다음 호출 카운트가 가속 백엔드의 정량 목표:

- `aten::bmm` **600회** (5 iter × 20 가우시안 × 6) → 1 fused 호출
- `aten::nonzero` **590회** (boolean indexing 내부) → 0 (tile-based culling)
- `aten::mul` **2455회**, 평균 7 µs → fused 커널에서 인라인

가속 백엔드(`render_2dgs/`) 작업 전에 `bench.py`로 베이스라인 재측정 권장.

## 산출물

- `outputs/gaussian_image_training.mp4` — 학습 과정 비디오
- 학습 중 matplotlib 창으로 표시되는 항목들:
  - 타겟 / 초기 / 학습 후 비교
  - 스냅샷 그리드 (지정된 iteration들)
  - Loss / LR 커브
  - 초기 vs 학습 후 가우시안 위치 + ellipse

![output.gif](./outputs/output.gif)
