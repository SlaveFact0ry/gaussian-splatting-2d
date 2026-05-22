# Gaussian Splatting Playground

<p align="center">
  <img src="2dgs/outputs/output.gif" width="480" alt="2D Gaussian Splatting fitting">
</p>

회전·anisotropic Gaussian Splatting을 직접 구현하고, 백엔드·응용까지 확장하는 작업 모음.

## Projects

### [2dgs/](./2dgs) — 2D Gaussian Splatting

- 회전·anisotropic 2D 가우시안 + density control (clone / split / prune) 구현
- torch.profiler 기반 병목 분석 → **SSIM (not render)** 이 진짜 병목임을 식별
- SSIM 구현 교체로 **2.84× speedup** (372 → 131 ms/iter on CPU)
- 다음: render 커널화 (forward + backward custom autograd)

[상세 README →](./2dgs/README.md)

### [2dgs-accelerated/](./2dgs-accelerated) — Multi-backend Renderer

2dgs의 프로파일링 결과(`bmm 600회`, `nonzero 590회` 등)를 정량 목표로 삼아
render를 단일 C++ + 멀티 스레드 + (예정) CUDA 로 가속.
forward + backward를 `torch.autograd.Function`으로 묶어 학습 루프에 통합.

#### `single_thread` 백엔드 — 단일 스레드 fused C++ + nanobind ✅

dispatch 오버헤드 제거만으로 의미 있는 가속. 병렬화 없이.

| | forward | backward | bwd/fwd |
|---|---:|---:|---|
| Python (SSIM 교체 후) | 43 ms | 88 ms | 2.03× |
| **C++ single_thread** (Release) | **1.22 ms** | **2.48 ms** | 2.03× |
| 가속 배율 | **35×** | **35×** | |

(N=20, 256×256, CPU.)

`single thread`인데 35× 빠른 이유: 가우시안당 ~60 PyTorch op dispatch가
**1번의 C++ 함수 호출**로 줄면서 ~24 ms / iter의 dispatch 오버헤드 삭제. (`-O3` Release 빌드 필수.)

#### `multi_thread` 백엔드 — ThreadPool + fork-join

가우시안 축을 멀티 스레드로 분할.
- **Backward**: 가우시안 슬롯 독립 → race-free, 자명한 병렬 (4-6× 가속).
- **Forward**: 픽셀 누적 race 회피 위해 per-thread buffer + reduce. N=20에선 alloc 오버헤드로 손해, N≥30 cross-over.

Render kernel 시간 (Release, H=W=256, 8 cores):

| N | single_thread (fwd / bwd) | multi_thread (fwd / bwd) | total speedup |
|---:|---:|---:|---:|
|   20 | 1.22 / 2.48 ms | 1.70 / 0.65 ms | 1.58× |
|  100 | 6.42 / 13.78 ms | 2.87 / 2.81 ms | 3.56× |
|  500 | 33.32 / 72.87 ms | 7.81 / 12.27 ms | 5.29× |
| 1000 | 66.91 / 146.19 ms | 13.69 / 24.49 ms | 5.58× |
| 2000 | 132.83 / 294.42 ms | 26.13 / 47.36 ms | **5.81×** (HW 한계) |

학습 시나리오 (density control 후 N=100-1000): **3.5×~5.6× 가속**.

#### 다음 단계

- `cuda/` — GPU. 큰 N + 큰 해상도용. `render_common.hpp` math 헬퍼는 `__host__ __device__`만 추가하면 그대로 이전.


### 3D Gaussian Splatting Pipeline
COLMAP 기반 카메라 포즈 추정 → 3DGS 학습 (7000 iterations) → novel view synthesis.
구현: [`3D Gaussian Splatting Colab.ipynb`](./3D%20Gaussian%20Splatting%20Colab.ipynb)
출처: [yassgan/3DGaussianSplatting-INRIA-Method-Colab](https://github.com/yassgan/3DGaussianSplatting-INRIA-Method-Colab)

### [stop_and_shoot/](./stop_and_shoot) — ROS2 / C++ 응용 *(진행중)*
