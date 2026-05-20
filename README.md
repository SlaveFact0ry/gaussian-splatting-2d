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
render를 OpenMP / std::thread / CUDA로 가속.
forward + backward를 `torch.autograd.Function`으로 묶어 학습 루프에 통합.

#### `cpp_single` 백엔드 — 단일 스레드 C++ + nanobind ✅

dispatch 오버헤드 제거만으로 의미 있는 가속. 병렬화 없이.

| | forward | backward | bwd/fwd | 비고 |
|---|---|---|---|---|
| Python (SSIM 교체 후) | 43 ms | 88 ms | 2.03× | autograd traversal 포함 |
| **C++ cpp_single** | **9.17 ms** | **11.15 ms** | **1.22×** | fused 1-call forward + 1-call backward |
| 가속 배율 | **4.7×** | **7.9×** | | bwd가 더 큰 가속은 autograd 노드 traversal 사라진 결과 |

(N=20, 256×256, CPU. bench.py / test_parity 검증 통과.)

`single thread`인데 4.7×–7.9× 빠른 이유: 가우시안당 ~60 PyTorch op dispatch가
**1번의 C++ 함수 호출**로 줄면서 ~24 ms / iter의 dispatch 오버헤드 삭제

#### 다음 단계

- `std_thread/` — std::thread + thread pool 기반 멀티 스레드 (backward 자명, forward는 per-thread buffer 후 reduce)
- `cuda/` — GPU. 큰 N + 큰 해상도용.

### 3D Gaussian Splatting Pipeline
COLMAP 기반 카메라 포즈 추정 → 3DGS 학습 (7000 iterations) → novel view synthesis.
구현: [`3D Gaussian Splatting Colab.ipynb`](./3D%20Gaussian%20Splatting%20Colab.ipynb)
출처: [yassgan/3DGaussianSplatting-INRIA-Method-Colab](https://github.com/yassgan/3DGaussianSplatting-INRIA-Method-Colab)

### [stop_and_shoot/](./stop_and_shoot) — ROS2 / C++ 응용 *(진행중)*
