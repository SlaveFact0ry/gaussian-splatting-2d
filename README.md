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

### [2dgs-accelerated/](./2dgs-accelerated) — Multi-backend Renderer *(진행중)*

2dgs의 프로파일링 결과(`bmm 600회`, `nonzero 590회` 등)를 정량 목표로 삼아
render를 OpenMP / std::thread / CUDA / Metal로 가속.
forward + backward를 `torch.autograd.Function`으로 묶어 학습 루프에 통합.

### 3D Gaussian Splatting Pipeline *(미진행)*

COLMAP 기반 카메라 포즈 추정 → 3DGS 학습 (7000 iterations) → novel view synthesis.
구현: [`3D Gaussian Splatting Colab.ipynb`](./3D%20Gaussian%20Splatting%20Colab.ipynb)

### [stop_and_shoot/](./stop_and_shoot) — ROS2 / C++ 응용 *(진행중)*
