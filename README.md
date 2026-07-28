# 2D Gaussian Splatting — from scratch to CUDA

<p align="center">
  <img src="2dgs/outputs/output.gif" width="380" alt="2DGS fitting">
</p>

회전·anisotropic 2D Gaussian Splatting을 Python으로 처음부터 구현하고,
프로파일링으로 병목을 찾아 **C++ → 멀티스레드 → CUDA** 로 render 백엔드를
단계별로 가속한다.

목적은 빠른 렌더러 자체가 아니라 **측정 → 모델 → 검증**을 각 단계마다 반복한
기록이다. 모든 백엔드는 numerical parity를 유지하며, 성능 수치는 전부
같은 기계(Ryzen 5900X 12C/24T · RTX 3090)에서 측정했다.

## 진행

| 단계 | 상태 | 핵심 결과 |
|---|---|---|
| Python from-scratch (`2dgs/`) | ✅ | density control(clone/split/prune) + custom autograd 통합 |
| 프로파일링 | ✅ | **render가 아니라 SSIM이 병목**임을 식별 → 구현 교체로 2.84× (372 → 131 ms/iter) |
| C++ single-thread | ✅ | dispatch 오버헤드 제거 |
| C++ multi-thread | ✅ | N=2000에서 cpp_single 대비 **15.1×** |
| CPU 기계론 진단 | ✅ | forward 천장의 원인 규명 (문서 예정) |
| CUDA render | 🔜 | tile binning + shared-memory 누적 |

## 서브 프로젝트

- **[2dgs/](./2dgs)** — Python 원본 구현 + 학습 루프
- **[2dgs-accelerated/](./2dgs-accelerated)** — 다중 백엔드 렌더러. 성능 표·상세 문서는 여기

## Scope

Compositing은 **additive**(순서 독립)다. 따라서 GPU에서도 depth sort가 필요 없다.
sorted alpha-compositing과 differentiable transmittance backward는 future work.
