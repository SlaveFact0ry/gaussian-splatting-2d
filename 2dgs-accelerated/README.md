# 2DGS Accelerated

[2D Gaussian Splatting](https://github.com/SlaveFact0ry/2dgs) 의 render 백엔드를 C++로 가속한 구현.

원본 Python 구현(`gs2d.render`)과 동일한 결과를 내면서 dispatch 오버헤드를 제거해 forward/backward 모두 수 배 빠름.

---

## 구조

```
2dgs-accelerated/
├── render_2dgs/
│   ├── __init__.py                # 기본 백엔드(cpp_single) export
│   ├── cpp_single/                # 단일 스레드 C++ 백엔드 (현재 기본)
│   │   ├── __init__.py            # torch.autograd.Function 래퍼
│   │   ├── CMakeLists.txt
│   │   └── src/
│   │       ├── render_common.hpp        # 공통 타입 + inline 헬퍼
│   │       ├── render_forward.hpp/cpp   # forward 커널
│   │       ├── render_backward.hpp/cpp  # backward 커널
│   │       └── render_2dgs_bindings.cpp # nanobind 진입점
│   ├── std_thread/                # std::thread + thread pool 백엔드 (진행중)
│   └── cuda/                      # CUDA 백엔드 (예정)
├── tests/
│   └── test_parity.py             # Python ref와의 numerical parity
├── train.py                       # 학습 entry point
├── bench.py                       # forward/backward 시간 측정
└── docs/
    ├── autograd_integration.md    # C++ 백엔드와 autograd 통합 패턴
    └── backward_derivation.md     # backward 커널 수식 도출
```

---

## Quick Start

### 빌드

전제: `2dgs-accelerated/`와 같은 위치에 [원본 `2dgs/`](https://github.com/SlaveFact0ry/2dgs) 리포가 있어야 함 (parity 테스트, train의 reference).

```bash
cd render_2dgs/cpp_single
mkdir -p build && cd build
cmake ..
make -j
make install
```

`render_2dgs_bindings.cpython-XYZ-darwin.so` (또는 `.so`)가 `cpp_single/` 디렉토리에 설치됨.

### Parity 검증

```bash
python tests/test_parity.py
```

forward와 backward 모두 Python ref와 동일 결과 확인. 통과 예시:

```
forward PARITY OK  max_abs_diff = 5.811e-07
backward PARITY OK
  mus        max_abs_diff = 2.575e-05
  sigmas     max_abs_diff = 8.545e-04
  thetas     max_abs_diff = 6.035e-06
  opacities  max_abs_diff = 3.052e-05
  rgbs       max_abs_diff = 4.768e-06
```

### 벤치마크

```bash
python bench.py
```

forward / backward wall-clock + 상위 op 시간 + Chrome trace 출력.

### 학습

```bash
python train.py
```

---

## 사용 예 (Python)

원본 `gs2d.render_gaussians_2d`와 시그니처 호환:

```python
from render_2dgs import render_gaussians_2d

img = render_gaussians_2d(H, W, mus, sigmas, thetas, opacities, rgbs)
loss = ((img - target) ** 2).mean()
loss.backward()   # mus.grad 등 자동 채워짐
```

내부적으로 `torch.autograd.Function` 래퍼가 C++ forward/backward를 PyTorch autograd 그래프에 연결.

---

## 백엔드 로드맵

| 백엔드 | 상태 | 특징 |
|---|---|---|
| `cpp_single` | ✅ 완료 | 단일 스레드 fused C++. dispatch 제거가 핵심. |
| `std_thread` | 🚧 진행 | std::thread + thread pool. backward는 가우시안 축 자명, forward는 per-thread buffer 후 reduce. |
| `cuda` | 🔜 예정 | GPU. 큰 N + 큰 해상도용. |

각 백엔드는 `render_2dgs/<backend>/__init__.py`의 `render_gaussians_2d` 시그니처를 동일하게 유지.

---

## 의존성

- Python 3.10+
- PyTorch (CPU 또는 CUDA)
- CMake 3.15+
- C++17 컴파일러 (Apple Clang / GCC / MSVC)
- [nanobind 2.0+](https://github.com/wjakob/nanobind) — `FetchContent`로 자동 다운로드
- 학습/벤치 추가: `numpy`, `opencv-python`, `pytorch_msssim`

