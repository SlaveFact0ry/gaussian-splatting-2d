# 2DGS Accelerated

[2D Gaussian Splatting](https://github.com/SlaveFact0ry/2dgs) 의 render 백엔드를 C++로 가속한 구현.

원본 Python 구현(`gs2d.render`)과 동일한 결과를 내면서 dispatch 오버헤드 제거 + 멀티 스레드 병렬화로 큰 N에서 수~수십 배 빠름.

---

## 구조

```
2dgs-accelerated/
├── render_2dgs/
│   ├── __init__.py                # 기본 백엔드(std_thread) export
│   ├── cpp_single/                # 단일 스레드 C++ 백엔드
│   │   ├── __init__.py            # torch.autograd.Function 래퍼
│   │   ├── CMakeLists.txt
│   │   └── src/
│   │       ├── render_common.hpp        # 공통 타입 + inline 헬퍼
│   │       ├── render_forward.hpp/cpp   # forward 커널
│   │       ├── render_backward.hpp/cpp  # backward 커널
│   │       └── render_2dgs_bindings.cpp # nanobind 진입점
│   ├── std_thread/                # std::thread + thread pool 백엔드 (기본)
│   │   ├── __init__.py
│   │   ├── CMakeLists.txt
│   │   └── src/
│   │       ├── thread_pool.hpp/cpp     # ThreadPool + ParallelFor
│   │       ├── render_forward.cpp      # per-thread buffer + reduce
│   │       ├── render_backward.cpp     # 가우시안 축 분할 (race-free)
│   │       └── ...
│   └── cuda/                      # CUDA 백엔드 (예정)
├── tests/
│   └── test_parity.py             # Python ref와의 numerical parity (양쪽 backend)
├── train.py                       # 학습 entry point
├── bench.py                       # 단일 config + profiler
├── bench_sweep.py                 # 양쪽 backend × N sweep
└── docs/
    ├── autograd_integration.md    # C++ 백엔드와 autograd 통합 패턴
    ├── backward_derivation.md     # backward 커널 수식 도출
    └── dispatch_overhead.md       # dispatch 비용 + fused 커널 원리
```

---

## Quick Start

### 빌드

전제: `2dgs-accelerated/`와 같은 위치에 [원본 `2dgs/`](https://github.com/SlaveFact0ry/2dgs) 리포가 있어야 함 (parity 테스트, train의 reference).

**`CMAKE_BUILD_TYPE=Release` 기본 적용**돼 있음 (`-O3 -DNDEBUG`). debug build는 forward 10× 느림.

```bash
# 두 backend 모두 빌드
for backend in cpp_single std_thread; do
  rm -rf render_2dgs/$backend/build
  cmake -S render_2dgs/$backend -B render_2dgs/$backend/build \
        -DPython_EXECUTABLE=$(which python)
  cmake --build render_2dgs/$backend/build -j
  cmake --install render_2dgs/$backend/build
done
```

각 backend `.so`가 자기 디렉토리에 설치됨 (`render_2dgs/<backend>/<module>_bindings.cpython-XYZ-*.so`).

### Parity 검증

```bash
python tests/test_parity.py
```

cpp_single과 std_thread 둘 다 Python ref와 비교. 통과 예시:
```
forward PARITY OK
  cpp_single   max_abs_diff = 5.811e-07
  std_thread   max_abs_diff = ~1e-6

backward PARITY OK
  [cpp_single]
    mus        max_abs_diff = 2.575e-05
    sigmas     max_abs_diff = 8.545e-04
    ...
  [std_thread]
    ...
```

### 벤치마크

단일 config + op-level profiler:
```bash
python bench.py
```

Backend × N sweep:
```bash
python bench_sweep.py
```

### 학습

```bash
python train.py
```

기본 backend는 `std_thread` — `render_2dgs/__init__.py`에서 결정.
다른 backend로 명시 호출하려면 `from render_2dgs.cpp_single import render_gaussians_2d`.

---

## 사용 예 (Python)

원본 `gs2d.render_gaussians_2d`와 시그니처 호환:

```python
from render_2dgs import render_gaussians_2d   # 기본: std_thread

img = render_gaussians_2d(H, W, mus, sigmas, thetas, opacities, rgbs)
loss = ((img - target) ** 2).mean()
loss.backward()   # mus.grad 등 자동 채워짐
```

내부적으로 `torch.autograd.Function` 래퍼가 C++ forward/backward를 PyTorch autograd 그래프에 연결.

명시적 backend 선택:
```python
from render_2dgs.cpp_single import render_gaussians_2d   # 단일 스레드
from render_2dgs.std_thread import render_gaussians_2d   # 멀티 스레드 (기본)
```

---

## 성능

### Render kernel 시간 (Release build, H=W=256, M1 Pro 8 cores)

| N | single_thread fwd / bwd / total | std_thread fwd / bwd / total | std_thread speedup |
|---:|---:|---:|---:|
|   20 | 1.22 / 2.48 /   3.70 ms | 1.70 / 0.65 /   2.35 ms | 1.58× |
|   50 | 3.26 / 6.97 /  10.23 ms | 2.19 / 1.59 /   3.78 ms | 2.71× |
|  100 | 6.42 / 13.78 /  20.20 ms | 2.87 / 2.81 /   5.67 ms | 3.56× |
|  200 | 13.45 / 29.14 /  42.59 ms | 4.27 / 5.54 /   9.80 ms | 4.34× |
|  500 | 33.32 / 72.87 / 106.19 ms | 7.81 / 12.27 /  20.08 ms | 5.29× |
| 1000 | 66.91 / 146.19 / 213.10 ms | 13.69 / 24.49 /  38.18 ms | 5.58× |
| 2000 | 132.83 / 294.42 / 427.24 ms | 26.13 / 47.36 /  73.49 ms | **5.81×** |

(`python bench_sweep.py`로 재현)

핵심 관찰:
- **Forward cross-over는 N≈30**. N=20에선 single_thread이 28% 빠름 (per-thread buffer 할당 오버헤드). 그 이상에선 std_thread 우위.
- **Backward는 모든 N에서 std_thread 우위** (4-6×). 가우시안 축 자명 병렬화.
- **N=2000에서 5.81× = 8 cores × ~73% efficiency** — HW 한계 근접.
- 실제 학습 시나리오 (density control 후 N=100-1000)에서 **3.5×~5.6× 가속**.

### Python ref 대비 (N=20)

| | forward | backward | bwd/fwd |
|---|---:|---:|---:|
| Python (`gs2d.render`, SSIM 교체 후) | 43 ms | 88 ms | 2.03× |
| C++ single_thread (Release) | 1.22 ms | 2.48 ms | 2.03× |
| C++ std_thread (Release, 8 threads) | **1.70 ms** | **0.65 ms** | 0.38× |
| C++ single_thread vs Python | **35×** | **35×** | — |
| C++ multi_thread vs Python | **25×** | **135×** | — |

**Backward 135× 가속** 원리:
1. dispatch 제거 (Python 1200 op → 1 C++ 호출): ~10×
2. autograd traversal 제거 (graph 노드 1200개 → 0): ~5×
3. 가우시안 축 멀티 스레드 (8 cores × 73%): ~6×

---

## 가속의 원리

Python 버전은 가우시안당 ~60 PyTorch op dispatch — N=20이면 한 forward에 ~1200 dispatch.
각 dispatch는 dtype/device/autograd 체크 + 노드 생성 + 메모리 할당 → 가우시안당 수십 µs 손해.

C++ 백엔드는 같은 작업을 **단일 함수 호출**로 묶음:
- **single_thread**: 1 dispatch, autograd 노드 0개, 스택 float 연산
- **multi_thread**: 1 dispatch + ThreadPool fork-join (forward는 per-thread buffer, backward는 가우시안 축 분할)


---

## 백엔드 로드맵

| 백엔드 | 상태 | 특징 |
|---|---|---|
| `cpp_single` | ✅ 완료 | 단일 스레드 fused C++. dispatch 제거가 핵심. |
| `std_thread` | ✅ 완료 (기본) | std::thread + thread pool. backward 가우시안 축, forward per-thread buffer + reduce. |
| `cuda` | 🔜 예정 | GPU. 큰 N + 큰 해상도용. `__host__ __device__` 헬퍼는 재사용 가능. |

각 백엔드는 `render_2dgs/<backend>/__init__.py`의 `render_gaussians_2d` 시그니처를 동일하게 유지.

---

## 의존성

- Python 3.10+
- PyTorch (CPU 또는 CUDA)
- CMake 3.15+
- C++17 컴파일러 (Apple Clang / GCC / MSVC)
- [nanobind 2.0+](https://github.com/wjakob/nanobind) — `FetchContent`로 자동 다운로드
- 학습/벤치 추가: `numpy`, `opencv-python`, `pytorch_msssim`
