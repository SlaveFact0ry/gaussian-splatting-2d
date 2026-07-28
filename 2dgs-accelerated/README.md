# 2DGS Accelerated

[../2dgs](../2dgs) 의 render 백엔드를 C++로 가속한 구현.

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
│   └── cuda/                      # CUDA pybind 템플릿 (add/multiply hello-world) — render 커널 미착수
├── tests/
│   └── test_parity.py             # Python ref와의 numerical parity (양쪽 backend)
├── train.py                       # 학습 entry point
├── bench.py                       # 단일 config + profiler
├── bench_sweep.py                 # 양쪽 backend × N sweep
└── docs/
    ├── autograd_integration.md    # C++ 백엔드와 autograd 통합 패턴
    ├── backward_derivation.md     # backward 커널 수식 도출
    ├── dispatch_overhead.md       # dispatch 비용 + fused 커널 원리
    └── cpu-profiling.md           # CPU 천장의 기계론 진단 (예정)
```

---

## Quick Start

### 빌드

전제: 같은 레포의 [../2dgs](../2dgs)가 parity 테스트와 train의 reference로 쓰인다.

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

`bench.py`의 forward 측정치는 render + L1 + SSIM을 합친 값이다(render 단독이 아님).
따라서 위 백엔드 비교 표의 canonical source는 render만 측정하는 `bench_sweep.py`다.

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

### Render kernel 시간 (Release build, H=W=256, Ryzen 5900X 12C/24T, 기본 T=24)

| N | cpp_single fwd / bwd / total | std_thread fwd / bwd / total | speedup fwd / bwd / total |
|---:|---:|---:|---:|
|   20 |   6.94 /   5.73 /   12.67 ms |  12.33 /  2.20 /  14.53 ms | 0.56× / 2.60× / 0.87× |
|   50 |  19.11 /  16.48 /   35.59 ms |  13.47 /  5.44 /  18.91 ms | 1.42× / 3.03× / 1.88× |
|  100 |  37.48 /  32.37 /   69.85 ms |  16.69 /  6.71 /  23.40 ms | 2.25× / 4.82× / 2.98× |
|  200 |  77.03 /  67.98 /  145.01 ms |  16.53 /  9.80 /  26.33 ms | 4.66× / 6.94× / 5.51× |
|  500 | 191.89 / 168.68 /  360.57 ms |  23.26 / 19.94 /  43.20 ms | 8.25× / 8.46× / 8.35× |
| 1000 | 378.93 / 333.12 /  712.06 ms |  33.36 / 32.01 /  65.37 ms | 11.36× / 10.41× / 10.89× |
| 2000 | 767.23 / 677.18 / 1444.41 ms |  50.82 / 62.32 / 113.14 ms | **15.10×** / 10.87× / 12.77× |

(`python bench_sweep.py`로 재현. 스레드 수는 `GS2D_NUM_THREADS`로 조절)

핵심 관찰:
- **Forward는 작은 N에서 오히려 손해**. 기본 T=24 기준 N=20에서 std_thread가
  cpp_single보다 느리다(0.56×). N=50부터 역전.
- **Backward는 모든 N에서 우위** (2.6× ~ 10.9×). 가우시안 슬롯이 독립이라
  reduce가 필요 없다.
- **최적 스레드 수가 N에 의존한다.** N=20이면 T=4, N=1000 이상이면 T=24가 가장
  빠르다 (`prof_T*.txt` 실측: N=20 total은 T=4에서 2.93ms로 최소, N=1000 total은
  T=24에서 32.47ms로 최소). 호출당 오버헤드가 T에 비례해 커지기 때문 —
  자세한 분해는 `docs/cpu-profiling.md` (문서 예정).
- N=2000에서 15.1×. 이전 M1 Pro 측정의 "5.81× = 하드웨어 한계"라는 결론은
  기계 교체 후 재측정으로 폐기되었다.

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
| `cuda` | 🔜 예정 | GPU. `add`/`multiply` pybind 템플릿만 존재 (`render_2dgs/cuda/`), render 커널은 미착수. `render_common.hpp` 헬퍼는 `__host__ __device__` 추가로 재사용 가능. |

각 백엔드는 `render_2dgs/<backend>/__init__.py`의 `render_gaussians_2d` 시그니처를 동일하게 유지.

---

### 환경 변수

| 변수 | 용도 |
|---|---|
| `GS2D_NUM_THREADS` | std_thread 백엔드의 스레드 수. 미지정 시 `hardware_concurrency()`. 풀은 프로세스당 1회만 생성되므로 스윕은 프로세스를 새로 띄워야 한다 |
| `GS2D_PROFILE` | 설정 시 forward의 구간별 시간(alloc / fill / reduce / clamp)을 stderr로 출력 |

### Compositing scope

현재 forward/backward는 **additive compositing**이다 — 가우시안 기여를 순서 무관하게
누적하고 마지막에 clamp한다. depth sort도 transmittance도 없다.
따라서 CUDA 백엔드도 정렬이 필요 없다.
sorted alpha-compositing + differentiable transmittance backward는 future work.

---

## 의존성

- Python 3.10+
- PyTorch (CPU 또는 CUDA)
- CMake 3.15+
- C++17 컴파일러 (GCC 권장, Linux)
- [nanobind 2.0+](https://github.com/wjakob/nanobind) — `FetchContent`로 자동 다운로드
- 학습/벤치 추가: `numpy`, `opencv-python`, `pytorch_msssim`
