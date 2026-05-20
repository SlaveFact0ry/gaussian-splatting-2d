# `std_thread` 백엔드 — 설계 + 학습 가이드

- 날짜: 2026-05-20
- 대상: `render_2dgs/std_thread/` 신규 백엔드
- 사용자 결정:
  - 디렉토리: `std_thread/` (cpp_single 미러)
  - Backward: 가우시안 축 분할 (race-free)
  - Forward: per-thread buffer + reduce
  - Thread pool: 매번 spawn이 아니라 풀 사용
  - 스레드 수: `std::thread::hardware_concurrency()` 기본
- 비교 기준: cpp_single 코드 + [backward_derivation.md §4.bis](../../docs/backward_derivation.md)

이 문서는 C++ 멀티스레딩 학습 + 우리 백엔드 설계를 한 곳에 정리한다.
사용자가 이 문서 기반으로 작업본을 만들고, 이후 diff 리뷰로 다듬는다.

---

## §0 — 큰 그림

cpp_single이 dispatch 오버헤드를 죽이고 single-thread fused 커널로 forward 9.17ms / backward 11.15ms를 냈다. 이제 같은 CPU의 **여러 코어**를 활용해 더 빠르게.

목표: T코어에서 ~T배 가까운 가속 (현실적으로 ~0.6T-0.8T배). 우리 워크로드는 임의 의존성 없는 가우시안 축 병렬 → 잘 scale될 것.

---

## §1 — C++ 멀티스레딩 기본 4종 세트

### 1.1 `std::thread`

스레드 시작 + 종료의 가장 기본 API.

```cpp
#include <thread>

std::thread t([]() {
    // 새 스레드에서 실행될 코드
    std::cout << "hello from thread\n";
});

t.join();   // 끝나길 기다림
// 또는 t.detach(); — 백그라운드로 보내고 잊음 (위험, 거의 안 씀)
```

- 람다 (또는 함수 포인터, functor) 받아서 새 스레드에서 호출
- `join()` 호출 안 하고 destructor 도달하면 `std::terminate()` 호출 (프로그램 죽음)
- spawn 비용: 50-200µs (macOS pthread 기준)

### 1.2 `std::mutex`

공유 데이터 보호. 한 번에 한 스레드만 lock 가능.

```cpp
#include <mutex>

std::mutex mu;
int shared_count = 0;

void increment() {
    std::lock_guard<std::mutex> lock(mu);   // RAII: 스코프 끝나면 자동 unlock
    shared_count++;
}
```

- `lock_guard`: RAII unlock 보장 (예외 던져도 해제됨)
- `unique_lock`: 더 유연. `condition_variable`과 같이 쓸 때 필수
- 직접 `mu.lock()` / `mu.unlock()`은 위험 (예외 시 leak)

### 1.3 `std::condition_variable`

스레드 간 신호. "조건이 충족됐을 때 깨워줘" 패턴.

```cpp
#include <condition_variable>

std::mutex mu;
std::condition_variable cv;
bool ready = false;

// 워커 스레드 (대기)
void worker() {
    std::unique_lock<std::mutex> lock(mu);
    cv.wait(lock, []{ return ready; });   // ready가 true 될 때까지 잠
    // 깨어남. mu는 자동으로 재취득된 상태.
}

// 메인 스레드 (신호)
void main_thread() {
    {
        std::lock_guard<std::mutex> lock(mu);
        ready = true;
    }
    cv.notify_one();   // 또는 notify_all()
}
```

핵심:
- `cv.wait(lock, predicate)`: predicate가 true가 될 때까지 잠. 자는 동안 lock은 풀려 있음 (다른 스레드가 mu를 잡을 수 있음). 깨면 lock 재취득.
- `cv.notify_one()` / `notify_all()`: 자고 있는 스레드 중 1개 / 전부 깨움.
- **predicate가 있는 wait 형태가 정답** — spurious wakeup (조건 충족 안 됐는데 깨는 OS 버그) 대비.

### 1.4 `std::atomic`

락 없이 단일 변수의 원자적 read/write.

```cpp
#include <atomic>

std::atomic<int> counter{0};
counter.fetch_add(1);   // ++counter, 원자적
int v = counter.load();
counter.store(42);
```

- 락보다 빠름 (CPU 명령어 한두 개)
- 단일 변수에만 가능. 여러 변수 일관성은 보장 안 됨
- 우리 코드에서는 task counter 등에 씀

---

## §2 — Thread Pool 설계

매번 `std::thread`를 spawn하면 호출당 50-200µs 오버헤드. 학습 1000 iter면 누적 50-200ms 헛돈.

**Thread pool**: T개의 워커 스레드를 처음에 한 번 만들고 **계속 살려둠**. 작업은 큐를 통해 던지고 결과만 받음.

```text
┌─ 메인 ─┐                 ┌─ 워커 0 ─┐  ┌─ 워커 1 ─┐  ┌─ 워커 N ─┐
│        │                 │          │  │          │  │          │
│ submit ├──→ [task queue] │ pop      │  │ pop      │  │ pop      │
│ work   │                 │ execute  │  │ execute  │  │ execute  │
│        │←── [done flag]──│ ...      │  │ ...      │  │ ...      │
└────────┘                 └──────────┘  └──────────┘  └──────────┘
```

### 2.1 핵심 구조

```cpp
class ThreadPool {
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    std::mutex queue_mu;
    std::condition_variable queue_cv;

    std::atomic<int> active_count{0};       // 실행 중인 task 수
    std::condition_variable done_cv;
    std::mutex done_mu;

    bool stop = false;

public:
    explicit ThreadPool(size_t n);
    ~ThreadPool();

    // N개 task를 큐잉 + 전부 끝날 때까지 대기
    void parallel_for(int total, std::function<void(int start, int end, int tid)> fn);

    int num_threads() const { return workers.size(); }
};
```

### 2.2 워커 루프

```cpp
ThreadPool::ThreadPool(size_t n) {
    for (size_t i = 0; i < n; ++i) {
        workers.emplace_back([this, i]() {
            while (true) {
                std::function<void()> task;
                {
                    std::unique_lock<std::mutex> lock(queue_mu);
                    queue_cv.wait(lock, [this]{ return stop || !tasks.empty(); });
                    if (stop && tasks.empty()) return;
                    task = std::move(tasks.front());
                    tasks.pop();
                }
                task();
                if (--active_count == 0) {
                    std::lock_guard<std::mutex> dlock(done_mu);
                    done_cv.notify_all();
                }
            }
        });
    }
}
```

각 워커는:
1. 큐가 비어있지 않을 때까지 대기
2. task 1개 꺼냄
3. 락 풀고 task 실행
4. `active_count` 감소, 0이 되면 메인 깨움
5. 반복

### 2.3 fork-join: `parallel_for`

```cpp
void ThreadPool::parallel_for(int total, std::function<void(int, int, int)> fn) {
    int T = workers.size();
    int chunk = (total + T - 1) / T;

    active_count = T;
    {
        std::lock_guard<std::mutex> lock(queue_mu);
        for (int t = 0; t < T; ++t) {
            int start = t * chunk;
            int end = std::min(start + chunk, total);
            tasks.emplace([fn, start, end, t]() { fn(start, end, t); });
        }
    }
    queue_cv.notify_all();

    std::unique_lock<std::mutex> dlock(done_mu);
    done_cv.wait(dlock, [this]{ return active_count == 0; });
}
```

- `total`을 T개 chunk로 분할
- 각 chunk는 `[start, end)` + thread id `tid`로 fn 호출
- 모든 chunk 완료될 때까지 메인 대기

### 2.4 소멸자 — 워커 정리

```cpp
ThreadPool::~ThreadPool() {
    {
        std::lock_guard<std::mutex> lock(queue_mu);
        stop = true;
    }
    queue_cv.notify_all();
    for (auto &w : workers) w.join();
}
```

`stop` flag 세팅 + 전부 깨움 → 워커들이 자기 루프 끝내고 빠져나옴 → join.

### 2.5 글로벌 싱글톤

전 프로세스에서 한 풀만 쓰면 됨:

```cpp
inline ThreadPool& global_pool() {
    static ThreadPool pool(std::thread::hardware_concurrency());
    return pool;
}
```

- `static`은 C++11 이후 thread-safe (Magic statics)
- 첫 호출에 lazy init, 프로세스 종료 시 소멸자

---

## §3 — Backward 병렬화 (자명)

[backward_derivation.md §4.bis](../../docs/backward_derivation.md): 각 가우시안 i의 backward 결과는 **자기 입력 슬롯에만** 쓴다 (`grad_mus[i]`, `grad_sigmas[i]`, ...). 다른 가우시안과 메모리 겹치지 않음 → race 없음.

```cpp
void render_gaussian_2d_backward(int H, int W, int N, ..., float *grad_mus, ...) {
    global_pool().parallel_for(N, [&](int start, int end, int tid) {
        for (int i = start; i < end; ++i) {
            Gaussian2D g_in = ...;   // mus, sigmas 등에서 i번째 추출
            Vec2 g_mu, g_sigma;
            float g_theta, g_op;
            Vec3 g_rgb;
            render_gaussian_2d_backward_one(H, W, g_in, out_raw, grad_out, n_sigma,
                                            g_mu, g_sigma, g_theta, g_op, g_rgb);
            // grad_mus[i], grad_sigmas[i] 등에 저장 (각자 자기 슬롯)
        }
    });
}
```

cpp_single의 backward와 거의 동일. `for (int i = 0; ...)` → `parallel_for(N, [...])`로만 바뀜.

`tid`는 backward에선 안 씀 (per-thread state 불필요). 시그니처 일관성 위해 받음.

---

## §4 — Forward 병렬화 (per-thread buffer + reduce)

Forward는 가우시안들이 **같은 픽셀에 누적**. 같은 픽셀에 두 스레드가 `+=` 하면 race.

해결: 각 스레드가 자기 H·W·3 버퍼를 가짐. 작업 끝나면 메인이 합침.

### 4.1 알고리즘

```cpp
void render_gaussian_2d(int H, int W, int N, ..., float *out_raw, float *output_image, float n_sigma) {
    auto& pool = global_pool();
    int T = pool.num_threads();

    // 1. per-thread buffer 할당 + 0으로 초기화
    std::vector<std::vector<float>> per_thread(T, std::vector<float>(H*W*3, 0.0f));

    // 2. 가우시안을 T개 chunk로 분할, 각 chunk를 thread가 자기 buffer에 누적
    pool.parallel_for(N, [&](int start, int end, int tid) {
        float *buf = per_thread[tid].data();
        for (int i = start; i < end; ++i) {
            Gaussian2D g = ...;
            // cpp_single forward의 가우시안 i 처리 로직 그대로,
            // 누적 대상만 out_raw → buf
            PixelRange r = bbox_pixel_range(g, W, H, n_sigma);
            for (int y = r.iy0; y <= r.iy1; y++) {
                for (int x = r.ix0; x <= r.ix1; x++) {
                    // ... value 계산 ...
                    int idx = (y * W + x) * 3;
                    buf[idx]     += value * g.rgb.r;
                    buf[idx + 1] += value * g.rgb.g;
                    buf[idx + 2] += value * g.rgb.b;
                }
            }
        }
    });

    // 3. T개 버퍼를 합쳐 out_raw에 (단일 스레드 reduce — H·W·3 크기 작아서 OK)
    std::memset(out_raw, 0, sizeof(float) * H * W * 3);
    for (int t = 0; t < T; ++t) {
        for (int i = 0; i < H * W * 3; ++i) {
            out_raw[i] += per_thread[t][i];
        }
    }

    // 4. clamp
    for (int i = 0; i < H * W * 3; ++i) {
        output_image[i] = std::clamp(out_raw[i], 0.0f, 1.0f);
    }
}
```

### 4.2 메모리 비용

- 256×256×3 float × T threads ≈ 6 MB (T=8 기준)
- 매 호출마다 할당/해제: malloc 비용 ~수 µs × T → 가우시안 처리에 비해 무시

추후 최적화 여지: per-thread buffer를 정적으로 들고 있기 (pool 멤버로). 지금은 YAGNI.

### 4.3 Reduce를 병렬화?

```
T=8, H=W=256 기준 reduce: 256·256·3 = 196608 add × 8 threads = ~1.5M float add
```

float add 1.5M ≈ 1-3ms. 굳이 병렬화하면 thread 동기화 오버헤드가 비슷한 수준. 단일 스레드 reduce가 단순하고 충분.

큰 해상도 (1024×1024)로 가면 reduce 자체도 병렬화 후보. 지금은 안 함.

---

## §5 — 파일 구조

`cpp_single/` 미러. 다른 점은 thread pool + parallel 호출.

```
std_thread/
├── __init__.py                    # cpp_single의 autograd.Function 패턴 그대로 복사,
│                                    # _C만 std_thread 모듈로 바꿈
├── CMakeLists.txt                 # cpp_single 복사, target 이름 std_thread_bindings
└── src/
    ├── render_common.hpp          # cpp_single에서 그대로 복사 (math 동일)
    ├── thread_pool.hpp            # 신규: 위 §2 내용 (헤더 전용, inline)
    ├── render_forward.hpp/cpp     # 신규: §4 알고리즘
    ├── render_backward.hpp/cpp    # 신규: §3 알고리즘
    └── render_2dgs_bindings.cpp   # cpp_single 거의 동일, NB_MODULE 이름만 변경
```

### 5.1 `render_common.hpp` — 복사

cpp_single과 수학 동일. 복제. (`yagni.md`: 미래에 공통 변경 필요해지면 그때 빼면 됨.)

### 5.2 `thread_pool.hpp` — 헤더 전용으로

작은 클래스 (~100줄 이내) + `inline` 함수들로 헤더에 다 넣음.
이유: 별도 `.cpp` 만들면 CMake에 또 추가, ODR 면제 위해 inline 필요. 헤더에 두면 한 번에.

```cpp
// thread_pool.hpp
#pragma once
#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

class ThreadPool { ... };

inline ThreadPool& global_pool() {
    static ThreadPool pool(std::thread::hardware_concurrency());
    return pool;
}
```

### 5.3 `__init__.py` — 패턴 복사

```python
import torch
from . import std_thread_bindings as _C    # ← 이름만 다름


class RenderGaussians2D(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mus, sigmas, thetas, opacities, rgbs, H, W, n_sigma):
        # cpp_single과 동일 (이름만 _C 다름)
        ...

    @staticmethod
    def backward(ctx, grad_output):
        ...


def render_gaussians_2d(H, W, mus, sigmas, thetas, opacities, rgbs, n_sigma=3.0):
    return RenderGaussians2D.apply(mus, sigmas, thetas, opacities, rgbs, H, W, n_sigma)
```

### 5.4 `CMakeLists.txt`

cpp_single 복사, 다음만 변경:
- `project(...)`의 이름
- `nanobind_add_module` target 이름: `render_2dgs_bindings` → `std_thread_bindings`
- 소스 목록에 `thread_pool.hpp`는 헤더라 등록 불필요

### 5.5 `render_2dgs/__init__.py` 선택

현재 cpp_single을 기본으로 export. 사용자가 std_thread를 선택할 수 있도록:

```python
# render_2dgs/__init__.py — 옵션 1: 환경변수
import os
backend = os.environ.get("RENDER_2DGS_BACKEND", "cpp_single")
if backend == "std_thread":
    from .std_thread import render_gaussians_2d
else:
    from .cpp_single import render_gaussians_2d
```

또는 명시적 import:
```python
from render_2dgs.std_thread import render_gaussians_2d
```

후자가 단순. **명시적 import 권장**. 환경변수는 어디서 어떤 백엔드 쓰는지 헷갈림.

---

## §6 — 함정 체크리스트

작업 중 빠지기 쉬운 곳:

- [ ] `std::function<void()>` capture에 **참조** (&)로 잡으면 closure가 끝난 뒤 dangling. 우리 코드는 람다가 즉시 실행되므로 OK이지만, 비동기 패턴이면 위험.
- [ ] `parallel_for` 안의 람다에서 `[&]`로 외부 변수 잡을 때 lifetime 확인. 우리는 람다가 끝나기 전에 `parallel_for`도 return 안 하므로 OK.
- [ ] `cv.wait(lock, predicate)` 형태 — predicate 없는 `wait(lock)`은 spurious wakeup에 취약.
- [ ] `mutex`를 `move`/`copy`하지 말 것 (불가능하지만 컴파일 에러). thread_pool 멤버로 가질 때 자연스러움.
- [ ] `active_count` 감소를 `--active_count` (post-decrement)로 하면 락 안에서 안전. atomic이라 OK.
- [ ] forward의 per-thread buffer를 `std::vector<float>` 대신 `new float[H*W*3]`로 해도 됨. vector는 자동 RAII라 편함.
- [ ] reduce 루프에서 `std::memset(out_raw, 0, ...)`을 먼저 — 이전 호출 값 잔존하면 안 됨.
- [ ] `cosf/sinf/expf` 대신 `std::cos/sin/exp` (cpp_single에서 익힌 패턴).
- [ ] `SIGMA_MIN` 상수 사용 (`render_common.hpp` 그대로).

---

## §7 — 작업 진행 권장 순서

1. `render_common.hpp` cpp_single에서 복사
2. `thread_pool.hpp` 작성 (§2)
3. **간단한 단위 테스트**로 thread pool 검증:
   ```cpp
   int main() {
       ThreadPool pool(4);
       std::atomic<int> sum{0};
       pool.parallel_for(100, [&](int start, int end, int tid) {
           for (int i = start; i < end; i++) sum.fetch_add(i);
       });
       assert(sum == 4950);   // 0+1+...+99
   }
   ```
   (이건 옵션 — 바로 forward/backward로 가도 됨. 하지만 thread pool 정확성 빨리 확인하는 게 안전.)
4. `render_forward.cpp` / `.hpp` (§4)
5. `render_backward.cpp` / `.hpp` (§3)
6. `render_2dgs_bindings.cpp` (cpp_single 거의 복사, target 이름만 변경)
7. `CMakeLists.txt`
8. `__init__.py`
9. 빌드
10. parity 테스트: `tests/test_parity.py`를 cpp_single과 std_thread 둘 다 테스트하도록 확장 (별도 PR 또는 같이)
11. bench.py 또는 별도 스크립트로 cpp_single vs std_thread 시간 비교

---

## §8 — 예상 성능

가우시안당 처리 시간이 dominate. T 스레드면 ~T배 가속 이상적.

cpp_single: forward 9.17ms, backward 11.15ms.

T=8 (M1 Pro 등):
- Forward: ~9.17/8 ≈ 1.2ms + reduce ~2ms = **~3-4ms** 예상
- Backward: ~11.15/8 ≈ 1.4ms + 동기화 오버헤드 ~0.5ms = **~2ms** 예상

실제로는 T가 hardware_concurrency보다 hyperthread 포함이라 효율 ~0.6-0.7T 정도. 그래도 forward 2-3x, backward 4-5x 가속 기대.

만약 측정해서 reduce가 dominate한다면 § 4.3대로 reduce 병렬화 도입. 측정 전 안 함.

---

## §9 — 검증

1. **parity**: cpp_single과 std_thread의 출력이 floating-point error 범위 내 일치
   - forward: cpp_single과 동일 출력 (math 같음, 누적 순서만 다름 → fp32 noise ~1e-6 정도 예상)
   - backward: 동일 출력 (각 가우시안 독립)

2. **타이밍**: cpp_single 대비 가속 확인. T=8 기준 forward/backward 모두 가속 보여야 함.

3. **rebuild + parity test**:
   ```bash
   cmake --build render_2dgs/std_thread/build -j
   cmake --install render_2dgs/std_thread/build
   python tests/test_parity.py  # 둘 다 테스트하도록 확장 후
   ```

---

## 참고

- 학습 가이드 §0-§12 (cpp_learning_guide_2026-05-20.md) — 컴파일 모델, ODR, `.hpp`/`.cpp` 분리는 모두 이 백엔드에도 적용
- [autograd_integration.md](../../docs/autograd_integration.md) — Python 래퍼 패턴은 cpp_single과 동일
- [backward_derivation.md](../../docs/backward_derivation.md) — backward 수식 (§3 부호 오류 이미 반영)
- C++ Concurrency in Action (Anthony Williams) — 스레드 풀 패턴 정석
