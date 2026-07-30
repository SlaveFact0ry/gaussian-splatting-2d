#pragma once

#include <cuda_runtime.h>
#include <cstdio>
#include <stdexcept>
#include <string>

#define CUDA_CHECK(expr)                                                     \
  do {                                                                       \
    cudaError_t err__ = (expr);                                              \
    if (err__ != cudaSuccess) {                                              \
      throw std::runtime_error(std::string("CUDA error at ") + __FILE__ +    \
                               ":" + std::to_string(__LINE__) + " : " +      \
                               cudaGetErrorString(err__));                   \
    }                                                                        \
  } while (0)

// 커널 런치 직후에 호출 — 런치 실패와 비동기 실행 오류를 모두 잡는다.
#define CUDA_CHECK_LAST()                                                    \
  do {                                                                       \
    CUDA_CHECK(cudaGetLastError());                                          \
    CUDA_CHECK(cudaDeviceSynchronize());                                     \
  } while (0)

__host__ __device__ inline int ceil_div(int a, int b) { return (a + b - 1) / b; }