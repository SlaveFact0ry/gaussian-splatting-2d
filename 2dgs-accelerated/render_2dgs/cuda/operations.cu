/*
 * operations.cu - CUDA kernels and C++ wrapper functions
 *
 * This file contains:
 * 1. CUDA kernels (__global__ functions that run on GPU)
 * 2. Wrapper functions that handle torch::Tensor and launch kernels
 */

//// =========================================================================
//// [CLAUDE-REVIEW · 2026-05-07] — added by AI, NOT part of original file
////
//// All lines beginning with "////" in this file are review notes. They were
//// inserted by Claude during a code review and should be removed/addressed
//// rather than treated as original documentation. The actual source code
//// below has NOT been modified.
////
//// File-level summary of issues (each has an inline "////" marker at its
//// site below):
////
////   #1  multiply_kernel: stray "+ 1024.0" — function name says multiply
////       but the kernel actually computes (x*scalar + 1024). The test
////       silently reports "Match: False" instead of failing.
////
////   #2  add_tensors_cuda / multiply_scalar_cuda: no cudaGetLastError()
////       after the kernel launch. Bad launch configs (e.g. > 1024 threads
////       per block) fail silently — the next CUDA call is what surfaces
////       the error, with a confusing line number.
////
////   #3  torch::zeros_like(a): zeros memory we immediately overwrite.
////       torch::empty_like(a) skips the zero-fill (~free perf win).
////
////   #4  a.contiguous().data_ptr<float>(): .contiguous() returns a
////       temporary tensor. The data_ptr stays valid only because torch's
////       caching allocator holds the underlying CUDA buffer alive past
////       the temporary's destruction. Fragile — bind the result to a
////       named local first:
////           auto a_c = a.contiguous();
////           auto a_ptr = a_c.data_ptr<float>();
////
//// =========================================================================

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

// ============================================================
// CUDA Kernels (run on GPU)
// ============================================================

// Kernel: Add two arrays element-wise
__global__ void add_kernel(const float* a, const float* b, float* out, int n) {
    // Calculate global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Bounds check
    if (idx < n) {
        out[idx] = a[idx] + b[idx];
    }
}

// Kernel: Multiply array by scalar
__global__ void multiply_kernel(const float* input, float* output, float scalar, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        //// [CLAUDE-REVIEW] Issue #1 — stray "+ 1024.0".
        //// Function is named multiply_kernel but actually computes
        //// (x * scalar) + 1024. Should be `output[idx] = input[idx] * scalar;`.
        //// test_multiply in test_cuda_hello.py prints "Match: False" but
        //// doesn't assert, so the failure is silent.
        output[idx] = input[idx] * scalar + 1024.0;
    }
}

// Kernel: Just print hello (for demonstration)
__global__ void hello_kernel() {
    printf("Hello from CUDA thread %d in block %d!\n", threadIdx.x, blockIdx.x);
}

// ============================================================
// C++ Wrapper Functions (called from Python via PyBind11)
// ============================================================

torch::Tensor add_tensors_cuda(torch::Tensor a, torch::Tensor b) {
    // Input validation
    TORCH_CHECK(a.device().is_cuda(), "Tensor 'a' must be on CUDA device");
    TORCH_CHECK(b.device().is_cuda(), "Tensor 'b' must be on CUDA device");
    TORCH_CHECK(a.sizes() == b.sizes(), "Tensor shapes must match");
    TORCH_CHECK(a.dtype() == torch::kFloat32, "Tensor 'a' must be float32");
    TORCH_CHECK(b.dtype() == torch::kFloat32, "Tensor 'b' must be float32");
    
    // Get total number of elements
    const int n = a.numel();

    //// [CLAUDE-REVIEW] Issue #3 — torch::zeros_like writes 0 to every
    //// element, but the kernel below overwrites every element. Use
    //// torch::empty_like(a) to skip the redundant zero-fill.
    // Create output tensor with same shape and device
    auto output = torch::zeros_like(a);

    // Calculate kernel launch parameters
    const int threads_per_block = 256;
    const int num_blocks = (n + threads_per_block - 1) / threads_per_block;

    //// [CLAUDE-REVIEW] Issue #4 — .contiguous() returns a temporary tensor
    //// whose lifetime ends at the semicolon. The data_ptr survives only
    //// because torch's caching allocator holds the buffer past the
    //// temporary's destruction. Bind to a named local for safety:
    ////   auto a_c = a.contiguous(); auto b_c = b.contiguous();
    ////   add_kernel<<<...>>>(a_c.data_ptr<float>(), b_c.data_ptr<float>(), ...);
    // Launch CUDA kernel
    // Syntax: kernel<<<num_blocks, threads_per_block>>>(args...)
    add_kernel<<<num_blocks, threads_per_block>>>(
        a.contiguous().data_ptr<float>(),      // Get raw float pointer
        b.contiguous().data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );
    //// [CLAUDE-REVIEW] Issue #2 — missing cudaGetLastError() after the
    //// kernel launch. A bad launch config (threads/block > 1024, etc.)
    //// fails silently here; the error surfaces on the next CUDA call
    //// with a confusing call-site. Add:
    ////   AT_CUDA_CHECK(cudaGetLastError());

    // Return result (still on GPU)
    return output;
}

torch::Tensor multiply_scalar_cuda(torch::Tensor input, float scalar) {
    // Input validation
    TORCH_CHECK(input.device().is_cuda(), "Tensor must be on CUDA device");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Tensor must be float32");
    
    const int n = input.numel();
    //// [CLAUDE-REVIEW] Issue #3 (again) — torch::empty_like(input) is
    //// the right call here.
    auto output = torch::zeros_like(input);

    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;

    //// [CLAUDE-REVIEW] Issue #4 (again) — .contiguous() temporary.
    multiply_kernel<<<blocks, threads>>>(
        input.contiguous().data_ptr<float>(),
        output.data_ptr<float>(),
        scalar,
        n
    );
    //// [CLAUDE-REVIEW] Issue #2 (again) — missing AT_CUDA_CHECK(cudaGetLastError()).

    return output;
}

void hello_cuda() {
    // Launch a small kernel just to print
    hello_kernel<<<2, 4>>>();  // 2 blocks, 4 threads each = 8 total threads
    
    // Wait for kernel to complete (so printf shows up)
    cudaDeviceSynchronize();
    
    printf("Hello from C++ (host)!\n");
}
