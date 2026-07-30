#include <cuda_runtime.h>
#include <cstdint>

#include "cuda_utils.cuh"
#include "render_common.cuh"

namespace {

__global__ void render_naive_kernel(
    int H, int W, int N, float n_sigma,
    const float *__restrict__ mus,        // [N,2]
    const float *__restrict__ sigmas,     // [N,2]
    const float *__restrict__ thetas,     // [N]
    const float *__restrict__ opacities,  // [N]
    const float *__restrict__ rgbs,       // [N,3]
    float *__restrict__ img,              // [H,W,3]
    float *__restrict__ out_raw)          // [H,W,3]
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;

    const Vec2 p = pixel_coords(x, y, W, H);
    float acc_r = 0.f, acc_g = 0.f, acc_b = 0.f;

    for (int i = 0; i < N; ++i) {
        Gaussian2D g;
        g.mu      = {mus[2 * i], mus[2 * i + 1]};
        g.sigma   = {fmaxf(sigmas[2 * i],     SIGMA_MIN),
                     fmaxf(sigmas[2 * i + 1], SIGMA_MIN)};
        g.theta   = thetas[i];
        g.opacity = opacities[i];
        g.rgb     = {rgbs[3 * i], rgbs[3 * i + 1], rgbs[3 * i + 2]};

        // CPU 레퍼런스와 동일한 컬링 — 빠뜨리면 parity 실패
        const PixelRange r = bbox_pixel_range(g, W, H, n_sigma);
        if (x < r.ix0 || x > r.ix1 || y < r.iy0 || y > r.iy1) continue;

        const Mat2 R  = make_rotation(g.theta);
        const Mat2 Si = make_sigma_inv(g.sigma, R);
        const float v = gaussian_value(p, g.mu, Si) * g.opacity;

        acc_r += v * g.rgb.r;
        acc_g += v * g.rgb.g;
        acc_b += v * g.rgb.b;
    }

    const int idx = (y * W + x) * 3;
    out_raw[idx + 0] = acc_r;
    out_raw[idx + 1] = acc_g;
    out_raw[idx + 2] = acc_b;
    img[idx + 0] = fminf(fmaxf(acc_r, 0.f), 1.f);
    img[idx + 1] = fminf(fmaxf(acc_g, 0.f), 1.f);
    img[idx + 2] = fminf(fmaxf(acc_b, 0.f), 1.f);
}

inline const float *P(std::uintptr_t p) {
    return reinterpret_cast<const float *>(p);
}

}  // namespace

void render_forward_cuda(int H, int W, int N,
                         std::uintptr_t mus, std::uintptr_t sigmas,
                         std::uintptr_t thetas, std::uintptr_t opacities,
                         std::uintptr_t rgbs,
                         std::uintptr_t img, std::uintptr_t out_raw,
                         float n_sigma)
{
    const dim3 block(16, 16);
    const dim3 grid(ceil_div(W, block.x), ceil_div(H, block.y));

    render_naive_kernel<<<grid, block>>>(
        H, W, N, n_sigma,
        P(mus), P(sigmas), P(thetas), P(opacities), P(rgbs),
        reinterpret_cast<float *>(img),
        reinterpret_cast<float *>(out_raw));

    CUDA_CHECK(cudaGetLastError());
}