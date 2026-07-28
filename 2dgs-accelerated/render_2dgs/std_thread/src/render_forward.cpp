#include "render_forward.hpp"
#include "render_common.hpp"
#include "thread_pool.hpp"

#include <algorithm>
#include <cstring>
#include <vector>

///////////
#include <chrono>
#include <cstdio>
#include <cstdlib>

namespace {
using clk = std::chrono::steady_clock;
inline bool profile_on() {
  static const bool on = std::getenv("GS2D_PROFILE") != nullptr;
  return on;
}
inline double ms(clk::time_point a, clk::time_point b) {
  return std::chrono::duration<double, std::milli>(b - a).count();
}
}  // namespace
///////////////

void render_gaussian_2d(int H, int W, int N, const float *mus,
                        const float *sigmas, const float *thetas,
                        const float *opacities, const float *rgbs,
                        float *output_image, float *out_raw, float n_sigma) {
  auto &pool = render_2dgs::GlobalPool();
  uint32_t T = pool.NumThreads();

  auto t0 = clk::now();
  std::vector<std::vector<float>> per_thread(
      T, std::vector<float>(static_cast<size_t>(H) * W * 3, 0.0f));
  auto t1 = clk::now();                     // (1) alloc + zero

  pool.ParallelFor(
      static_cast<uint32_t>(N),
      [&](uint32_t start, uint32_t end, int tid) {
        float *buf = per_thread[tid].data();
        for (uint32_t i = start; i < end; ++i) {
          Gaussian2D g;
          g.mu = {mus[2 * i], mus[2 * i + 1]};
          g.sigma = {std::max(sigmas[2 * i], SIGMA_MIN),
                     std::max(sigmas[2 * i + 1], SIGMA_MIN)};
          g.theta = thetas[i];
          g.opacity = opacities[i];
          g.rgb = {rgbs[3 * i], rgbs[3 * i + 1], rgbs[3 * i + 2]};

          PixelRange r = bbox_pixel_range(g, W, H, n_sigma);
          Mat2 R = make_rotation(g.theta);
          Mat2 sigma_inv = make_sigma_inv(g.sigma, R);

          for (int y = r.iy0; y <= r.iy1; ++y) {
            for (int x = r.ix0; x <= r.ix1; ++x) {
              Vec2 pixel = pixel_coords(x, y, W, H);
              float value = gaussian_value(pixel, g.mu, sigma_inv) * g.opacity;

              int idx = (y * W + x) * 3;
              buf[idx]     += value * g.rgb.r;
              buf[idx + 1] += value * g.rgb.g;
              buf[idx + 2] += value * g.rgb.b;
            }
          }
        }
      });
  auto t2 = clk::now();                     // (2) parallel fill

  std::memset(out_raw, 0, sizeof(float) * H * W * 3);
  
  for (uint32_t t = 0; t < T; ++t) {
    const float *buf = per_thread[t].data();
    for (int i = 0; i < H * W * 3; ++i) {
      out_raw[i] += buf[i];
    }
  }
  auto t3 = clk::now();                     // (3) serial reduce
  
  for (int i = 0; i < H * W * 3; ++i) {
    output_image[i] = std::clamp(out_raw[i], 0.0f, 1.0f);
  }
  auto t4 = clk::now();                     // (4) clamp
  if (profile_on())
  std::fprintf(stderr, "PHASE T=%u N=%d alloc=%.4f fill=%.4f reduce=%.4f clamp=%.4f\n",
                T, N, ms(t0,t1), ms(t1,t2), ms(t2,t3), ms(t3,t4));
}
