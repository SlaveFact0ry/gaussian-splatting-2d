#include "render_forward.hpp"
#include "render_common.hpp"
#include "thread_pool.hpp"

#include <algorithm>
#include <cstring>
#include <vector>

void render_gaussian_2d(int H, int W, int N, const float *mus,
                        const float *sigmas, const float *thetas,
                        const float *opacities, const float *rgbs,
                        float *output_image, float *out_raw, float n_sigma) {
  auto &pool = render_2dgs::GlobalPool();
  uint32_t T = pool.NumThreads();

  std::vector<std::vector<float>> per_thread(
      T, std::vector<float>(static_cast<size_t>(H) * W * 3, 0.0f));

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

  std::memset(out_raw, 0, sizeof(float) * H * W * 3);
  for (uint32_t t = 0; t < T; ++t) {
    const float *buf = per_thread[t].data();
    for (int i = 0; i < H * W * 3; ++i) {
      out_raw[i] += buf[i];
    }
  }

  for (int i = 0; i < H * W * 3; ++i) {
    output_image[i] = std::clamp(out_raw[i], 0.0f, 1.0f);
  }
}
