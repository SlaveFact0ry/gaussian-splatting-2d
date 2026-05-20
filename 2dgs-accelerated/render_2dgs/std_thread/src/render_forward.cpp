#include "render_forward.hpp"
#include "render_common.hpp"

#include <algorithm>
#include <cstring>

void render_gaussian_2d(int H, int W, int N,
                        const float *mus,
                        const float *sigmas,
                        const float *thetas,
                        const float *opacities,
                        const float *rgbs,
                        float *output_image,
                        float *out_raw,
                        float n_sigma) {
  std::memset(out_raw, 0, sizeof(float) * H * W * 3);

  for (int i = 0; i < N; i++) {
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

    for (int y = r.iy0; y <= r.iy1; y++) {
      for (int x = r.ix0; x <= r.ix1; x++) {
        Vec2 pixel = pixel_coords(x, y, W, H);
        float value = gaussian_value(pixel, g.mu, sigma_inv) * g.opacity;

        int idx = (y * W + x) * 3;
        out_raw[idx]     += value * g.rgb.r;
        out_raw[idx + 1] += value * g.rgb.g;
        out_raw[idx + 2] += value * g.rgb.b;
      }
    }
  }

  for (int i = 0; i < H * W * 3; i++) {
    output_image[i] = std::clamp(out_raw[i], 0.0f, 1.0f);
  }
}
