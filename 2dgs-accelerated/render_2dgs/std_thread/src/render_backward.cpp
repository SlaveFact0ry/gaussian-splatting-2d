#include "render_backward.hpp"
#include "render_common.hpp"

#include <algorithm>
#include <cmath>

void render_gaussian_2d_backward_one(int H, int W,
                                     const Gaussian2D &g_in,
                                     const float *out_raw,
                                     const float *grad_out,
                                     float n_sigma,
                                     Vec2 &grad_mu,
                                     Vec2 &grad_sigma,
                                     float &grad_theta,
                                     float &grad_opacity,
                                     Vec3 &grad_rgb) {
  Vec2 sigma = {std::max(g_in.sigma.x, SIGMA_MIN),
                std::max(g_in.sigma.y, SIGMA_MIN)};
  bool sx_clamped = (g_in.sigma.x < SIGMA_MIN);
  bool sy_clamped = (g_in.sigma.y < SIGMA_MIN);

  float c = std::cos(g_in.theta);
  float s = std::sin(g_in.theta);
  float d_x = 1.0f / (sigma.x * sigma.x);
  float d_y = 1.0f / (sigma.y * sigma.y);

  Gaussian2D g_clamped = g_in;
  g_clamped.sigma = sigma;
  Mat2 R = make_rotation(g_in.theta);
  Mat2 A = make_sigma_inv(sigma, R);
  float A00 = A.m[0][0];
  float A01 = A.m[0][1];
  float A11 = A.m[1][1];

  PixelRange r = bbox_pixel_range(g_clamped, W, H, n_sigma);

  float dA00 = 0, dA01 = 0, dA11 = 0;
  float gmu_x = 0, gmu_y = 0;
  float gop = 0;
  Vec3 g_rgb = {0, 0, 0};

  for (int y = r.iy0; y <= r.iy1; ++y) {
    for (int x = r.ix0; x <= r.ix1; ++x) {
      Vec2 pixel = pixel_coords(x, y, W, H);
      float dx = pixel.x - g_in.mu.x;
      float dy = pixel.y - g_in.mu.y;

      float e = A00 * dx * dx + 2.0f * A01 * dx * dy + A11 * dy * dy;
      float f = std::exp(-0.5f * e);

      int idx = (y * W + x) * 3;
      float g0 = (out_raw[idx]     > 0.f && out_raw[idx]     < 1.f) ? grad_out[idx]     : 0.f;
      float g1 = (out_raw[idx + 1] > 0.f && out_raw[idx + 1] < 1.f) ? grad_out[idx + 1] : 0.f;
      float g2 = (out_raw[idx + 2] > 0.f && out_raw[idx + 2] < 1.f) ? grad_out[idx + 2] : 0.f;

      float op_f = g_in.opacity * f;
      g_rgb.r += op_f * g0;
      g_rgb.g += op_f * g1;
      g_rgb.b += op_f * g2;

      float rg_dot = g_in.rgb.r * g0 + g_in.rgb.g * g1 + g_in.rgb.b * g2;
      float df = g_in.opacity * rg_dot;
      float de = -0.5f * f * df;

      gop += f * rg_dot;

      dA00 += de * dx * dx;
      dA01 += de * dx * dy;
      dA11 += de * dy * dy;

      gmu_x += -de * 2.0f * (A00 * dx + A01 * dy);
      gmu_y += -de * 2.0f * (A11 * dy + A01 * dx);
    }
  }

  float gd_x = dA00 * c * c + 2.0f * dA01 * c * s + dA11 * s * s;
  float gd_y = dA00 * s * s - 2.0f * dA01 * c * s + dA11 * c * c;
  float gsx = sx_clamped ? 0.f : (-2.0f / (sigma.x * sigma.x * sigma.x)) * gd_x;
  float gsy = sy_clamped ? 0.f : (-2.0f / (sigma.y * sigma.y * sigma.y)) * gd_y;

  float gc = dA00 * 2.f * c * d_x + 2.f * dA01 * s * (d_x - d_y) + dA11 * 2.f * c * d_y;
  float gs = dA00 * 2.f * s * d_y + 2.f * dA01 * c * (d_x - d_y) + dA11 * 2.f * s * d_x;
  float gtheta = -s * gc + c * gs;

  grad_mu      = {gmu_x, gmu_y};
  grad_sigma   = {gsx, gsy};
  grad_theta   = gtheta;
  grad_opacity = gop;
  grad_rgb     = g_rgb;
}

void render_gaussian_2d_backward(int H, int W, int N,
                                 const float *mus,
                                 const float *sigmas,
                                 const float *thetas,
                                 const float *opacities,
                                 const float *rgbs,
                                 const float *out_raw,
                                 const float *grad_out,
                                 float n_sigma,
                                 float *grad_mus,
                                 float *grad_sigmas,
                                 float *grad_thetas,
                                 float *grad_opacities,
                                 float *grad_rgbs) {
  for (int i = 0; i < N; i++) {
    Gaussian2D g_in;
    g_in.mu      = {mus[2 * i], mus[2 * i + 1]};
    g_in.sigma   = {sigmas[2 * i], sigmas[2 * i + 1]};
    g_in.theta   = thetas[i];
    g_in.opacity = opacities[i];
    g_in.rgb     = {rgbs[3 * i], rgbs[3 * i + 1], rgbs[3 * i + 2]};

    Vec2 g_mu, g_sigma;
    float g_theta, g_op;
    Vec3 g_rgb;
    render_gaussian_2d_backward_one(H, W, g_in, out_raw, grad_out, n_sigma,
                                    g_mu, g_sigma, g_theta, g_op, g_rgb);

    grad_mus[2 * i]      = g_mu.x;
    grad_mus[2 * i + 1]  = g_mu.y;
    grad_sigmas[2 * i]   = g_sigma.x;
    grad_sigmas[2 * i + 1] = g_sigma.y;
    grad_thetas[i]       = g_theta;
    grad_opacities[i]    = g_op;
    grad_rgbs[3 * i]     = g_rgb.r;
    grad_rgbs[3 * i + 1] = g_rgb.g;
    grad_rgbs[3 * i + 2] = g_rgb.b;
  }
}
