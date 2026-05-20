#pragma once

#include <algorithm>
#include <cmath>

constexpr float SIGMA_MIN = 1e-3f;

struct Vec2 {
  float x, y;
  Vec2 operator-(const Vec2 &o) const { return {x - o.x, y - o.y}; }
};

struct Vec3 {
  float r, g, b;
};

struct Mat2 {
  float m[2][2];
};

struct Gaussian2D {
  Vec2 mu;
  Vec2 sigma;
  float theta;
  float opacity;
  Vec3 rgb;
};

struct PixelRange {
  int ix0, ix1, iy0, iy1;
};

inline Mat2 make_rotation(float theta) {
  float c = std::cos(theta);
  float s = std::sin(theta);
  return {{{c, -s}, {s, c}}};
}

inline Mat2 mat_mul(const Mat2 &A, const Mat2 &B) {
  Mat2 result;
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++) {
      result.m[i][j] = A.m[i][0] * B.m[0][j] + A.m[i][1] * B.m[1][j];
    }
  }
  return result;
}

inline Mat2 transpose(const Mat2 &A) {
  Mat2 result;
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++) {
      result.m[i][j] = A.m[j][i];
    }
  }
  return result;
}

inline Mat2 make_sigma_inv(const Vec2 &sigma, const Mat2 &R) {
  Mat2 D_inv = {{{1.0f / (sigma.x * sigma.x), 0.0f},
                 {0.0f, 1.0f / (sigma.y * sigma.y)}}};
  Mat2 R_T = transpose(R);
  return mat_mul(mat_mul(R, D_inv), R_T);
}

inline float gaussian_value(Vec2 pixel, Vec2 mu, const Mat2 &sigma_inv) {
  Vec2 diff = pixel - mu;
  float exponent =
      diff.x * (sigma_inv.m[0][0] * diff.x + sigma_inv.m[0][1] * diff.y) +
      diff.y * (sigma_inv.m[1][0] * diff.x + sigma_inv.m[1][1] * diff.y);
  return std::exp(-0.5f * exponent);
}

inline Vec2 pixel_coords(int x, int y, int W, int H) {
  float px = (x + 0.5f) / W * 2.0f - 1.0f;
  float py = ((H - 1 - y) + 0.5f) / H * 2.0f - 1.0f;
  return {px, py};
}

inline PixelRange bbox_pixel_range(const Gaussian2D &g, int W, int H,
                                   float n_sigma) {
  float c = std::cos(g.theta);
  float s = std::sin(g.theta);
  float half_w =
      n_sigma * (std::abs(c * g.sigma.x) + std::abs(s * g.sigma.y));
  float half_h =
      n_sigma * (std::abs(s * g.sigma.x) + std::abs(c * g.sigma.y));

  float x0 = g.mu.x - half_w;
  float y0 = g.mu.y - half_h;
  float x1 = g.mu.x + half_w;
  float y1 = g.mu.y + half_h;

  return {
      std::max(0,     (int)std::ceil ((x0 + 1.0f) * 0.5f * W - 0.5f)),
      std::min(W - 1, (int)std::floor((x1 + 1.0f) * 0.5f * W - 0.5f)),
      std::max(0,     (int)std::ceil ((1.0f - y1) * 0.5f * H - 0.5f)),
      std::min(H - 1, (int)std::floor((1.0f - y0) * 0.5f * H - 0.5f)),
  };
}
