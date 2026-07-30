#pragma once

#include <cmath>

#ifdef __CUDACC__
#define GS2D_HD __host__ __device__
#else
#define GS2D_HD
#endif

constexpr float SIGMA_MIN = 1e-3f;

struct Vec2 {
  float x, y;
  GS2D_HD Vec2 operator-(const Vec2 &o) const { return {x - o.x, y - o.y}; }
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

GS2D_HD inline Mat2 make_rotation(float theta) {
  float c = cosf(theta);
  float s = sinf(theta);
  return {{{c, -s}, {s, c}}};
}

GS2D_HD inline Mat2 mat_mul(const Mat2 &A, const Mat2 &B) {
  Mat2 result;
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++) {
      result.m[i][j] = A.m[i][0] * B.m[0][j] + A.m[i][1] * B.m[1][j];
    }
  }
  return result;
}

GS2D_HD inline Mat2 transpose(const Mat2 &A) {
  Mat2 result;
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++) {
      result.m[i][j] = A.m[j][i];
    }
  }
  return result;
}

GS2D_HD inline Mat2 make_sigma_inv(const Vec2 &sigma, const Mat2 &R) {
  Mat2 D_inv = {{{1.0f / (sigma.x * sigma.x), 0.0f},
                 {0.0f, 1.0f / (sigma.y * sigma.y)}}};
  Mat2 R_T = transpose(R);
  return mat_mul(mat_mul(R, D_inv), R_T);
}

GS2D_HD inline float gaussian_value(Vec2 pixel, Vec2 mu, const Mat2 &sigma_inv) {
  Vec2 diff = pixel - mu;
  float exponent =
      diff.x * (sigma_inv.m[0][0] * diff.x + sigma_inv.m[0][1] * diff.y) +
      diff.y * (sigma_inv.m[1][0] * diff.x + sigma_inv.m[1][1] * diff.y);
  return expf(-0.5f * exponent);
}

GS2D_HD inline Vec2 pixel_coords(int x, int y, int W, int H) {
  float px = (x + 0.5f) / W * 2.0f - 1.0f;
  float py = ((H - 1 - y) + 0.5f) / H * 2.0f - 1.0f;
  return {px, py};
}

GS2D_HD inline PixelRange bbox_pixel_range(const Gaussian2D &g, int W, int H,
                                           float n_sigma) {
  float c = cosf(g.theta);
  float s = sinf(g.theta);
  float half_w = n_sigma * (fabsf(c * g.sigma.x) + fabsf(s * g.sigma.y));
  float half_h = n_sigma * (fabsf(s * g.sigma.x) + fabsf(c * g.sigma.y));

  float x0 = g.mu.x - half_w;
  float y0 = g.mu.y - half_h;
  float x1 = g.mu.x + half_w;
  float y1 = g.mu.y + half_h;

  return {
      max(0,     (int)ceilf ((x0 + 1.0f) * 0.5f * W - 0.5f)),
      min(W - 1, (int)floorf((x1 + 1.0f) * 0.5f * W - 0.5f)),
      max(0,     (int)ceilf ((1.0f - y1) * 0.5f * H - 0.5f)),
      min(H - 1, (int)floorf((1.0f - y0) * 0.5f * H - 0.5f)),
  };
}