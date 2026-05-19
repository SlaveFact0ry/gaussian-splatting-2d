#include <cmath>
#include <cstring>
#include <vector>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

namespace nb = nanobind;
using namespace std;
struct Vec2 {
  float x, y;
  Vec2 operator-(const Vec2 &o) const { return {x - o.x, y - o.y}; }
};
struct Vec3 {
  float r, g, b;
};

struct Mat2 {
  float m[2][2];
  // 힌트: @ 연산, transpose, diag 필요
};

struct Gaussian2D {
  Vec2 mu;
  Vec2 sigma; // [sigma_x, sigma_y]
  float theta;
  float opacity;
  Vec3 rgb;
};
// 픽셀 그리드 - (H*W, 2) 대신 그냥 루프로 대체 가능
// make_pixel_grid → 루프에서 직접 계산하면 더 자연스러움

Mat2 make_rotation(float theta) {
  float cos_t = cos(theta);
  float sin_t = sin(theta);
  Mat2 R = {{cos_t, -sin_t, sin_t, cos_t}};
  return R;
};
Mat2 mat_mul(const Mat2 &A, const Mat2 &B) {

  Mat2 result;
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++) {
      result.m[i][j] = A.m[i][0] * B.m[0][j] + A.m[i][1] * B.m[1][j];
    }
  }
  return result;
};
Mat2 transpose(const Mat2 &A) {
  Mat2 result;
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++) {
      result.m[i][j] = A.m[j][i];
    }
  }
  return result;
};
Mat2 make_sigma_inv(const Vec2 &sigma, const Mat2 &R) {
  // Sigma_inv = R @ diag(1/sigma^2) @ R^T
  Mat2 D_inv;
  D_inv.m[0][0] = 1.0f / (sigma.x * sigma.x);
  D_inv.m[0][1] = 0.0f;
  D_inv.m[1][0] = 0.0f;
  D_inv.m[1][1] = 1.0f / (sigma.y * sigma.y);

  Mat2 R_T = transpose(R);

  return mat_mul(mat_mul(R, D_inv), R_T);
};

float gaussian_value(Vec2 pixel, Vec2 mu, const Mat2 &sigma_inv) {
  Vec2 diff = pixel - mu;
  float exponent =
      diff.x * (sigma_inv.m[0][0] * diff.x + sigma_inv.m[0][1] * diff.y) +
      diff.y * (sigma_inv.m[1][0] * diff.x + sigma_inv.m[1][1] * diff.y);
  return exp(-0.5f * exponent);
};

void render_gaussian_2d(int H, int W, int N,
                        const float *mus,       // (N, 2)
                        const float *sigmas,    // (N, 2)
                        const float *thetas,    // (N,)
                        const float *opacities, // (N,)
                        const float *rgbs,      // (N, 3)
                        float *output_image,    // (H, W, 3)
                        float n_sigma) {
  std::memset(output_image, 0, sizeof(float) * H * W * 3);
  for (int i = 0; i < N; i++) {
    Gaussian2D g;
    g.mu     = {mus[2 * i], mus[2 * i + 1]};
    g.sigma  = {max(sigmas[2 * i], 1e-3f), max(sigmas[2 * i + 1], 1e-3f)};
    g.theta  = thetas[i];
    g.opacity = opacities[i];
    g.rgb    = {rgbs[3 * i], rgbs[3 * i + 1], rgbs[3 * i + 2]};

    float half_w = n_sigma * (abs(cos(g.theta) * g.sigma.x) +
                              abs(sin(g.theta) * g.sigma.y));
    float half_h = n_sigma * (abs(sin(g.theta) * g.sigma.x) +
                              abs(cos(g.theta) * g.sigma.y));

    float x0 = g.mu.x - half_w;
    float y0 = g.mu.y - half_h;
    float x1 = g.mu.x + half_w;
    float y1 = g.mu.y + half_h;

    int ix0 = max(0,     (int)ceil ((x0 + 1.0f) * 0.5f * W - 0.5f));
    int ix1 = min(W - 1, (int)floor((x1 + 1.0f) * 0.5f * W - 0.5f));
    int iy0 = max(0,     (int)ceil ((1.0f - y1) * 0.5f * H - 0.5f));
    int iy1 = min(H - 1, (int)floor((1.0f - y0) * 0.5f * H - 0.5f));

    Mat2 R = make_rotation(g.theta);
    Mat2 sigma_inv = make_sigma_inv(g.sigma, R);

    for (int y = iy0; y <= iy1; y++) {
      for (int x = ix0; x <= ix1; x++) {
        float px = (x + 0.5f) / W * 2.0f - 1.0f;
        float py = ((H - 1 - y) + 0.5f) / H * 2.0f - 1.0f;
        Vec2 pixel = {px, py};
        float value = gaussian_value(pixel, g.mu, sigma_inv) * g.opacity;

        int idx = (y * W + x) * 3;
        output_image[idx] += value * g.rgb.r;
        output_image[idx + 1] += value * g.rgb.g;
        output_image[idx + 2] += value * g.rgb.b;
      }
    }
  }
  for (int i = 0; i < H * W * 3; i++) {
    output_image[i] = max(0.0f, min(output_image[i], 1.0f)); // clamp(0, 1)
  }
}

using FArr = nb::ndarray<float, nb::c_contig, nb::device::cpu>;

NB_MODULE(render_2dgs_bindings, m) {
  m.def(
      "render_gaussians_2d",
      [](int H, int W, FArr mus, FArr sigmas, FArr thetas, FArr opacities,
         FArr rgbs, float n_sigma) {
        int N = static_cast<int>(mus.shape(0));

        float *out = new float[H * W * 3];
        size_t shape[3] = {(size_t)H, (size_t)W, 3};

        {
          nb::gil_scoped_release guard;
          render_gaussian_2d(H, W, N, mus.data(), sigmas.data(), thetas.data(),
                             opacities.data(), rgbs.data(), out, n_sigma);
        }

        nb::capsule owner(out, [](void *p) noexcept {
          delete[] static_cast<float *>(p);
        });
        return nb::ndarray<nb::numpy, float>(out, 3, shape, owner);
      },
      nb::arg("H"), nb::arg("W"), nb::arg("mus"), nb::arg("sigmas"),
      nb::arg("thetas"), nb::arg("opacities"), nb::arg("rgbs"),
      nb::arg("n_sigma") = 3.0f);
}
