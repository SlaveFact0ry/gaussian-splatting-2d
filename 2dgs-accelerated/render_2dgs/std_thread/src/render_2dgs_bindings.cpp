#include "render_backward.hpp"
#include "render_forward.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

namespace nb = nanobind;

using FArr = nb::ndarray<float, nb::c_contig, nb::device::cpu>;
using NpArr = nb::ndarray<nb::numpy, float>;

static NpArr make_owned_array(float *data, size_t ndim, const size_t *shape) {
  nb::capsule owner(data, [](void *p) noexcept {
    delete[] static_cast<float *>(p);
  });
  return NpArr(data, ndim, shape, owner);
}

NB_MODULE(render_2dgs_bindings, m) {
  m.def(
      "render_gaussians_2d",
      [](int H, int W, FArr mus, FArr sigmas, FArr thetas, FArr opacities,
         FArr rgbs, float n_sigma) {
        int N = static_cast<int>(mus.shape(0));

        float *out = new float[H * W * 3];
        float *out_raw = new float[H * W * 3];

        {
          nb::gil_scoped_release guard;
          render_gaussian_2d(H, W, N, mus.data(), sigmas.data(), thetas.data(),
                             opacities.data(), rgbs.data(), out, out_raw,
                             n_sigma);
        }

        size_t shape[3] = {(size_t)H, (size_t)W, 3};
        return nb::make_tuple(make_owned_array(out, 3, shape),
                              make_owned_array(out_raw, 3, shape));
      },
      nb::arg("H"), nb::arg("W"), nb::arg("mus"), nb::arg("sigmas"),
      nb::arg("thetas"), nb::arg("opacities"), nb::arg("rgbs"),
      nb::arg("n_sigma") = 3.0f);

  m.def(
      "render_gaussians_2d_backward",
      [](int H, int W, FArr mus, FArr sigmas, FArr thetas, FArr opacities,
         FArr rgbs, FArr out_raw, FArr grad_out, float n_sigma) {
        int N = static_cast<int>(mus.shape(0));

        float *gmus = new float[N * 2];
        float *gsigmas = new float[N * 2];
        float *gthetas = new float[N];
        float *gops = new float[N];
        float *grgbs = new float[N * 3];

        {
          nb::gil_scoped_release guard;
          render_gaussian_2d_backward(
              H, W, N, mus.data(), sigmas.data(), thetas.data(),
              opacities.data(), rgbs.data(), out_raw.data(), grad_out.data(),
              n_sigma, gmus, gsigmas, gthetas, gops, grgbs);
        }

        size_t shape_n2[2] = {(size_t)N, 2};
        size_t shape_n[1]  = {(size_t)N};
        size_t shape_n3[2] = {(size_t)N, 3};

        return nb::make_tuple(make_owned_array(gmus, 2, shape_n2),
                              make_owned_array(gsigmas, 2, shape_n2),
                              make_owned_array(gthetas, 1, shape_n),
                              make_owned_array(gops, 1, shape_n),
                              make_owned_array(grgbs, 2, shape_n3));
      },
      nb::arg("H"), nb::arg("W"), nb::arg("mus"), nb::arg("sigmas"),
      nb::arg("thetas"), nb::arg("opacities"), nb::arg("rgbs"),
      nb::arg("out_raw"), nb::arg("grad_out"), nb::arg("n_sigma") = 3.0f);
}
