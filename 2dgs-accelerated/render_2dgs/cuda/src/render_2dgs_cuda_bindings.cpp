#include <nanobind/nanobind.h>
#include <cstdint>

namespace nb = nanobind;

void render_forward_cuda(int H, int W, int N,
                         std::uintptr_t mus, std::uintptr_t sigmas,
                         std::uintptr_t thetas, std::uintptr_t opacities,
                         std::uintptr_t rgbs,
                         std::uintptr_t img, std::uintptr_t out_raw,
                         float n_sigma);

NB_MODULE(render_2dgs_cuda_bindings, m) {
    m.doc() = "2DGS CUDA render backend (forward only)";
    m.def("render_forward", &render_forward_cuda,
          nb::arg("H"), nb::arg("W"), nb::arg("N"),
          nb::arg("mus"), nb::arg("sigmas"), nb::arg("thetas"),
          nb::arg("opacities"), nb::arg("rgbs"),
          nb::arg("img"), nb::arg("out_raw"), nb::arg("n_sigma"));
}