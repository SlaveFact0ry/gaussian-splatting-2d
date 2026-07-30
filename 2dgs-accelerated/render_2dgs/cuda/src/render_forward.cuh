#pragma once

#include <cstdint>

void render_forward_cuda(int H, int W, int N,
                         std::uintptr_t mus, std::uintptr_t sigmas,
                         std::uintptr_t thetas, std::uintptr_t opacities,
                         std::uintptr_t rgbs,
                         std::uintptr_t img, std::uintptr_t out_raw,
                         float n_sigma);
