#pragma once

#include "render_common.hpp"

void render_gaussian_2d_backward_one(int H, int W,
                                     const Gaussian2D &g_in,
                                     const float *out_raw,
                                     const float *grad_out,
                                     float n_sigma,
                                     Vec2 &grad_mu,
                                     Vec2 &grad_sigma,
                                     float &grad_theta,
                                     float &grad_opacity,
                                     Vec3 &grad_rgb);

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
                                 float *grad_rgbs);
