#pragma once

void render_gaussian_2d(int H, int W, int N,
                        const float *mus,
                        const float *sigmas,
                        const float *thetas,
                        const float *opacities,
                        const float *rgbs,
                        float *output_image,
                        float *out_raw,
                        float n_sigma);
