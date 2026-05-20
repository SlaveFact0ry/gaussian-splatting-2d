import torch


def make_pixel_grid(H, W, device):
    xs = (torch.arange(W, device=device) + 0.5) / W * 2 - 1
    ys = (torch.arange(H, device=device) + 0.5) / H * 2 - 1
    Y, X = torch.meshgrid(ys, xs, indexing="ij")
    Y = torch.flip(Y, dims=[0])
    return torch.stack([X, Y], dim=-1)


def render_gaussians_2d(H, W, mus, sigmas, thetas, opacities, rgbs, n_sigma=3.0):
    P = make_pixel_grid(H, W, device=mus.device)
    img = torch.zeros((H, W, 3), device=mus.device, dtype=mus.dtype)

    for i in range(len(mus)):
        mu = mus[i]
        sigma = torch.clamp(sigmas[i], min=1e-3)
        theta = thetas[i]
        opacity = opacities[i]
        rgb = rgbs[i]

        cos_t, sin_t = torch.cos(theta), torch.sin(theta)
        half_w = n_sigma * (torch.abs(cos_t) * sigma[0] + torch.abs(sin_t) * sigma[1])
        half_h = n_sigma * (torch.abs(sin_t) * sigma[0] + torch.abs(cos_t) * sigma[1])
        x0, x1 = mu[0] - half_w, mu[0] + half_w
        y0, y1 = mu[1] - half_h, mu[1] + half_h
        mask_bbox = (P[..., 0] >= x0) & (P[..., 0] <= x1) & (P[..., 1] >= y0) & (P[..., 1] <= y1)
        if not mask_bbox.any():
            continue

        R = torch.stack([torch.stack([cos_t, -sin_t]), torch.stack([sin_t, cos_t])])
        sigma_inv = R @ torch.diag(1.0 / (sigma ** 2)) @ R.T

        diff = P[mask_bbox] - mu
        exponent = torch.einsum("bi,ij,bj->b", diff, sigma_inv, diff)
        f_x = torch.exp(-0.5 * exponent)
        img[mask_bbox] = img[mask_bbox] + (opacity * rgb).view(1, 3) * f_x.unsqueeze(-1)

    return torch.clamp(img, 0.0, 1.0)

