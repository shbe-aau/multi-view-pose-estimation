import torch
import torch.nn as nn
from pytorch3d.renderer.blending import softmax_rgb_blend
from pytorch3d.renderer.blending import sigmoid_alpha_blend

class HardDepthShader(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:
        image = fragments.zbuf[..., 0]
        #mask = image > 0
        #image = image - mask*1000.0
        return image


class DepthShader(nn.Module):
    def __init__(self, blend_params=None):
        super().__init__()
        self.blend_params = blend_params if blend_params is not None else BlendParams()


    def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:
        # Z buffer as color
        colors = torch.stack([fragments.zbuf]).permute(1,2,3,4,0)

        mask = colors > 0
        colors = colors - mask*700

        N, H, W, K = fragments.pix_to_face.shape
        device = fragments.pix_to_face.device
        pixel_colors = torch.ones((N, H, W, 1), dtype=colors.dtype, device=colors.device)
        background = self.blend_params.background_color
        background = (1.0)
        if not torch.is_tensor(background):
            background = torch.tensor(background, dtype=torch.float32, device=device)
        else:
            background = background.to(device)

        # Weight for background color
        eps = 1e-10

        # Mask for padded pixels.
        mask = fragments.pix_to_face >= 0
        
        # Sigmoid probability map based on the distance of the pixel to the face.
        prob_map = torch.sigmoid(-fragments.dists / self.blend_params.sigma) * mask
        
        # The cumulative product ensures that alpha will be 0.0 if at least 1
        # face fully covers the pixel as for that face, prob will be 1.0.
        # This results in a multiplication by 0.0 because of the (1.0 - prob)
        # term. Therefore 1.0 - alpha will be 1.0.
        alpha = torch.prod((1.0 - prob_map), dim=-1)
        
        znear = -10000.0
        zfar = 10000.0
        z_inv = (zfar - fragments.zbuf) / (zfar - znear) * mask
        # pyre-fixme[16]: `Tuple` has no attribute `values`.
        # pyre-fixme[6]: Expected `Tensor` for 1st param but got `float`.
        z_inv_max = torch.max(z_inv, dim=-1).values[..., None].clamp(min=eps)
        # pyre-fixme[6]: Expected `Tensor` for 1st param but got `float`.
        #weights_num = prob_map
        weights_num = prob_map * torch.exp((z_inv - z_inv_max) / self.blend_params.gamma)
        
        # Also apply exp normalize trick for the background color weight.
        # Clamp to ensure delta is never 0.
        # pyre-fixme[20]: Argument `max` expected.
        # pyre-fixme[6]: Expected `Tensor` for 1st param but got `float`.
        delta = torch.exp((eps - z_inv_max) / self.blend_params.gamma).clamp(min=eps)
        
        # Normalize weights.
        # weights_num shape: (N, H, W, K). Sum over K and divide through by the sum.
        denom = weights_num.sum(dim=-1)[..., None] + delta

        # Sum: weights * textures + background color
        weighted_colors = (weights_num[..., None] * colors).sum(dim=-2)
        weighted_background = delta * background
        
        #print(pixel_colors.shape)

        pixel_colors = (weighted_colors + weighted_background) / denom
        #pixel_colors[..., :3] = (weighted_colors + weighted_background) / denom
        #pixel_colors[..., 3] = 1.0 - alpha
        return pixel_colors[..., 0]
    
        # image = fragments.zbuf
        
        # mask = image > 0
        # image = image - mask*700.0

        # return image

        #colors = torch.stack([fragments.zbuf,fragments.zbuf,fragments.zbuf]).permute(1,2,3,4,0)
        #images = softmax_rgb_blend(colors, fragments, self.blend_params)
        #images = sigmoid_alpha_blend(colors, fragments, self.blend_params)
        #return images
