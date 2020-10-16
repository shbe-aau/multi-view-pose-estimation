import torch
import torch.nn as nn
from pytorch3d.renderer.blending import softmax_rgb_blend
from pytorch3d.renderer.blending import sigmoid_alpha_blend

class DepthShader(nn.Module):
    def __init__(self, blend_params=None):
        super().__init__()
        self.blend_params = blend_params if blend_params is not None else BlendParams()

    def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:
        image = fragments.zbuf

        mask = image > 0
        image = image - mask*700.0

        return image

        #colors = torch.stack([fragments.zbuf,fragments.zbuf,fragments.zbuf]).permute(1,2,3,4,0)
        #images = softmax_rgb_blend(colors, fragments, self.blend_params)
        #images = sigmoid_alpha_blend(colors, fragments, self.blend_params)
        #return images
