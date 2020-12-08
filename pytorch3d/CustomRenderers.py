import torch
import torch.nn as nn
from pytorch3d.renderer.blending import softmax_rgb_blend
from pytorch3d.renderer.blending import sigmoid_alpha_blend

class HardDepthShader(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:
        image = fragments.zbuf[..., 0]
        return image
