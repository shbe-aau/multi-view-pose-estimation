import torch
import torch.nn as nn
from pytorch3d.renderer.blending import softmax_rgb_blend
from pytorch3d.renderer.blending import sigmoid_alpha_blend

class DepthShader(nn.Module):
    def __init__(self, blend_params=None):
        super().__init__()
        self.blend_params = blend_params if blend_params is not None else BlendParams()

    def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:
        #image = fragments.zbuf
        #return image

        image = fragments.zbuf#[..., 0]

        mask = image > 0
        image = image - mask*1000
        return image

        # batch_mins,_ = torch.min(image.flatten(start_dim=1),dim=1)
        # batch_maxs,_ = torch.max(image.flatten(start_dim=1),dim=1)

        # mins = batch_mins.expand(image.shape[2],image.shape[1],image.shape[0]).permute(2,1,0)
        # maxs = batch_maxs.expand(image.shape[2],image.shape[1],image.shape[0]).permute(2,1,0)

        # image = image / (maxs - mins)

        #colors = torch.stack([fragments.zbuf,fragments.zbuf,fragments.zbuf]).permute(1,2,3,4,0)
        #images = softmax_rgb_blend(colors, fragments, self.blend_params)
        #images = sigmoid_alpha_blend(colors, fragments, self.blend_params)
        #return images
