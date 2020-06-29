import torch
import numpy as np
from utils.utils import *
from utils.tools import *
import torch.nn as nn

from pytorch3d.renderer import look_at_view_transform

# Required to backpropagate when thresholding (torch.where)
# See: https://discuss.pytorch.org/t/how-do-i-pass-grad-through-torch-where/74671
# And: https://discuss.pytorch.org/t/torch-where-function-blocks-gradient/72570/6
class ThauThreshold(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        thau = 20
        ones = torch.ones(x.shape).cuda()
        zeros = torch.zeros(x.shape).cuda()
        return torch.where(x > thau, ones, zeros)

    @staticmethod
    def backward(ctx, g):
        return g

# Required to backpropagate when thresholding (torch.where)
# See: https://discuss.pytorch.org/t/how-do-i-pass-grad-through-torch-where/74671
# And: https://discuss.pytorch.org/t/torch-where-function-blocks-gradient/72570/6
class NonZero(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ones = torch.ones(x.shape).cuda()
        zeros = torch.zeros(x.shape).cuda()
        return torch.where(x != 0, ones, zeros)

    @staticmethod
    def backward(ctx, g):
        return g

def renderNormCat(Rs, ts, renderer, mean, std, views):
    images = []
    for v in views:
        # Render images
        Rs_new = torch.matmul(Rs, v.to(renderer.device))
        imgs = renderer.renderBatch(Rs_new, ts)
        imgs = (imgs-mean)/std
        images.append(imgs)
    return torch.cat(images)

def Loss(predicted_poses, gt_poses, renderer, ts, mean, std, loss_method="diff", views=None, fixed_gt_images=None):
    Rs_gt = torch.tensor(np.stack(gt_poses), device=renderer.device,
                            dtype=torch.float32)

    if fixed_gt_images is None:
        Rs_predicted = compute_rotation_matrix_from_ortho6d(predicted_poses)
        gt_imgs = renderNormCat(Rs_gt, ts, renderer, mean, std, views)
    else: # this version is for using loss with prerendered ref image and regular rot matrix for predicted pose
        Rs_predicted = predicted_poses
        Rs_predicted = torch.Tensor(Rs_predicted).to(renderer.device)
        gt_imgs = fixed_gt_images
    predicted_imgs = renderNormCat(Rs_predicted, ts, renderer, mean, std, views)
    diff = torch.abs(gt_imgs - predicted_imgs).flatten(start_dim=1) # not needed for "multiview-l2"

    if(loss_method=="bce-loss"):
        loss = nn.BCEWithLogitsLoss(reduction="none")
        loss = loss(predicted_imgs.flatten(start_dim=1),gt_imgs.flatten(start_dim=1))
        loss = torch.mean(loss, dim=1)
        return torch.mean(loss), loss, gt_imgs, predicted_imgs

    elif(loss_method=="bce-loss-sum"):
        loss = nn.BCEWithLogitsLoss(reduction="none")
        loss = loss(predicted_imgs.flatten(start_dim=1),gt_imgs.flatten(start_dim=1))
        loss = torch.sum(loss, dim=1)
        return torch.sum(loss), loss, gt_imgs, predicted_imgs

    elif(loss_method=="pose-mul-depth"):
        # Calc pose loss
        #mseLoss = nn.MSELoss(reduction='none')
        pose_diff = torch.abs(Rs_gt - Rs_predicted).flatten(start_dim=1)/2.0
        pose_loss = torch.mean(pose_diff)
        pose_batch_loss = torch.mean(pose_diff, dim=1)

        # Calc depth loss
        depth_diff = torch.clamp(diff, 0.0, 20.0)/20.0
        depth_loss = torch.mean(depth_diff)
        depth_batch_loss = torch.mean(depth_diff, dim=1)
        
        loss = pose_loss*depth_loss
        batch_loss = pose_batch_loss*depth_batch_loss
        return loss, batch_loss, gt_imgs, predicted_imgs
    
    elif(loss_method=="l2-pose"):
        mseLoss = nn.MSELoss(reduction='none')
        l2loss = mseLoss(Rs_predicted, Rs_gt)
        loss = torch.sum(l2loss)
        batch_loss = torch.sum(l2loss, dim=(1,2))
        return loss, batch_loss, gt_imgs, predicted_imgs

    elif(loss_method=="l1-clamped"):
        diff = torch.clamp(diff, 0.0, 20.0)
        loss = torch.mean(diff)
        return loss, torch.mean(diff, dim=1), gt_imgs, predicted_imgs

    elif(loss_method=="vsd"):
        outliers = torch.randn(diff.shape, device=renderer.device, requires_grad=True)
        outliers = ThauThreshold.apply(diff)
        total = torch.randn(diff.shape, device=renderer.device, requires_grad=True)
        total = NonZero.apply(diff)
        vsd_batch = torch.sum(outliers, dim=1)/torch.sum(total, dim=1)
        vsd_all = torch.sum(outliers)/torch.sum(total)
        return vsd_all, vsd_batch, gt_imgs, predicted_imgs

    elif(loss_method=="multiview"):
        loss = torch.mean(diff)
        return loss, torch.mean(diff, dim=1), gt_imgs, predicted_imgs

    elif(loss_method=="multiview-l2"):
        lossf = nn.MSELoss(reduction="none")
        loss = lossf(gt_imgs.flatten(start_dim=1), predicted_imgs.flatten(start_dim=1))
        loss = torch.mean(loss, dim=1)
        return torch.mean(loss), loss, gt_imgs, predicted_imgs

    print("Unknown loss specified")
    return -1, None, None, None
