import torch
import numpy as np
from utils.utils import *
from utils.tools import *
import torch.nn as nn
#from pyquaternion import Quaternion

from pytorch3d.renderer import look_at_view_transform
from pytorch3d.renderer import look_at_rotation
from scipy.spatial.transform import Rotation as scipyR

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


# Truely random
# Based on: https://www.cmu.edu/biolphys/deserno/pdf/sphere_equi.pdf
def sphere_sampling():
    #z = np.random.uniform(low=-self.dist, high=self.dist, size=1)[0]
    z = np.random.uniform(low=-1, high=1, size=1)[0]
    theta_sample = np.random.uniform(low=0.0, high=2.0*np.pi, size=1)[0]
    x = np.sqrt((1**2 - z**2))*np.cos(theta_sample)
    y = np.sqrt((1**2 - z**2))*np.sin(theta_sample)

    cam_position = torch.tensor([x, y, z]).unsqueeze(0)
    if(z < 0):
        R = look_at_rotation(cam_position, up=((0, 0, -1),)).squeeze()
    else:
        R = look_at_rotation(cam_position, up=((0, 0, 1),)).squeeze()

    # Rotate in-plane
    rot_degrees = np.random.uniform(low=-90.0, high=90.0, size=1)
    rot = scipyR.from_euler('z', rot_degrees, degrees=True)
    rot_mat = torch.tensor(rot.as_matrix(), dtype=torch.float32)
    R = torch.matmul(R, rot_mat)
    R = R.squeeze()

    return R

def renderNormCat(Rs, ts, renderer, mean, std, views):
    images = []
    for v in views:
        # Render images
        Rs_new = torch.matmul(Rs, v.to(renderer.device))
        imgs = renderer.renderBatch(Rs_new, ts)
        imgs = (imgs-mean)/std
        images.append(imgs)
    return torch.cat(images, dim=1)


def renderMulti(Rs_gt, predicted_poses, ts, renderer):
    num_views = 4
    pred_images = []
    gt_images = []
    confidences = []
    pred_poses = []
    pose_index = num_views
    for v in np.arange(num_views):
        # Render groundtruth images
        gt_images.append(renderer.renderBatch(Rs_gt, ts))

        # Extract predicted poses
        end_index = pose_index + 6
        curr_poses = predicted_poses[:,pose_index:end_index]
        curr_poses = compute_rotation_matrix_from_ortho6d(curr_poses)
        pose_index = end_index
        pred_images.append(renderer.renderBatch(curr_poses, ts))
        pred_poses.append(curr_poses)

    # Extract confidences and perform softmax
    confidences.append(torch.nn.functional.softmax(predicted_poses[:,:4],dim=1))

    gt_images = torch.cat(gt_images, dim=1)
    pred_images = torch.cat(pred_images, dim=1)
    confidences = torch.cat(confidences, dim=1)
    pred_poses = torch.cat(pred_poses, dim=1)

    return gt_images, pred_images, confidences, pred_poses

def mat_theta( A, B ):
    """ comment cos between vectors or matrices """
    At = np.transpose(A)
    AB = np.dot(At, B)
    temp = (np.trace(AB) - 1) / 2
    if temp > 1:
        #print(temp)
        temp = 1
    if temp < -1:
        #print(temp)
        temp = -1
    return np.arccos(temp)


def Loss(predicted_poses, gt_poses, renderer, ts, mean, std, loss_method="diff", pose_rep="6d-pose", views=None, fixed_gt_images=None, loss_params=0.5):
    Rs_gt = torch.tensor(np.stack(gt_poses), device=renderer.device,
                            dtype=torch.float32)

    if('-random-multiview' in loss_method):
        for i,v in enumerate(views):
            if(i > 0):
                v = sphere_sampling()
            views[i] = v
        loss_method = loss_method.replace('-random-multiview','')

    if fixed_gt_images is None:
        if(pose_rep == '6d-pose'):
            Rs_predicted = compute_rotation_matrix_from_ortho6d(predicted_poses)
        elif(pose_rep == 'rot-mat'):
            batch_size = predicted_poses.shape[0]
            Rs_predicted = predicted_poses.view(batch_size, 3, 3)
        elif(pose_rep == 'quat'):
            Rs_predicted = compute_rotation_matrix_from_quaternion(predicted_poses)
        elif(pose_rep == 'euler'):
            Rs_predicted = look_at_rotation(predicted_poses).to(renderer.device)
            #Rs_predicted = compute_rotation_matrix_from_euler(predicted_poses)
        elif(pose_rep == 'axis-angle'):
            Rs_predicted = compute_rotation_matrix_from_axisAngle(predicted_poses)
        else:
            print("Unknown pose representation specified: ", pose_rep)
            return -1.0
        #gt_imgs = renderNormCat(Rs_gt, ts, renderer, mean, std, views)
    else: # this version is for using loss with prerendered ref image and regular rot matrix for predicted pose
        Rs_predicted = predicted_poses
        Rs_predicted = torch.Tensor(Rs_predicted).to(renderer.device)
        gt_imgs = fixed_gt_images

    #predicted_imgs = renderNormCat(Rs_predicted, ts, renderer, mean, std, views)
    #diff = torch.abs(gt_imgs - predicted_imgs).flatten(start_dim=1) # not needed for "multiview-l2"

    if(loss_method=="bce-loss"):
        loss = nn.BCEWithLogitsLoss(reduction="none")
        loss = loss(predicted_imgs.flatten(start_dim=1),gt_imgs.flatten(start_dim=1))
        loss = torch.mean(loss, dim=1)
        return torch.mean(loss), loss, gt_imgs, predicted_imgs

    elif(loss_method=="predicted-multiview"):
        gt_imgs, predicted_imgs, confs, pred_poses = renderMulti(Rs_gt, predicted_poses, ts, renderer)
        diff = torch.abs(gt_imgs - predicted_imgs).view(-1,4,128,128).flatten(start_dim=2)

        pred_poses = pred_poses.view(-1,4,3,3)
        pred_poses_rev = torch.flip(pred_poses,[1])

        # Calc pose loss
        pose_diff = 1.0 - torch.abs(pred_poses - pred_poses_rev).flatten(start_dim=1) #/2.0
        pose_loss = torch.mean(pose_diff)
        pose_batch_loss = torch.mean(pose_diff, dim=1)

        # Calc depth loss
        depth_diff = torch.clamp(diff, 0.0, 5.0)/5.0
        depth_diff = depth_diff*confs.view(-1,4,1) # Apply the confidence as weights
        depth_diff = depth_diff.flatten(start_dim=1)
        depth_loss = torch.mean(depth_diff)
        depth_batch_loss = torch.mean(depth_diff, dim=1)

        loss = loss_params*pose_loss + (1-loss_params)*depth_loss
        batch_loss = loss_params*pose_batch_loss + (1-loss_params)*depth_batch_loss
        return loss, batch_loss, gt_imgs, predicted_imgs

    elif(loss_method=="bce-loss-sum"):
        loss = nn.BCEWithLogitsLoss(reduction="none")
        loss = loss(predicted_imgs.flatten(start_dim=1),gt_imgs.flatten(start_dim=1))
        loss = torch.sum(loss, dim=1)
        return torch.sum(loss), loss, gt_imgs, predicted_imgs

    elif(loss_method=="pose-mul-depth"):
        # Calc pose loss
        mseLoss = nn.MSELoss(reduction='none')
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

    elif(loss_method=="pose-plus-depth"):
        # Calc pose loss
        mseLoss = nn.MSELoss(reduction='none')
        pose_diff = torch.abs(Rs_gt - Rs_predicted).flatten(start_dim=1)/2.0
        pose_loss = torch.mean(pose_diff)
        pose_batch_loss = torch.mean(pose_diff, dim=1)

        # Calc depth loss
        depth_diff = torch.clamp(diff, 0.0, 20.0)/20.0
        depth_loss = torch.mean(depth_diff)
        depth_batch_loss = torch.mean(depth_diff, dim=1)

        loss = loss_params*pose_loss + (1-loss_params)*depth_loss
        batch_loss = loss_params*pose_batch_loss + (1-loss_params)*depth_batch_loss
        return loss, batch_loss, gt_imgs, predicted_imgs

    elif(loss_method=="l2-pose"):
        mseLoss = nn.MSELoss(reduction='none')
        l2loss = mseLoss(Rs_predicted, Rs_gt)/6.0
        loss = torch.sum(l2loss)
        batch_loss = torch.sum(l2loss, dim=(1,2))
        return loss, batch_loss, gt_imgs, predicted_imgs

    elif(loss_method=="l1-depth"):
        loss = torch.mean(diff)
        return loss, torch.mean(diff, dim=1), gt_imgs, predicted_imgs

    elif(loss_method=="l1-clamped"):
        diff = torch.clamp(diff, 0.0, loss_params)/loss_params
        loss = torch.mean(diff)
        batch_loss = torch.mean(diff, dim=1)
        return loss, batch_loss, gt_imgs, predicted_imgs

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

    elif(loss_method=="z-diff"):
        z_predicted = Rs_predicted[:,2,:]
        z_gt = Rs_gt[:,2,:]
        batch_loss = (1.0 - (z_predicted*z_gt).sum(-1))/2.0
        loss = torch.mean(batch_loss)
        return loss, batch_loss, gt_imgs, predicted_imgs

    elif(loss_method=="sil-ratio"):
        loss = torch.sum(diff) / torch.sum(gt_imgs) #(diff.shape[0]*diff.shape[1])
        batch_loss = torch.sum(diff, dim=1) / torch.sum(gt_imgs, dim=(1,2)) #(diff.shape[1])
        return loss, batch_loss, gt_imgs, predicted_imgs

    elif(loss_method=="sil-ratio-plus-zdiff"):
        ratio_loss = torch.sum(diff) / torch.sum(gt_imgs)
        ratio_batch_loss = torch.sum(diff, dim=1) / torch.sum(gt_imgs, dim=(1,2))

        z_predicted = Rs_predicted[:,2,:]
        z_gt = Rs_gt[:,2,:]
        z_batch_loss = (1.0 - (z_predicted*z_gt).sum(-1))/2.0
        z_loss = torch.mean(z_batch_loss)

        loss = loss_params*z_loss + (1-loss_params)*ratio_loss
        batch_loss = loss_params*z_batch_loss + (1-loss_params)*ratio_batch_loss
        return loss, batch_loss, gt_imgs, predicted_imgs

    elif(loss_method=="sil-ratio-mul-zdiff"):
        ratio_loss = torch.sum(diff) / torch.sum(gt_imgs)
        ratio_batch_loss = torch.sum(diff, dim=1) / torch.sum(gt_imgs, dim=(1,2))

        z_predicted = Rs_predicted[:,2,:]
        z_gt = Rs_gt[:,2,:]
        z_batch_loss = (1.0 - (z_predicted*z_gt).sum(-1))/2.0
        z_loss = torch.mean(z_batch_loss)

        loss = z_loss*ratio_loss
        batch_loss = z_batch_loss*ratio_batch_loss
        return loss, batch_loss, gt_imgs, predicted_imgs

    # elif(loss_method=="qpose"):
    #     diff = []
    #     Rs_gt = Rs_gt.detach().cpu().numpy()
    #     Rs_predicted = Rs_predicted.detach().cpu().numpy()
    #     for i in range(len(Rs_gt)):
    #         r = scipyR.from_matrix(Rs_gt[i])
    #         q_gt = Quaternion(r.as_quat())
    #         r = scipyR.from_matrix(Rs_predicted[i])
    #         q_pred = Quaternion(r.as_quat())
    #         q_diff = Quaternion.absolute_distance(q_gt, q_pred)
    #         #q_diff = Quaternion.distance(q_gt, q_pred)
    #         #q_diff = Quaternion.sym_distance(q_gt, q_pred)
    #         diff.append(q_diff)
    #     loss = np.mean(diff)
    #     batch_loss = np.mean(diff)
    #     loss = torch.tensor(loss, device=renderer.device,
    #                             dtype=torch.float32)
    #     batch_loss = torch.tensor(batch_loss, device=renderer.device,
    #                             dtype=torch.float32)
    #     return loss, batch_loss, gt_imgs, predicted_imgs

    elif(loss_method=="mat-theta"):
        diff = []
        Rs_gt = Rs_gt.detach().cpu().numpy()
        Rs_predicted = Rs_predicted.detach().cpu().numpy()
        for i in range(len(Rs_gt)):
            theta = mat_theta(Rs_gt[i], Rs_predicted[i])
            diff.append(theta)
        loss = np.mean(diff)
        batch_loss = np.mean(diff)
        loss = torch.tensor(loss, device=renderer.device,
                                dtype=torch.float32)
        batch_loss = torch.tensor(batch_loss, device=renderer.device,
                                dtype=torch.float32)
        return loss, batch_loss, gt_imgs, predicted_imgs


    print("Unknown loss specified")
    return -1, None, None, None
