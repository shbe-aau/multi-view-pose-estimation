import torch
import numpy as np
from utils.utils import *
from utils.tools import *
import torch.nn as nn
import configparser
#from pyquaternion import Quaternion

from pytorch3d.renderer import look_at_view_transform
from pytorch3d.renderer import look_at_rotation
from scipy.spatial.transform import Rotation as scipyR

from pytorch3d.transforms import Transform3d, Rotate

dbg_losses = False

def dbg(message, flag):
    if flag:
        print(message)

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

def Loss(predicted_poses, gt_poses, renderer, ts, mean, std, ids=[0], views=None, config=None, fixed_gt_images=None, eval_mode=False):
    Rs_gt = torch.tensor(np.stack(gt_poses), device=renderer.device,
                            dtype=torch.float32)
    if config is None:
        config = configparser.ConfigParser()

    loss_method = config.get('Training', 'LOSS', fallback='vsd-union')
    pose_rep = config.get('Training', 'POSE_REPRESENTATION', fallback='6d-pose')
    verbose = False

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
    else: # this version is for using loss with prerendered ref image and regular rot matrix for predicted pose
        #Rs_predicted = predicted_poses
        #Rs_predicted = torch.Tensor(Rs_predicted).to(renderer.device)
        gt_imgs = fixed_gt_images

    if(loss_method=="vsd-union"):
        depth_max = config.getfloat('Loss_parameters', 'DEPTH_MAX', fallback=30.0)
        pose_max = config.getfloat('Loss_parameters', 'POSE_MAX', fallback=40.0)
        num_views = len(views)
        gamma = config.getfloat('Loss_parameters', 'GAMMA', fallback=1.0 / num_views)
        pose_start = num_views
        pose_end = pose_start + 6

        # Prepare gt images
        gt_images = []
        predicted_images = []
        gt_imgs = renderer.renderBatch(Rs_gt, ts)

        losses = []
        confs = predicted_poses[:,:num_views]
        prev_poses = []
        pose_losses = []
        for i,v in enumerate(views):
            # Extract current pose and move to next one
            if fixed_gt_images is None:
                curr_pose = predicted_poses[:,pose_start:pose_end]
                Rs_predicted = compute_rotation_matrix_from_ortho6d(curr_pose)
            else:
                pose_matrix = predicted_poses[:,1:].reshape(1,3,3)
                #Rs_predicted = torch.Tensor(pose_matrix).to(renderer.device)
                Rs_predicted = pose_matrix
            pose_start = pose_end
            pose_end = pose_start + 6

            # Render predicted images
            imgs = renderer.renderBatch(Rs_predicted, ts)
            predicted_images.append(imgs)
            gt_images.append(gt_imgs)

            # Visiblity mask
            mask_gt = gt_imgs != 0
            mask_pd = imgs != 0
            mask_union = torch.zeros_like(gt_imgs)
            mask_union[mask_gt] = 1.0
            mask_union[mask_pd] = 1.0

            # Calculate loss
            diff = torch.abs(gt_imgs - imgs)
            diff = torch.clamp(diff, 0.0, depth_max)/depth_max
            batch_loss = torch.sum(diff*mask_union, dim=(1,2))/torch.sum(mask_union, dim=(1,2))

            batch_loss = (batch_loss*confs[:,i] + gamma*batch_loss)/2.0
            losses.append(batch_loss.unsqueeze(-1))

            # Calculate pose loss
            for k,p in enumerate(prev_poses):
                R = torch.matmul(p, torch.transpose(Rs_predicted, 1, 2))
                R_trace = torch.diagonal(R, dim1=-2, dim2=-1).sum(-1)
                theta = (R_trace - 1.0)/2.0
                epsilon=1e-5
                theta = torch.acos(torch.clamp(theta, -1 + epsilon, 1 - epsilon))
                degree = theta * (180.0/3.14159)

                pose_diff = 1.0 - (torch.clamp(degree, 0.0, pose_max)/pose_max)

                pose_batch_loss = pose_diff
                pose_losses.append(pose_batch_loss.unsqueeze(-1))

            # Add current predicted poses to list of previous ones
            prev_poses.append(Rs_predicted)



        # Concat different views
        gt_imgs = torch.cat(gt_images, dim=1)
        predicted_imgs = torch.cat(predicted_images, dim=1)
        losses = torch.cat(losses, dim=1)
        if(num_views == 1):
            pose_losses = torch.zeros_like(losses)
        else:
            pose_losses = torch.cat(pose_losses, dim=1)
        pose_losses = torch.mean(pose_losses, dim=1)
        depth_losses = torch.sum(losses, dim=1)

        dbg("depth loss {}".format(torch.mean(depth_losses)), dbg_losses)
        dbg("pose loss  {}".format(torch.mean(pose_losses)), dbg_losses)

        #batch_loss = batch_loss + batch_loss * (torch.mean(pose_losses, dim=1)-0.5)/2.0
        batch_loss = depth_losses + pose_losses
        batch_loss = batch_loss.unsqueeze(-1)
        loss = torch.mean(batch_loss) #losses) #+torch.mean(pose_losses)
        return loss, batch_loss, gt_imgs, predicted_imgs

    print("Unknown loss specified")
    return -1, None, None, None
