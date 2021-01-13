import torch
import numpy as np
from utils.utils import *
from utils.tools import *
import torch.nn as nn
#from pyquaternion import Quaternion

from pytorch3d.renderer import look_at_view_transform
from pytorch3d.renderer import look_at_rotation
from scipy.spatial.transform import Rotation as scipyR

from pytorch3d.transforms import Transform3d, Rotate

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
    return None
    images = []
    for v in views:
        # Render images
        Rs_new = torch.matmul(Rs, v.to(renderer.device))
        imgs = renderer.renderBatch(Rs_new, ts)
        imgs = (imgs-mean)/std
        images.append(imgs)
    return torch.cat(images, dim=1)


def renderMulti(Rs_gt, predicted_poses, ts, renderer, views):
    num_views = len(views)
    pred_images = []
    gt_images = []
    confidences = []
    pred_poses = []
    pose_index = num_views
    for i,v in enumerate(views): #np.arange(num_views):
        # Render groundtruth images
        gt_images.append(renderer.renderBatch(Rs_gt, ts))

        # Extract predicted pose
        end_index = pose_index + 6
        curr_poses = predicted_poses[:,pose_index:end_index]
        curr_poses = compute_rotation_matrix_from_ortho6d(curr_poses)
        pose_index = end_index

        # Render images
        imgs = renderer.renderBatch(curr_poses, ts)
        pred_images.append(imgs)
        pred_poses.append(curr_poses)

    # Extract confidences and perform softmax
    confidences.append(torch.nn.functional.softmax(predicted_poses[:,:num_views],dim=1))

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


def Loss(predicted_poses, gt_poses, renderer, ts, mean, std, loss_method="diff", pose_rep="6d-pose", views=None, fixed_gt_images=None, loss_params=0.5, eval_mode=False):
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
        gt_imgs = renderNormCat(Rs_gt, ts, renderer, mean, std, views)
    else: # this version is for using loss with prerendered ref image and regular rot matrix for predicted pose
        Rs_predicted = predicted_poses
        Rs_predicted = torch.Tensor(Rs_predicted).to(renderer.device)
        gt_imgs = fixed_gt_images

    predicted_imgs = renderNormCat(Rs_predicted, ts, renderer, mean, std, views)
    #diff = torch.abs(gt_imgs - predicted_imgs).flatten(start_dim=1) # not needed for "multiview-l2"

    if(loss_method=="bce-loss"):
        loss = nn.BCEWithLogitsLoss(reduction="none")
        loss = loss(predicted_imgs.flatten(start_dim=1),gt_imgs.flatten(start_dim=1))
        loss = torch.mean(loss, dim=1)
        return torch.mean(loss), loss, gt_imgs, predicted_imgs

    elif(loss_method=="predictive-multiview"):
        num_views = len(views)
        gt_imgs, predicted_imgs, confs, pred_poses = renderMulti(Rs_gt, predicted_poses, ts, renderer, views)
        diff = torch.abs(gt_imgs - predicted_imgs).view(-1,num_views,128,128).flatten(start_dim=2)

        pred_poses = pred_poses.view(-1,num_views,3,3)
        pred_poses_rev = torch.flip(pred_poses,[1])

        # Calc pose loss
        pose_diff = 1.0 - torch.abs(pred_poses - pred_poses_rev).flatten(start_dim=1) #/2.0
        pose_loss = torch.mean(pose_diff)
        pose_batch_loss = torch.mean(pose_diff, dim=1)

        # Calc depth loss
        depth_diff = torch.clamp(diff, 0.0, 20.0)/20.0
        depth_diff = depth_diff*confs.view(-1,num_views,1) # Apply the confidence as weights
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

    elif(loss_method=="trace-pose"):
        num_views = len(views)
        pose_start = num_views
        pose_end = pose_start + 6
        Rs_predicted = compute_rotation_matrix_from_ortho6d(predicted_poses[:,pose_start:pose_end])

        gt_imgs = renderer.renderBatch(Rs_gt, ts)
        predicted_imgs = renderer.renderBatch(Rs_predicted, ts)

        R = torch.matmul(Rs_gt, torch.transpose(Rs_predicted, 1, 2))
        R_trace = torch.diagonal(R, dim1=-2, dim2=-1).sum(-1)
        theta = (R_trace - 1.0)/2.0
        epsilon=1e-5
        theta = torch.acos(torch.clamp(theta, -1 + epsilon, 1 - epsilon))
        degree = theta * (180.0/3.14159)
        batch_loss = degree/180.0
        loss = torch.sum(degree/180.0)
        print(gt_imgs.shape)
        return loss, batch_loss, gt_imgs, predicted_imgs

    elif(loss_method=="l2-pose"):
        num_views = len(views)
        pose_start = num_views
        pose_end = pose_start + 6
        Rs_predicted = compute_rotation_matrix_from_ortho6d(predicted_poses[:,pose_start:pose_end])

        gt_imgs = renderer.renderBatch(Rs_gt, ts)
        predicted_imgs = renderer.renderBatch(Rs_predicted, ts)

        mseLoss = nn.MSELoss(reduction='none')
        l2loss = mseLoss(Rs_predicted, Rs_gt)/6.0
        loss = torch.sum(l2loss)
        batch_loss = torch.sum(l2loss, dim=(1,2))
        return loss, batch_loss, gt_imgs, predicted_imgs

    elif(loss_method=="l1-depth"):
        loss = torch.mean(diff)
        return loss, torch.mean(diff, dim=1), gt_imgs, predicted_imgs

    elif(loss_method=="depth-masked"):
        mask_gt = gt_imgs.flatten(start_dim=1) > 0
        mask_pred = predicted_imgs.flatten(start_dim=1) > 0
        mask = mask_gt * mask_pred
        masked = diff*mask

        # jaccard index
        jaccard = torch.sum(mask, dim=1)/(torch.sum(mask_gt, dim=1) + torch.sum(mask_pred, dim=1) - torch.sum(mask, dim=1))
        jaccard_inv = 1.0 - jaccard
        batch_j = jaccard_inv
        j = torch.mean(batch_j)

        loss = torch.sum(masked)/torch.sum(mask)
        loss = loss*j
        batch_loss = torch.sum(masked, dim=1)/torch.sum(mask, dim=1)
        batch_loss = batch_loss*batch_j
        return loss, batch_loss, gt_imgs, predicted_imgs

    elif(loss_method=="l1-clamped"):
        diff = torch.clamp(diff, 0.0, loss_params)/loss_params
        loss = torch.mean(diff)
        batch_loss = torch.mean(diff, dim=1)
        return loss, batch_loss, gt_imgs, predicted_imgs


    elif(loss_method=="l1-clamped-smooth"):
        num_views = len(views)
        pose_start = num_views

        Rs_predicted = compute_rotation_matrix_from_ortho6d(predicted_poses[:,pose_start:pose_start+6])
        gt_imgs = renderer.renderBatch(Rs_gt, ts)
        predicted_imgs = renderer.renderBatch(Rs_predicted, ts)

        #lossf = nn.SmoothL1Loss(reduction="none", beta=loss_params*0.5)
        #diff = lossf(gt_imgs, predicted_imgs).flatten(start_dim=1)
        diff = (gt_imgs - predicted_imgs)**2
        diff = diff.flatten(start_dim=1)
        diff = torch.clamp(diff, 0.0, loss_params)/loss_params
        loss = torch.mean(diff)
        batch_loss = torch.mean(diff, dim=1).unsqueeze(-1)
        return loss, batch_loss, gt_imgs, predicted_imgs



    elif(loss_method=="depth-fixed-view"):
        num_views = len(views)
        pose_start = num_views
        pose_end = pose_start + 6

        # Prepare gt images
        gt_images = []
        predicted_images = []
        gt_imgs = renderer.renderBatch(Rs_gt, ts)

        losses = []
        confs = predicted_poses[:,:num_views]
        prev_pose = None
        for i,v in enumerate(views):
            # Extract current pose and move to next one
            curr_pose = predicted_poses[:,pose_start:pose_end]
            Rs_predicted = compute_rotation_matrix_from_ortho6d(curr_pose)
            #pose_start = pose_end
            #pose_end = pose_start + 6

            # Prepare rotation matrix
            Rs_predicted = Rs_predicted.permute(0,2,1)
            Rs_predicted = torch.matmul(Rs_predicted, v.to(renderer.device))
            Rs_predicted = Rs_predicted.permute(0,2,1)

            # Render predicted images
            imgs = renderer.renderBatch(Rs_predicted, ts)
            predicted_images.append(imgs)
            gt_images.append(gt_imgs)

            # Calculate loss
            diff = torch.abs(gt_imgs - imgs).flatten(start_dim=1)
            batch_loss = torch.mean(diff, dim=1)
            batch_loss = batch_loss*confs[:,i]
            losses.append(batch_loss.unsqueeze(-1))

            prev_pose = Rs_predicted

        # Concat different views
        gt_imgs = torch.cat(gt_images, dim=1)
        predicted_imgs = torch.cat(predicted_images, dim=1)
        losses = torch.cat(losses, dim=1)

        batch_loss = losses #torch.mean(losses, dim=1)
        loss = torch.mean(losses)
        return loss, batch_loss, gt_imgs, predicted_imgs

    elif(loss_method=="vsd-fixed-view"):
        num_views = len(views)
        pose_start = num_views
        pose_end = pose_start + 6

        # Prepare gt images
        gt_images = []
        predicted_images = []
        gt_imgs = renderer.renderBatch(Rs_gt, ts)

        losses = []
        confs = predicted_poses[:,:num_views]
        for i,v in enumerate(views):
            # Extract current pose and move to next one
            curr_pose = predicted_poses[:,pose_start:pose_end]
            Rs_predicted = compute_rotation_matrix_from_ortho6d(curr_pose)
            #pose_start = pose_end
            #pose_end = pose_start + 6

            # Prepare rotation matrix
            Rs_predicted = Rs_predicted.permute(0,2,1)
            Rs_predicted = torch.matmul(Rs_predicted, v.to(renderer.device))
            Rs_predicted = Rs_predicted.permute(0,2,1)

            # Render predicted images
            imgs = renderer.renderBatch(Rs_predicted, ts)
            predicted_images.append(imgs)
            gt_images.append(gt_imgs)

            # Calculate VSD depth loss
            diff = torch.abs(gt_imgs - imgs)
            non_zero = torch.clamp(gt_imgs, 0, 1)
            inliers = torch.clamp(diff, 0, 20.0)/20.0 * non_zero
            inliers = inliers * non_zero
            inliers = torch.sum(inliers, dim=(1,2))
            total_gt = torch.sum(non_zero, dim=(1,2))
            batch_loss = (inliers/total_gt)

            #batch_loss = batch_loss*confs[:,i] + (1.0/num_views)*batch_loss
            batch_loss = batch_loss*confs[:,i] # + (1.0/num_views)*batch_loss
            losses.append(batch_loss.unsqueeze(-1))

        # Concat different views
        gt_imgs = torch.cat(gt_images, dim=1)
        predicted_imgs = torch.cat(predicted_images, dim=1)
        losses = torch.cat(losses, dim=1)

        batch_loss = torch.mean(losses, dim=1)
        batch_loss = batch_loss.unsqueeze(-1)
        loss = torch.mean(batch_loss) #losses) #+torch.mean(pose_losses)
        return loss, batch_loss, gt_imgs, predicted_imgs

    elif(loss_method=="depth-clamped-predicted-view"):
        num_views = len(views)
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
            curr_pose = predicted_poses[:,pose_start:pose_end]
            Rs_predicted = compute_rotation_matrix_from_ortho6d(curr_pose)
            pose_start = pose_end
            pose_end = pose_start + 6

            # Render predicted images
            imgs = renderer.renderBatch(Rs_predicted, ts)
            predicted_images.append(imgs)
            gt_images.append(gt_imgs)

            # Calculate depth loss
            diff = torch.abs(gt_imgs - imgs).flatten(start_dim=1)
            diff = torch.clamp(diff, 0.0, loss_params)/loss_params
            batch_loss = torch.mean(diff, dim=1)
            #print("weighted: ", torch.mean(batch_loss*confs[:,i]))
            #print("averaged: ", torch.mean((1.0/num_views)*batch_loss))
            batch_loss = batch_loss*confs[:,i] + (1.0/num_views)*batch_loss
            losses.append(batch_loss.unsqueeze(-1))

            # Calculate pose loss
            for k,p in enumerate(prev_poses):
                mseLoss = nn.MSELoss(reduction='non e')
                pose_diff = torch.abs(p - Rs_predicted).flatten(start_dim=1)
                pose_max = 0.25
                pose_diff = 1.0 - (torch.clamp(pose_diff, 0.0, pose_max)/pose_max)
                pose_batch_loss = torch.mean(pose_diff, dim=1)
                pose_losses.append(pose_batch_loss.unsqueeze(-1))

            # Add current predicted poses to list of previous ones
            prev_poses.append(Rs_predicted)


        # Concat different views
        gt_imgs = torch.cat(gt_images, dim=1)
        predicted_imgs = torch.cat(predicted_images, dim=1)
        losses = torch.cat(losses, dim=1)
        pose_losses = torch.cat(pose_losses, dim=1)**3

        #print("depth loss ", torch.mean(losses, dim=1))
        #print("pose loss ", torch.mean(pose_losses, dim=1))

        batch_loss = torch.mean(losses, dim=1) + torch.mean(pose_losses, dim=1)
        batch_loss = batch_loss.unsqueeze(-1)
        #batch_loss = losses
        loss = torch.mean(losses)+torch.mean(pose_losses)
        return loss, batch_loss, gt_imgs, predicted_imgs

    elif(loss_method=="vsd-predicted-view"):
        num_views = len(views)
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
            curr_pose = predicted_poses[:,pose_start:pose_end]
            Rs_predicted = compute_rotation_matrix_from_ortho6d(curr_pose)
            pose_start = pose_end
            pose_end = pose_start + 6

            # Render predicted images
            imgs = renderer.renderBatch(Rs_predicted, ts)
            predicted_images.append(imgs)
            gt_images.append(gt_imgs)

            # Calculate VSD depth loss
            diff = torch.abs(gt_imgs - imgs)
            non_zero = torch.clamp(gt_imgs, 0, 1)
            inliers = torch.clamp(diff, 0, loss_params)/loss_params * non_zero

            inliers = inliers * non_zero
            inliers = torch.sum(inliers, dim=(1,2))
            total_gt = torch.sum(non_zero, dim=(1,2))
            batch_loss = (inliers/total_gt)

            #print("weighted: ", torch.mean(batch_loss*confs[:,i]))
            #print("averaged: ", torch.mean((1.0/num_views)*batch_loss))
            batch_loss = batch_loss*confs[:,i] + (1.0/num_views)*batch_loss
            #batch_loss = (1.0/num_views)*batch_loss
            losses.append(batch_loss.unsqueeze(-1))

            # Calculate pose loss
            for k,p in enumerate(prev_poses):
                pose_diff = torch.abs(p - Rs_predicted).flatten(start_dim=1)
                pose_max = 0.6
                pose_diff = 1.0 - (torch.clamp(pose_diff, 0.0, pose_max)/pose_max)
                pose_batch_loss = torch.mean(pose_diff, dim=1)
                pose_losses.append(pose_batch_loss.unsqueeze(-1))

            # Add current predicted poses to list of previous ones
            prev_poses.append(Rs_predicted)


        # Concat different views
        gt_imgs = torch.cat(gt_images, dim=1)
        predicted_imgs = torch.cat(predicted_images, dim=1)
        losses = torch.cat(losses, dim=1)*(1.0/5.0)
        pose_losses = torch.cat(pose_losses, dim=1)*0.0 #**3

        #print("depth loss ", torch.mean(losses, dim=1))
        #print("pose loss ", torch.mean(pose_losses, dim=1))

        # Conf loss
        conf_loss = F.softmax(losses, dim=1)
        conf_loss = torch.abs(confs - conf_loss)
        #print("Conf: ", conf_loss)
        #print("VSD: ", losses)
        #print("Pose: ", pose_losses)
        #losses = losses + conf_loss

        batch_loss = torch.mean(losses, dim=1) + torch.mean(pose_losses, dim=1)
        batch_loss = batch_loss.unsqueeze(-1)
        loss = torch.mean(losses) +torch.mean(pose_losses)
        return loss, batch_loss, gt_imgs, predicted_imgs


    elif(loss_method=="vsd-predicted-view-degrees"):
        num_views = len(views)
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
            curr_pose = predicted_poses[:,pose_start:pose_end]
            Rs_predicted = compute_rotation_matrix_from_ortho6d(curr_pose)
            pose_start = pose_end
            pose_end = pose_start + 6

            # Render predicted images
            imgs = renderer.renderBatch(Rs_predicted, ts)
            predicted_images.append(imgs)
            gt_images.append(gt_imgs)

            # Calculate VSD depth loss - old
            depth_max = 20.0
            diff = torch.abs(gt_imgs - imgs)
            #non_zero = torch.clamp(gt_imgs, 0, 1) # Non zero in gt only
            non_zero = torch.clamp(gt_imgs + imgs, 0, 1) # Non zero in either gt or prediction
            inliers = torch.clamp(diff, 0, depth_max)/depth_max * non_zero
            inliers = inliers * non_zero #remove?
            inliers = torch.sum(inliers, dim=(1,2))
            total_gt = torch.sum(non_zero, dim=(1,2))
            batch_loss = (inliers/total_gt)

            # # Calculate VSD depth loss - new, better for obj 10 and 12?
            # depth_max = 20.0
            # diff = torch.abs(gt_imgs - imgs)
            # non_zero = torch.clamp(gt_imgs + imgs, 0, 1) # Non zero in either gt or prediction
            # inliers = torch.clamp(diff, 0, depth_max)/depth_max * non_zero
            # inliers = torch.sum(inliers, dim=(1,2))
            # non_zero_gt = torch.clamp(gt_imgs, 0, 1) # Non zero in gt only
            # total_gt = torch.sum(non_zero_gt, dim=(1,2))
            # batch_loss = (inliers/total_gt)

            batch_loss = batch_loss*confs[:,i] + (1.0/num_views)*batch_loss
            losses.append(batch_loss.unsqueeze(-1))

            # Calculate pose loss
            for k,p in enumerate(prev_poses):
                R = torch.matmul(p, torch.transpose(Rs_predicted, 1, 2))
                R_trace = torch.diagonal(R, dim1=-2, dim2=-1).sum(-1)
                theta = (R_trace - 1.0)/2.0
                epsilon=1e-5
                theta = torch.acos(torch.clamp(theta, -1 + epsilon, 1 - epsilon))
                degree = theta * (180.0/3.14159)

                pose_max = loss_params
                pose_diff = 1.0 - (torch.clamp(degree, 0.0, pose_max)/pose_max)

                pose_batch_loss = pose_diff
                pose_losses.append(pose_batch_loss.unsqueeze(-1))

            # Add current predicted poses to list of previous ones
            prev_poses.append(Rs_predicted)


        # Concat different views
        gt_imgs = torch.cat(gt_images, dim=1)
        predicted_imgs = torch.cat(predicted_images, dim=1)
        losses = torch.cat(losses, dim=1)
        pose_losses = torch.cat(pose_losses, dim=1)

        #print("depth loss ", torch.mean(losses, dim=1))
        #print("pose loss ", torch.mean(pose_losses, dim=1))

        batch_loss = torch.mean(losses, dim=1)
        batch_loss = batch_loss + batch_loss * (torch.mean(pose_losses, dim=1)-0.5)/2.0
        #print("VSD: ", torch.mean(losses, dim=1))
        #print("pose: ", torch.mean(pose_losses, dim=1))
        batch_loss = batch_loss.unsqueeze(-1)
        loss = torch.mean(batch_loss) #losses) #+torch.mean(pose_losses)
        return loss, batch_loss, gt_imgs, predicted_imgs


    elif(loss_method=="vsd-predicted-view-degrees-hard"):
        num_views = len(views)
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
            curr_pose = predicted_poses[:,pose_start:pose_end]
            Rs_predicted = compute_rotation_matrix_from_ortho6d(curr_pose)
            pose_start = pose_end
            pose_end = pose_start + 6

            # Render predicted images
            imgs = renderer.renderBatch(Rs_predicted, ts)
            predicted_images.append(imgs)
            gt_images.append(gt_imgs)

            # Calculate VSD depth loss - old
            depth_max = 20.0
            diff = torch.abs(gt_imgs - imgs)
            #non_zero = torch.clamp(gt_imgs, 0, 1) # Non zero in gt only
            non_zero = torch.clamp(gt_imgs + imgs, 0, 1) # Non zero in either gt or prediction
            inliers = torch.clamp(diff, 0, depth_max)/depth_max * non_zero
            inliers = inliers * non_zero #remove?
            inliers = torch.sum(inliers, dim=(1,2))
            total_gt = torch.sum(non_zero, dim=(1,2))
            batch_loss = (inliers/total_gt)

            # # Calculate VSD depth loss - new, better for obj 10 and 12?
            # depth_max = 20.0
            # diff = torch.abs(gt_imgs - imgs)
            # non_zero = torch.clamp(gt_imgs + imgs, 0, 1) # Non zero in either gt or prediction
            # inliers = torch.clamp(diff, 0, depth_max)/depth_max * non_zero
            # inliers = torch.sum(inliers, dim=(1,2))
            # non_zero_gt = torch.clamp(gt_imgs, 0, 1) # Non zero in gt only
            # total_gt = torch.sum(non_zero_gt, dim=(1,2))
            # batch_loss = (inliers/total_gt)

            batch_loss = batch_loss*confs[:,i] + (1.0/num_views)*batch_loss
            losses.append(batch_loss.unsqueeze(-1))

            # Calculate pose loss
            for k,p in enumerate(prev_poses):
                R = torch.matmul(p, torch.transpose(Rs_predicted, 1, 2))
                R_trace = torch.diagonal(R, dim1=-2, dim2=-1).sum(-1)
                theta = (R_trace - 1.0)/2.0
                epsilon=1e-5
                theta = torch.acos(torch.clamp(theta, -1 + epsilon, 1 - epsilon))
                degree = theta * (180.0/3.14159)

                pose_max = loss_params
                pose_diff = 1.0 - (torch.clamp(degree, 0.0, pose_max)/pose_max)

                pose_batch_loss = pose_diff
                pose_losses.append(pose_batch_loss.unsqueeze(-1))

            # Add current predicted poses to list of previous ones
            prev_poses.append(Rs_predicted)


        # Concat different views
        gt_imgs = torch.cat(gt_images, dim=1)
        predicted_imgs = torch.cat(predicted_images, dim=1)
        losses = torch.cat(losses, dim=1)
        pose_losses = torch.cat(pose_losses, dim=1)

        #print("depth loss ", torch.mean(losses, dim=1))
        #print("pose loss ", torch.mean(pose_losses, dim=1))

        batch_loss = torch.mean(losses, dim=1)
        batch_loss = batch_loss + batch_loss * (torch.mean(pose_losses, dim=1)-0.5)/2.0
        #print("VSD: ", torch.mean(losses, dim=1))
        #print("pose: ", torch.mean(pose_losses, dim=1))

        # Only back-prop the hardest samples
        if(eval_mode==False):
            hard_ratio = 4.0
            k = int(batch_loss.shape[0]/hard_ratio)
            print("batch_loss: ", batch_loss.shape)
            top_val, top_ind = torch.topk(batch_loss, k)
            print("indices: ", top_ind)
            batch_loss = batch_loss[top_ind]
            predicted_imgs = predicted_imgs[top_ind]
            gt_imgs = gt_imgs[top_ind]
            print("images: ", predicted_imgs.shape)
            print("gt images: ", gt_imgs.shape)
            print("Only using the {0} hardest samples!".format(batch_loss.shape[0]))

        batch_loss = batch_loss.unsqueeze(-1)
        loss = torch.mean(batch_loss) #losses) #+torch.mean(pose_losses)
        return loss, batch_loss, gt_imgs, predicted_imgs, top_ind


    elif(loss_method=="vsd-predicted-view-fixed"):
        num_views = len(views)
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
            curr_pose = predicted_poses[:,pose_start:pose_end]
            Rs_predicted = compute_rotation_matrix_from_ortho6d(curr_pose)
            pose_start = pose_end
            pose_end = pose_start + 6

            # Render predicted images
            imgs = renderer.renderBatch(Rs_predicted, ts)
            predicted_images.append(imgs)
            gt_images.append(gt_imgs)

            # Calculate VSD depth loss - new, better for obj 10 and 12?
            depth_max = 20.0
            diff = torch.abs(gt_imgs - imgs)
            non_zero = torch.clamp(gt_imgs + imgs, 0, 1) # Non zero in either gt or prediction
            inliers = torch.clamp(diff, 0, depth_max)/depth_max * non_zero
            inliers = torch.sum(inliers, dim=(1,2))
            non_zero_gt = torch.clamp(gt_imgs, 0, 1)
            non_zero_pred = torch.clamp(imgs, 0, 1)
            non_zero_both = torch.mul(non_zero_gt,non_zero_pred) # Non zero in both gt and prediction
            total_gt = torch.sum(non_zero_both, dim=(1,2))
            batch_loss = (inliers/total_gt)

            batch_loss = batch_loss*confs[:,i] + (1.0/num_views)*batch_loss
            losses.append(batch_loss.unsqueeze(-1))

            # Calculate pose loss
            for k,p in enumerate(prev_poses):
                R = torch.matmul(p, torch.transpose(Rs_predicted, 1, 2))
                R_trace = torch.diagonal(R, dim1=-2, dim2=-1).sum(-1)
                theta = (R_trace - 1.0)/2.0
                epsilon=1e-5
                theta = torch.acos(torch.clamp(theta, -1 + epsilon, 1 - epsilon))
                degree = theta * (180.0/3.14159)

                pose_max = loss_params
                pose_diff = 1.0 - (torch.clamp(degree, 0.0, pose_max)/pose_max)

                pose_batch_loss = pose_diff
                pose_losses.append(pose_batch_loss.unsqueeze(-1))

            # Add current predicted poses to list of previous ones
            prev_poses.append(Rs_predicted)


        # Concat different views
        gt_imgs = torch.cat(gt_images, dim=1)
        predicted_imgs = torch.cat(predicted_images, dim=1)
        losses = torch.cat(losses, dim=1)
        pose_losses = torch.cat(pose_losses, dim=1)

        #print("depth loss ", torch.mean(losses, dim=1))
        #print("pose loss ", torch.mean(pose_losses, dim=1))

        batch_loss = torch.mean(losses, dim=1)
        batch_loss = batch_loss + batch_loss * (torch.mean(pose_losses, dim=1)-0.5)/2.0
        #print("VSD: ", torch.mean(losses, dim=1))
        #print("pose: ", torch.mean(pose_losses, dim=1))
        batch_loss = batch_loss.unsqueeze(-1)
        loss = torch.mean(batch_loss) #losses) #+torch.mean(pose_losses)
        return loss, batch_loss, gt_imgs, predicted_imgs


    elif(loss_method=="vsd-predicted-view-log"):
        num_views = len(views)
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
            curr_pose = predicted_poses[:,pose_start:pose_end]
            Rs_predicted = compute_rotation_matrix_from_ortho6d(curr_pose)
            pose_start = pose_end
            pose_end = pose_start + 6

            # Render predicted images
            imgs = renderer.renderBatch(Rs_predicted, ts)
            predicted_images.append(imgs)
            gt_images.append(gt_imgs)

            # Calculate VSD depth loss
            diff = torch.abs(gt_imgs - imgs)
            #non_zero = torch.clamp(gt_imgs, 0, 1) # Non zero in gt only
            non_zero = torch.clamp(gt_imgs + imgs, 0, 1) # Non zero in either gt or prediction
            inliers = torch.clamp(torch.log10(diff+1e-9)/3.0, 0.0, 1.0) * non_zero
            inliers = inliers * non_zero #remove?
            inliers = torch.sum(inliers, dim=(1,2))
            total_gt = torch.sum(non_zero, dim=(1,2))
            batch_loss = (inliers/total_gt)

            batch_loss = batch_loss*confs[:,i] + (1.0/num_views)*batch_loss
            losses.append(batch_loss.unsqueeze(-1))

            # Calculate pose loss
            for k,p in enumerate(prev_poses):
                R = torch.matmul(p, torch.transpose(Rs_predicted, 1, 2))
                R_trace = torch.diagonal(R, dim1=-2, dim2=-1).sum(-1)
                theta = (R_trace - 1.0)/2.0
                epsilon=1e-5
                theta = torch.acos(torch.clamp(theta, -1 + epsilon, 1 - epsilon))
                degree = theta * (180.0/3.14159)

                pose_max = loss_params
                pose_diff = 1.0 - (torch.clamp(degree, 0.0, pose_max)/pose_max)

                pose_batch_loss = pose_diff
                pose_losses.append(pose_batch_loss.unsqueeze(-1))

            # Add current predicted poses to list of previous ones
            prev_poses.append(Rs_predicted)


        # Concat different views
        gt_imgs = torch.cat(gt_images, dim=1)
        predicted_imgs = torch.cat(predicted_images, dim=1)
        losses = torch.cat(losses, dim=1)
        pose_losses = torch.cat(pose_losses, dim=1)

        #print("depth loss ", torch.mean(losses, dim=1))
        #print("pose loss ", torch.mean(pose_losses, dim=1))

        batch_loss = torch.mean(losses, dim=1)
        batch_loss = batch_loss + batch_loss * (torch.mean(pose_losses, dim=1)-0.5)/2.0
        #print("VSD: ", torch.mean(losses, dim=1))
        #print("pose: ", torch.mean(pose_losses, dim=1))
        batch_loss = batch_loss.unsqueeze(-1)
        loss = torch.mean(batch_loss) #losses) #+torch.mean(pose_losses)
        return loss, batch_loss, gt_imgs, predicted_imgs


    elif(loss_method=="vsd-predicted-view-log-fixed"):
        num_views = len(views)
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
            curr_pose = predicted_poses[:,pose_start:pose_end]
            Rs_predicted = compute_rotation_matrix_from_ortho6d(curr_pose)
            pose_start = pose_end
            pose_end = pose_start + 6

            # Render predicted images
            imgs = renderer.renderBatch(Rs_predicted, ts)
            predicted_images.append(imgs)
            gt_images.append(gt_imgs)

            # Calculate VSD depth loss
            diff = torch.abs(gt_imgs - imgs)
            #non_zero = torch.clamp(gt_imgs, 0, 1) # Non zero in gt only
            non_zero = torch.clamp(gt_imgs + imgs, 0, 1) # Non zero in either gt or prediction
            inliers = torch.clamp(torch.log10(diff+1e-9)/3.0, 0.0, 1.0) * non_zero
            inliers = torch.sum(inliers, dim=(1,2))
            non_zero_gt = torch.clamp(gt_imgs, 0, 1)
            non_zero_pred = torch.clamp(imgs, 0, 1)
            non_zero_both = torch.mul(non_zero_gt,non_zero_pred) # Non zero in both gt and predict
            total_gt = torch.sum(non_zero_both, dim=(1,2))
            batch_loss = (inliers/total_gt)

            batch_loss = batch_loss*confs[:,i] + (1.0/num_views)*batch_loss
            losses.append(batch_loss.unsqueeze(-1))

            # Calculate pose loss
            for k,p in enumerate(prev_poses):
                R = torch.matmul(p, torch.transpose(Rs_predicted, 1, 2))
                R_trace = torch.diagonal(R, dim1=-2, dim2=-1).sum(-1)
                theta = (R_trace - 1.0)/2.0
                epsilon=1e-5
                theta = torch.acos(torch.clamp(theta, -1 + epsilon, 1 - epsilon))
                degree = theta * (180.0/3.14159)

                pose_max = loss_params
                pose_diff = 1.0 - (torch.clamp(degree, 0.0, pose_max)/pose_max)

                pose_batch_loss = pose_diff
                pose_losses.append(pose_batch_loss.unsqueeze(-1))

            # Add current predicted poses to list of previous ones
            prev_poses.append(Rs_predicted)


        # Concat different views
        gt_imgs = torch.cat(gt_images, dim=1)
        predicted_imgs = torch.cat(predicted_images, dim=1)
        losses = torch.cat(losses, dim=1)
        pose_losses = torch.cat(pose_losses, dim=1)

        #print("depth loss ", torch.mean(losses, dim=1))
        #print("pose loss ", torch.mean(pose_losses, dim=1))

        batch_loss = torch.mean(losses, dim=1)
        batch_loss = batch_loss + batch_loss * (torch.mean(pose_losses, dim=1)-0.5)/2.0
        #print("VSD: ", torch.mean(losses, dim=1))
        #print("pose: ", torch.mean(pose_losses, dim=1))
        batch_loss = batch_loss.unsqueeze(-1)
        loss = torch.mean(batch_loss) #losses) #+torch.mean(pose_losses)
        return loss, batch_loss, gt_imgs, predicted_imgs

    elif(loss_method=="vsd-paper"):
        num_views = len(views)
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
            curr_pose = predicted_poses[:,pose_start:pose_end]
            Rs_predicted = compute_rotation_matrix_from_ortho6d(curr_pose)
            pose_start = pose_end
            pose_end = pose_start + 6

            # Render predicted images
            imgs = renderer.renderBatch(Rs_predicted, ts)
            predicted_images.append(imgs)
            gt_images.append(gt_imgs)

            # Calculate visiblity masks
            v_gt = gt_imgs != 0
            v_pd = imgs != 0
            v_intersection = v_pd * v_gt
            v_union = torch.zeros_like(gt_imgs)
            v_union[v_gt] = 1.0
            v_union[v_pd] = 1.0

            # Calculate loss
            #log_diff = torch.log(torch.abs(gt_imgs - imgs)+1e-9)
            diff = torch.clamp(torch.abs(gt_imgs - imgs), 0.0, 100.0)
            batch_loss = torch.sum(diff*v_union, dim=(1,2))/torch.sum(v_intersection, dim=(1,2))

            batch_loss = batch_loss*confs[:,i] + (1.0/num_views)*batch_loss
            losses.append(batch_loss.unsqueeze(-1))

            # Calculate pose loss
            for k,p in enumerate(prev_poses):
                R = torch.matmul(p, torch.transpose(Rs_predicted, 1, 2))
                R_trace = torch.diagonal(R, dim1=-2, dim2=-1).sum(-1)
                theta = (R_trace - 1.0)/2.0
                epsilon=1e-5
                theta = torch.acos(torch.clamp(theta, -1 + epsilon, 1 - epsilon))
                degree = theta * (180.0/3.14159)

                pose_max = loss_params
                pose_diff = 1.0 - (torch.clamp(degree, 0.0, pose_max)/pose_max)

                pose_batch_loss = pose_diff
                pose_losses.append(pose_batch_loss.unsqueeze(-1))

            # Add current predicted poses to list of previous ones
            prev_poses.append(Rs_predicted)


        # Concat different views
        gt_imgs = torch.cat(gt_images, dim=1)
        predicted_imgs = torch.cat(predicted_images, dim=1)
        losses = torch.cat(losses, dim=1)
        pose_losses = torch.cat(pose_losses, dim=1)

        #print("depth loss ", torch.mean(losses, dim=1))
        #print("pose loss ", torch.mean(pose_losses, dim=1))

        batch_loss = torch.sum(losses, dim=1)
        batch_loss = batch_loss + batch_loss * (torch.mean(pose_losses, dim=1)-0.5)/2.0
        #print("VSD: ", torch.mean(losses, dim=1))
        #print("pose: ", torch.mean(pose_losses, dim=1))
        batch_loss = batch_loss.unsqueeze(-1)
        loss = torch.mean(batch_loss) #losses) #+torch.mean(pose_losses)
        return loss, batch_loss, gt_imgs, predicted_imgs

    elif(loss_method=="vsd-paper-max20"):
        num_views = len(views)
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
            curr_pose = predicted_poses[:,pose_start:pose_end]
            Rs_predicted = compute_rotation_matrix_from_ortho6d(curr_pose)
            pose_start = pose_end
            pose_end = pose_start + 6

            # Render predicted images
            imgs = renderer.renderBatch(Rs_predicted, ts)
            predicted_images.append(imgs)
            gt_images.append(gt_imgs)

            # Calculate visiblity masks
            v_gt = gt_imgs != 0
            v_pd = imgs != 0
            v_intersection = v_pd * v_gt
            v_union = torch.zeros_like(gt_imgs)
            v_union[v_gt] = 1.0
            v_union[v_pd] = 1.0

            # Calculate loss
            #log_diff = torch.log(torch.abs(gt_imgs - imgs)+1e-9)
            diff = torch.clamp(torch.abs(gt_imgs - imgs), 0.0, 20.0)
            batch_loss = torch.sum(diff*v_union, dim=(1,2))/torch.sum(v_intersection, dim=(1,2))

            batch_loss = batch_loss*confs[:,i] + (1.0/num_views)*batch_loss
            losses.append(batch_loss.unsqueeze(-1))

            # Calculate pose loss
            for k,p in enumerate(prev_poses):
                R = torch.matmul(p, torch.transpose(Rs_predicted, 1, 2))
                R_trace = torch.diagonal(R, dim1=-2, dim2=-1).sum(-1)
                theta = (R_trace - 1.0)/2.0
                epsilon=1e-5
                theta = torch.acos(torch.clamp(theta, -1 + epsilon, 1 - epsilon))
                degree = theta * (180.0/3.14159)

                pose_max = loss_params
                pose_diff = 1.0 - (torch.clamp(degree, 0.0, pose_max)/pose_max)

                pose_batch_loss = pose_diff
                pose_losses.append(pose_batch_loss.unsqueeze(-1))

            # Add current predicted poses to list of previous ones
            prev_poses.append(Rs_predicted)


        # Concat different views
        gt_imgs = torch.cat(gt_images, dim=1)
        predicted_imgs = torch.cat(predicted_images, dim=1)
        losses = torch.cat(losses, dim=1)
        pose_losses = torch.cat(pose_losses, dim=1)

        #print("depth loss ", torch.mean(losses, dim=1))
        #print("pose loss ", torch.mean(pose_losses, dim=1))

        batch_loss = torch.sum(losses, dim=1)
        batch_loss = batch_loss + batch_loss * (torch.mean(pose_losses, dim=1)-0.5)/2.0
        #print("VSD: ", torch.mean(losses, dim=1))
        #print("pose: ", torch.mean(pose_losses, dim=1))
        batch_loss = batch_loss.unsqueeze(-1)
        loss = torch.mean(batch_loss) #losses) #+torch.mean(pose_losses)
        return loss, batch_loss, gt_imgs, predicted_imgs

    elif(loss_method=="vsd-paper-log"):
        num_views = len(views)
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
            curr_pose = predicted_poses[:,pose_start:pose_end]
            Rs_predicted = compute_rotation_matrix_from_ortho6d(curr_pose)
            pose_start = pose_end
            pose_end = pose_start + 6

            # Render predicted images
            imgs = renderer.renderBatch(Rs_predicted, ts)
            predicted_images.append(imgs)
            gt_images.append(gt_imgs)

            # Calculate visiblity masks
            v_gt = gt_imgs != 0
            v_pd = imgs != 0
            v_intersection = v_pd * v_gt
            v_union = torch.zeros_like(gt_imgs)
            v_union[v_gt] = 1.0
            v_union[v_pd] = 1.0

            # Calculate loss
            diff = torch.log10(torch.abs(gt_imgs - imgs)+1)
            #diff = torch.clamp(torch.abs(gt_imgs - imgs), 0.0, 100.0)
            batch_loss = torch.sum(diff*v_union, dim=(1,2))/torch.sum(v_intersection, dim=(1,2))

            batch_loss = batch_loss*confs[:,i] + (1.0/num_views)*batch_loss
            losses.append(batch_loss.unsqueeze(-1))

            # Calculate pose loss
            for k,p in enumerate(prev_poses):
                R = torch.matmul(p, torch.transpose(Rs_predicted, 1, 2))
                R_trace = torch.diagonal(R, dim1=-2, dim2=-1).sum(-1)
                theta = (R_trace - 1.0)/2.0
                epsilon=1e-5
                theta = torch.acos(torch.clamp(theta, -1 + epsilon, 1 - epsilon))
                degree = theta * (180.0/3.14159)

                pose_max = loss_params
                pose_diff = 1.0 - (torch.clamp(degree, 0.0, pose_max)/pose_max)

                pose_batch_loss = pose_diff
                pose_losses.append(pose_batch_loss.unsqueeze(-1))

            # Add current predicted poses to list of previous ones
            prev_poses.append(Rs_predicted)


        # Concat different views
        gt_imgs = torch.cat(gt_images, dim=1)
        predicted_imgs = torch.cat(predicted_images, dim=1)
        losses = torch.cat(losses, dim=1)
        pose_losses = torch.cat(pose_losses, dim=1)

        #print("depth loss ", torch.mean(losses, dim=1))
        #print("pose loss ", torch.mean(pose_losses, dim=1))

        batch_loss = torch.sum(losses, dim=1)
        batch_loss = batch_loss + batch_loss * (torch.mean(pose_losses, dim=1)-0.5)/2.0
        #print("VSD: ", torch.mean(losses, dim=1))
        #print("pose: ", torch.mean(pose_losses, dim=1))
        batch_loss = batch_loss.unsqueeze(-1)
        loss = torch.mean(batch_loss) #losses) #+torch.mean(pose_losses)
        return loss, batch_loss, gt_imgs, predicted_imgs




    elif(loss_method=="vsd-intersection-max20"):
        num_views = len(views)
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
            curr_pose = predicted_poses[:,pose_start:pose_end]
            Rs_predicted = compute_rotation_matrix_from_ortho6d(curr_pose)
            pose_start = pose_end
            pose_end = pose_start + 6

            # Render predicted images
            imgs = renderer.renderBatch(Rs_predicted, ts)
            predicted_images.append(imgs)
            gt_images.append(gt_imgs)

            # Calculate VSD depth loss
            diff = torch.abs(gt_imgs - imgs)
            diff = torch.clamp(diff, 0, 20.0)

            # Visiblity mask
            mask_gt = gt_imgs != 0
            mask_pd = imgs != 0
            mask_intersection = mask_pd * mask_gt
            mask_union = torch.zeros_like(gt_imgs)
            mask_union[mask_gt] = 1.0
            mask_union[mask_pd] = 1.0
            batch_loss = torch.sum(diff*mask_intersection, dim=(1,2))

            # #non_zero = torch.clamp(gt_imgs, 0, 1) # Non zero in gt only
            # non_zero = torch.clamp(gt_imgs + imgs, 0, 1) # Non zero in either gt or prediction
            # inliers = torch.clamp(torch.log10(diff+1e-9)/3.0, 0.0, 1.0) * non_zero
            # inliers = torch.sum(inliers, dim=(1,2))
            # non_zero_gt = torch.clamp(gt_imgs, 0, 1)
            # non_zero_pred = torch.clamp(imgs, 0, 1)
            # non_zero_both = torch.mul(non_zero_gt,non_zero_pred) # Non zero in both gt and predict
            # total_gt = torch.sum(non_zero_both, dim=(1,2))
            # batch_loss = (inliers/total_gt)

            batch_loss = batch_loss*confs[:,i] + (1.0/num_views)*batch_loss
            losses.append(batch_loss.unsqueeze(-1))

            # Calculate pose loss
            for k,p in enumerate(prev_poses):
                R = torch.matmul(p, torch.transpose(Rs_predicted, 1, 2))
                R_trace = torch.diagonal(R, dim1=-2, dim2=-1).sum(-1)
                theta = (R_trace - 1.0)/2.0
                epsilon=1e-5
                theta = torch.acos(torch.clamp(theta, -1 + epsilon, 1 - epsilon))
                degree = theta * (180.0/3.14159)

                pose_max = loss_params
                pose_diff = 1.0 - (torch.clamp(degree, 0.0, pose_max)/pose_max)

                pose_batch_loss = pose_diff
                pose_losses.append(pose_batch_loss.unsqueeze(-1))

            # Add current predicted poses to list of previous ones
            prev_poses.append(Rs_predicted)


        # Concat different views
        gt_imgs = torch.cat(gt_images, dim=1)
        predicted_imgs = torch.cat(predicted_images, dim=1)
        losses = torch.cat(losses, dim=1)
        pose_losses = torch.cat(pose_losses, dim=1)

        #print("depth loss ", torch.mean(losses, dim=1))
        #print("pose loss ", torch.mean(pose_losses, dim=1))

        batch_loss = torch.mean(losses, dim=1)
        batch_loss = batch_loss + batch_loss * (torch.mean(pose_losses, dim=1)-0.5)/2.0
        #print("VSD: ", torch.mean(losses, dim=1))
        #print("pose: ", torch.mean(pose_losses, dim=1))
        batch_loss = batch_loss.unsqueeze(-1)
        loss = torch.mean(batch_loss) #losses) #+torch.mean(pose_losses)
        return loss, batch_loss, gt_imgs, predicted_imgs

    elif(loss_method=="vsd-union-max20"):
        num_views = len(views)
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
            curr_pose = predicted_poses[:,pose_start:pose_end]
            Rs_predicted = compute_rotation_matrix_from_ortho6d(curr_pose)
            pose_start = pose_end
            pose_end = pose_start + 6

            # Render predicted images
            imgs = renderer.renderBatch(Rs_predicted, ts)
            predicted_images.append(imgs)
            gt_images.append(gt_imgs)

            # Calculate VSD depth loss
            diff = torch.abs(gt_imgs - imgs)
            diff = torch.clamp(diff, 0, 20.0)

            # Visiblity mask
            mask_gt = gt_imgs != 0
            mask_pd = imgs != 0
            mask_union = torch.zeros_like(gt_imgs)
            mask_union[mask_gt] = 1.0
            mask_union[mask_pd] = 1.0
            batch_loss = torch.sum(diff*mask_union, dim=(1,2))/torch.sum(mask_union, dim=(1,2))

            batch_loss = batch_loss*confs[:,i] + (1.0/num_views)*batch_loss
            losses.append(batch_loss.unsqueeze(-1))

            # Calculate pose loss
            for k,p in enumerate(prev_poses):
                R = torch.matmul(p, torch.transpose(Rs_predicted, 1, 2))
                R_trace = torch.diagonal(R, dim1=-2, dim2=-1).sum(-1)
                theta = (R_trace - 1.0)/2.0
                epsilon=1e-5
                theta = torch.acos(torch.clamp(theta, -1 + epsilon, 1 - epsilon))
                degree = theta * (180.0/3.14159)

                pose_max = loss_params
                pose_diff = 1.0 - (torch.clamp(degree, 0.0, pose_max)/pose_max)

                pose_batch_loss = pose_diff
                pose_losses.append(pose_batch_loss.unsqueeze(-1))

            # Add current predicted poses to list of previous ones
            prev_poses.append(Rs_predicted)


        # Concat different views
        gt_imgs = torch.cat(gt_images, dim=1)
        predicted_imgs = torch.cat(predicted_images, dim=1)
        losses = torch.cat(losses, dim=1)
        pose_losses = torch.cat(pose_losses, dim=1)

        #print("depth loss ", torch.mean(losses, dim=1))
        #print("pose loss ", torch.mean(pose_losses, dim=1))

        batch_loss = torch.mean(losses, dim=1)
        batch_loss = batch_loss + batch_loss * (torch.mean(pose_losses, dim=1)-0.5)/2.0
        #print("VSD: ", torch.mean(losses, dim=1))
        #print("pose: ", torch.mean(pose_losses, dim=1))
        batch_loss = batch_loss.unsqueeze(-1)
        loss = torch.mean(batch_loss) #losses) #+torch.mean(pose_losses)
        return loss, batch_loss, gt_imgs, predicted_imgs



    elif(loss_method=="depth-clamped-predicted-view-sil"):
        num_views = len(views)
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
            curr_pose = predicted_poses[:,pose_start:pose_end]
            Rs_predicted = compute_rotation_matrix_from_ortho6d(curr_pose)
            pose_start = pose_end
            pose_end = pose_start + 6

            # Render predicted images
            imgs = renderer.renderBatch(Rs_predicted, ts)
            predicted_images.append(imgs)
            gt_images.append(gt_imgs)

            # Calculate depth loss
            #diff = torch.abs(gt_imgs - imgs).flatten(start_dim=1)
            #diff = torch.clamp(diff, 0.0, loss_params)/loss_params
            #batch_loss = torch.mean(diff, dim=1)

            # Jaccard index
            overlap = torch.sum(gt_imgs * imgs, dim=(1,2))
            union = torch.sum(gt_imgs, dim=(1,2)) + torch.sum(imgs, dim=(1,2)) - overlap
            diff = 1.0 - overlap / (union + 1e-5)
            batch_loss = diff

            #batch_loss = (1.0/num_views)*batch_loss #+ batch_loss*confs[:,i]
            losses.append(batch_loss.unsqueeze(-1))

            # Calculate pose loss
            for k,p in enumerate(prev_poses):
                mseLoss = nn.MSELoss(reduction='non e')
                pose_diff = torch.abs(p - Rs_predicted).flatten(start_dim=1)
                pose_max = 0.25
                pose_diff = 1.0 - (torch.clamp(pose_diff, 0.0, pose_max)/pose_max)
                pose_batch_loss = torch.mean(pose_diff, dim=1)
                pose_losses.append(pose_batch_loss.unsqueeze(-1))

            # Add current predicted poses to list of previous ones
            prev_poses.append(Rs_predicted)


        # Concat different views
        gt_imgs = torch.cat(gt_images, dim=1)
        predicted_imgs = torch.cat(predicted_images, dim=1)
        losses = torch.cat(losses, dim=1)
        pose_losses = torch.cat(pose_losses, dim=1)**3

        batch_loss = torch.mean(losses, dim=1) + torch.mean(pose_losses, dim=1)
        batch_loss = batch_loss.unsqueeze(-1)
        loss = torch.mean(losses)+torch.mean(pose_losses)
        return loss, batch_loss, gt_imgs, predicted_imgs


    elif(loss_method=="depth-clamped-predicted-view-conf"):
        num_views = len(views)
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
            curr_pose = predicted_poses[:,pose_start:pose_end]
            Rs_predicted = compute_rotation_matrix_from_ortho6d(curr_pose)
            pose_start = pose_end
            pose_end = pose_start + 6

            # Render predicted images
            imgs = renderer.renderBatch(Rs_predicted, ts)
            predicted_images.append(imgs)
            gt_images.append(gt_imgs)

            # Calculate depth loss
            diff = torch.abs(gt_imgs - imgs).flatten(start_dim=1)
            diff = torch.clamp(diff, 0.0, loss_params)/loss_params

            batch_loss = torch.mean(diff, dim=1)
            #print("weighted: ", torch.mean(batch_loss*confs[:,i]))
            #print("averaged: ", torch.mean((1.0/num_views)*batch_loss))
            #batch_loss = batch_loss*confs[:,i] + (1.0/num_views)*batch_loss
            batch_loss = (1.0/num_views)*batch_loss
            losses.append(batch_loss.unsqueeze(-1))

            # Calculate pose loss
            for k,p in enumerate(prev_poses):
                mseLoss = nn.MSELoss(reduction='none')
                pose_diff = torch.abs(p - Rs_predicted).flatten(start_dim=1)
                pose_max = 0.25
                pose_diff = 1.0 - (torch.clamp(pose_diff, 0.0, pose_max)/pose_max)
                pose_batch_loss = torch.mean(pose_diff, dim=1)
                pose_losses.append(pose_batch_loss.unsqueeze(-1))

            # Add current predicted poses to list of previous ones
            prev_poses.append(Rs_predicted)


        # Concat different views
        gt_imgs = torch.cat(gt_images, dim=1)
        predicted_imgs = torch.cat(predicted_images, dim=1)
        losses = torch.cat(losses, dim=1)
        pose_losses = torch.cat(pose_losses, dim=1)**3

        #print("depth loss ", torch.mean(losses, dim=1))
        #print("pose loss ", torch.mean(pose_losses, dim=1))

        # Conf loss
        conf_loss = F.softmax(losses, dim=1)
        conf_loss = torch.abs(confs - conf_loss)**3
        losses = losses + conf_loss

        batch_loss = torch.mean(losses, dim=1) + torch.mean(pose_losses, dim=1)
        batch_loss = batch_loss.unsqueeze(-1)
        loss = torch.mean(losses)+torch.mean(pose_losses)
        return loss, batch_loss, gt_imgs, predicted_imgs


    elif(loss_method=="depth-clamped-predicted-view-centered"):
        num_views = len(views)
        pose_start = num_views
        pose_end = pose_start + 6

        # Prepare gt images
        gt_images = []
        predicted_images = []
        gt_imgs = renderer.renderBatch(Rs_gt, ts)

        # Zero center the images
        mask = gt_imgs > 0
        gt_imgs = gt_imgs - mask*ts[0][-1]

        losses = []
        confs = predicted_poses[:,:num_views]
        prev_poses = []
        pose_losses = []
        for i,v in enumerate(views):
            # Extract current pose and move to next one
            curr_pose = predicted_poses[:,pose_start:pose_end]
            Rs_predicted = compute_rotation_matrix_from_ortho6d(curr_pose)
            pose_start = pose_end
            pose_end = pose_start + 6

            # Render predicted images
            imgs = renderer.renderBatch(Rs_predicted, ts)

            # Zero center the images
            mask = imgs > 0
            imgs = imgs - mask*ts[0][-1]

            predicted_images.append(imgs)
            gt_images.append(gt_imgs)

            # Calculate depth loss
            diff = torch.abs(gt_imgs - imgs).flatten(start_dim=1)**2
            #diff = torch.clamp(diff, 0.0, loss_params)/loss_params
            batch_loss = torch.mean(diff, dim=1)
            #print("weighted: ", torch.mean(batch_loss*confs[:,i]))
            #print("averaged: ", torch.mean((1.0/num_views)*batch_loss))
            batch_loss = batch_loss*confs[:,i] + (1.0/num_views)*batch_loss
            losses.append(batch_loss.unsqueeze(-1))

            # Calculate pose loss
            for k,p in enumerate(prev_poses):
                mseLoss = nn.MSELoss(reduction='none')
                pose_diff = torch.abs(p - Rs_predicted).flatten(start_dim=1)
                pose_max = 0.25
                pose_diff = 1.0 - (torch.clamp(pose_diff, 0.0, pose_max)/pose_max)
                pose_batch_loss = torch.mean(pose_diff, dim=1)
                pose_losses.append(pose_batch_loss.unsqueeze(-1))

            # Add current predicted poses to list of previous ones
            prev_poses.append(Rs_predicted)


        # Concat different views
        gt_imgs = torch.cat(gt_images, dim=1)
        predicted_imgs = torch.cat(predicted_images, dim=1)
        losses = torch.cat(losses, dim=1)/1500.0
        pose_losses = torch.cat(pose_losses, dim=1)**3

        #print("depth loss ", torch.mean(losses, dim=1))
        #print("pose loss ", torch.mean(pose_losses, dim=1))

        batch_loss = torch.mean(losses, dim=1) + torch.mean(pose_losses, dim=1)
        batch_loss = batch_loss.unsqueeze(-1)
        loss = torch.mean(losses)+torch.mean(pose_losses)
        return loss, batch_loss, gt_imgs, predicted_imgs


    elif(loss_method=="chamfer2"):
        num_views = 4
        pose_start = num_views
        pose_end = pose_start + 6

        #Prepare gt point cloud
        gt_t = Rotate(Rs_gt).to(renderer.device)
        gt_points = gt_t.transform_points(renderer.points)

        # Prepare gt images
        gt_images = []
        predicted_images = []
        gt_imgs = renderer.renderBatch(Rs_gt, ts)

        losses = []
        predictions = []
        confs = predicted_poses[:,:num_views]
        prev_pose = None
        for i in np.arange(num_views):
            # Extract current pose and move to next one
            curr_pose = predicted_poses[:,pose_start:pose_end]
            Rs_predicted = compute_rotation_matrix_from_ortho6d(curr_pose)
            pose_start = pose_end
            pose_end = pose_start + 6

            # Apply predicted pose to point cloud
            predicted_t = Rotate(Rs_predicted).to(renderer.device)
            predicted_points = predicted_t.transform_points(renderer.points)
            predictions.append(Rs_predicted)

            # Calculate loss
            batch_loss,_ = chamfer_bootstrap(gt_points, predicted_points,
                                             bootstrap_ratio=int(loss_params),
                                             batch_reduction=None)
            batch_loss = batch_loss*confs[:,i]
            losses.append(batch_loss.unsqueeze(-1))

            # Render images
            imgs = renderer.renderBatch(Rs_predicted, ts)
            predicted_images.append(imgs)
            gt_images.append(gt_imgs)

            prev_pose = Rs_predicted

        # Repel similar poses
        #preds = torch.cat(predictions, dim=1)
        preds = torch.stack(predictions).permute(1,0,2,3)
        unit_v = torch.tensor([1.0, 1.0, 1.0])/3.0
        pred_v = torch.matmul(preds, unit_v.to(renderer.device))

        pose_diffs = []
        for i in np.arange(num_views-1):
            current = pred_v[:,i,:].unsqueeze(1)
            others = pred_v[:,i+1:,:]
            diff = torch.sum(torch.sum(others*current, dim=2), dim=1)
            diff = (diff + 1.0)/2.0
            pose_diffs.append(diff)

        # Concat different views
        pose_loss = torch.stack(pose_diffs).permute(1,0)
        gt_imgs = torch.cat(gt_images, dim=1)
        predicted_imgs = torch.cat(predicted_images, dim=1)
        losses = torch.cat(losses, dim=1)

        losses = torch.mean(losses, dim=1)
        pose_loss = torch.mean(pose_loss, dim=1)

        batch_loss = (pose_loss + 100.0*losses) #torch.mean(losses, dim=1)
        loss = torch.mean(batch_loss)
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
