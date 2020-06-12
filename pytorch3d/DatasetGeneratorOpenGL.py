import os
import torch
import numpy as np
import imageio
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from skimage import img_as_ubyte
import argparse
import glob
import hashlib
import cv2 as cv

import time
import pickle
import random
from utils.utils import *
from Encoder import Encoder

import imgaug as ia
import imgaug.augmenters as iaa

from scipy.spatial.transform import Rotation as scipyR
from plot_loss_landscape import eqv_dist_points

# io utils
from pytorch3d.io import load_obj

# datastructures
from pytorch3d.structures import Meshes, Textures, list_to_padded

# 3D transformations functions
from pytorch3d.transforms import Rotate, Translate

# rendering components
from pytorch3d.renderer import (
    OpenGLPerspectiveCameras, look_at_view_transform, look_at_rotation,
    RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
    HardPhongShader, PointLights, DirectionalLights
)

from utils.pytless import inout, renderer, misc

class DatasetGenerator():

    def __init__(self, background_path, obj_path, obj_distance, batch_size,
                 encoder_weights, device, sampling_method="sphere"):
        self.device = device
        self.obj_path = obj_path
        self.batch_size = batch_size
        self.dist = obj_distance
        self.img_size = 320
        self.K = np.array([1075.65, 0, self.img_size/2,
                           0, 1073.90, self.img_size/2,
                           0, 0, 1]).reshape(3,3)
        self.aug = self.setup_augmentation()
        self.model = inout.load_ply(obj_path.replace(".obj",".ply"))
        self.backgrounds = self.load_bg_images("backgrounds", background_path, 170,
                                               self.img_size, self.img_size)
        self.encoder = self.load_encoder(encoder_weights)
        #self.view_points = eqv_dist_points(100000)

        self.simple_pose_sampling = False
        if(sampling_method == "tless"):
            self.pose_sampling = self.tless_sampling
            self.simple_pose_sampling = False
        elif(sampling_method == "sphere"):
            self.pose_sampling = self.sphere_sampling
            self.simple_pose_sampling = False
        elif(sampling_method == "tless-simple"):
            self.pose_sampling = self.tless_sampling
            self.simple_pose_sampling = True
        elif(sampling_method == "sphere-simple"):
            self.pose_sampling = self.sphere_sampling
            self.simple_pose_sampling = True
        else:
            print("ERROR! Invalid view sampling method: {0}".format(sampling_method))

    def load_encoder(self, weights_path):
        model = Encoder(weights_path).to(self.device)
        model.eval()
        return model

    def load_bg_images(self, output_path, background_path, num_bg_images, h, w, c=3):
        if(background_path == ""):
            return []
    
        bg_img_paths = glob.glob(background_path + "*.jpg")
        noof_bg_imgs = min(num_bg_images, len(bg_img_paths))
        shape = (h, w, c)
        bg_imgs = np.empty( (noof_bg_imgs,) + shape, dtype=np.uint8 )

        current_config_hash = hashlib.md5((str(shape) + str(noof_bg_imgs) + str(background_path)).encode('utf-8')).hexdigest()
        current_file_name = os.path.join(output_path + '-' + current_config_hash +'.npy')
        if os.path.exists(current_file_name):
            bg_imgs = np.load(current_file_name)
        else:
            file_list = bg_img_paths[:noof_bg_imgs]
            print(len(file_list))
            from random import shuffle
            shuffle(file_list)
            
            for j,fname in enumerate(file_list):
                print('loading bg img %s/%s' % (j,noof_bg_imgs))
                bgr = cv2.imread(fname)
                H,W = bgr.shape[:2]
                y_anchor = int(np.random.rand() * (H-shape[0]))
                x_anchor = int(np.random.rand() * (W-shape[1]))
                # bgr = cv2.resize(bgr, shape[:2])
                bgr = bgr[y_anchor:y_anchor+shape[0],x_anchor:x_anchor+shape[1],:]
                if bgr.shape[0]!=shape[0] or bgr.shape[1]!=shape[1]:
                    continue
                if shape[2] == 1:
                    bgr = cv2.cvtColor(np.uint8(bgr), cv2.COLOR_BGR2GRAY)[:,:,np.newaxis]
                bg_imgs[j] = bgr
            np.save(current_file_name,bg_imgs)
            print('loaded %s bg images' % noof_bg_imgs)
        return bg_imgs

    def setup_augmentation(self):
        # Augmentation
        aug = iaa.Sequential([
            #iaa.Sometimes(0.5, iaa.PerspectiveTransform(0.05)),
            #iaa.Sometimes(0.5, iaa.CropAndPad(percent=(-0.05, 0.1))),
            #iaa.Sometimes(0.5, iaa.Affine(scale=(1.0, 1.2))),
            iaa.Sometimes(0.5, iaa.CoarseDropout( p=0.05, size_percent=0.01) ),
            iaa.Sometimes(0.5, iaa.GaussianBlur(1.2*np.random.rand())),
            iaa.Sometimes(0.5, iaa.Add((-0.1, 0.1), per_channel=0.3)),
            iaa.Sometimes(0.3, iaa.Invert(0.2, per_channel=True)),
            iaa.Sometimes(0.5, iaa.Multiply((0.6, 1.4), per_channel=0.5)),
            iaa.Sometimes(0.5, iaa.Multiply((0.6, 1.4))),
            iaa.Sometimes(0.5, iaa.ContrastNormalization((0.5, 2.2), per_channel=0.3))],
                             random_order=False)
        return aug

    # Sampling based on the T-LESS dataset
    def tless_sampling(self):
        # Generate random pose for the batch
        # All images in the batch will share pose but different augmentations
        R, t = look_at_view_transform(self.dist, elev=0, azim=0, up=((0, 1, 0),))
    
        # Sample azimuth and apply transformation
        azim = np.random.uniform(low=0.0, high=360.0, size=1)
        rot = scipyR.from_euler('z', azim, degrees=True)    
        rot_mat = torch.tensor(rot.as_matrix(), dtype=torch.float32)
        R = torch.matmul(R, rot_mat)
        
        # Sample elevation and apply transformation
        elev = np.random.uniform(low=-180, high=0.0, size=1)
        rot = scipyR.from_euler('x', elev, degrees=True)    
        rot_mat = torch.tensor(rot.as_matrix(), dtype=torch.float32)
        R = torch.matmul(R, rot_mat)

        # Sample cam plane rotation and apply
        if(not self.simple_pose_sampling):
            in_plane = np.random.uniform(low=0.0, high=360.0, size=1)
            rot = scipyR.from_euler('y', in_plane, degrees=True)    
            rot_mat = torch.tensor(rot.as_matrix(), dtype=torch.float32)
            R = torch.inverse(R)
            R = torch.matmul(R, rot_mat)
            R = torch.inverse(R)
        
        t = torch.tensor([0.0, 0.0, self.dist])
        return R,t

    # Truely random
    # Based on: https://www.cmu.edu/biolphys/deserno/pdf/sphere_equi.pdf
    def sphere_sampling(self):
        z_sample = np.random.uniform(low=-self.dist, high=self.dist, size=1)[0]
        theta_sample = np.random.uniform(low=0.0, high=2.0*np.pi, size=1)[0]
        x = np.sqrt((self.dist**2 - z_sample**2))*np.cos(theta_sample)
        y = np.sqrt((self.dist**2 - z_sample**2))*np.sin(theta_sample)
        z = np.sqrt(self.dist**2 - x**2 - y**2)

        cam_position = torch.tensor([x, y, z]).unsqueeze(0)
        R = look_at_rotation(cam_position, up=((0, 0, 1),)).squeeze()

        # Rotate in-plane
        if(not self.simple_pose_sampling):
            rot_degrees = np.random.uniform(low=0.0, high=360.0, size=1)
            rot = scipyR.from_euler('z', rot_degrees, degrees=True)    
            rot_mat = torch.tensor(rot.as_matrix(), dtype=torch.float32)
            R = torch.matmul(R, rot_mat)
        
        t = torch.tensor([0.0, 0.0, self.dist])
        return R,t
    
    def generate_image_batch(self):
        # Generate random poses
        curr_Rs = []
        curr_ts = []

        image_renders = []
        for k in np.arange(self.batch_size):
            R, t = self.pose_sampling()
            R = R.detach().cpu().numpy()[0]
            t = t.detach().cpu().numpy()

            curr_Rs.append(R)
            curr_ts.append(t)

            # Convert R matrix from pytorch to opengl format
            # for rendering only!
            xy_flip = np.eye(3, dtype=np.float)
            xy_flip[0,0] = -1.0
            xy_flip[1,1] = -1.0
            R_opengl = np.dot(R,xy_flip)
            R_opgegl = np.transpose(R_opengl)
        
            # Render images
            ren_rgb = renderer.render(self.model, (self.img_size,self.img_size),
                                      self.K, R_opengl, t, surf_color=(1, 1, 1), mode='rgb')
            
            image_renders.append(ren_rgb)

        # Augment data
        image_renders = np.array(image_renders)
        images_aug = self.aug(images=image_renders)

        if(len(self.backgrounds) > 0):
            bg_im_isd = np.random.choice(len(self.backgrounds), self.batch_size, replace=False)

        images = []
        for k in np.arange(self.batch_size):
            image_base = image_renders[k]
            image_ref = images_aug[k]

            if(len(self.backgrounds) > 0):
                img_back = self.backgrounds[bg_im_isd[k]]
                img_back = cv.cvtColor(img_back, cv.COLOR_BGR2RGBA).astype(float)
                alpha = image_base[:, :, 0:3].astype(float)
                sum_img = np.sum(image_base[:,:,:3], axis=2)
                alpha[sum_img > 0] = 1
                image_ref[:, :, 0:3] = image_ref[:, :, 0:3] * alpha + img_back[:, :, 0:3]/255 * (1 - alpha)
            else:
                image_ref = image_ref[:, :, 0:3]
                

            image_ref = image_ref.astype(np.float)/255.0
            image_ref = np.clip(image_ref, 0.0, 1.0)

            org_img = image_renders[k]
            ys, xs = np.nonzero(org_img[:,:,0] > 0)
            obj_bb = calc_2d_bbox(xs,ys,[640,640])
            cropped = extract_square_patch(image_ref, obj_bb)
            cropped_org = extract_square_patch(org_img, obj_bb)
            images.append(cropped[:,:,:3])

        data = {"images":images,
                "Rs":curr_Rs}
        return data

    def generate_images(self, num_samples):
        data = {"images":[],
                "Rs":[]}
        while(len(data["images"]) < num_samples):
            print("Generating images: {0}/{1}".format(len(data["images"]), num_samples))
            curr_data = self.generate_image_batch()
            data["images"] = data["images"] + curr_data["images"]
            data["Rs"] = data["Rs"] + curr_data["Rs"]
        data["images"] = data["images"][:num_samples]
        data["Rs"] = data["Rs"][:num_samples]
        return data

    def generate_samples(self, num_samples):
        data = self.generate_images(num_samples)

        codes = []
        for img in data["images"]:
            img = torch.from_numpy(img).unsqueeze(0).permute(0,3,1,2).to(self.device)
            code = self.encoder(img.float())
            code = code.detach().cpu().numpy()
            codes.append(code[0])
        data["codes"] = codes
        return data
