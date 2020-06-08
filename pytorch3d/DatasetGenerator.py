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

class DatasetGenerator():

    def __init__(self, background_path, obj_path, obj_distance, batch_size,
                 encoder_weights, device, sampling_method="sphere"):
        self.device = device
        self.obj_path = obj_path
        self.batch_size = batch_size
        self.dist = obj_distance
        self.img_size = 320
        self.renderer, self.batch_mesh = self.prepare_renderer(self.obj_path, self.batch_size)
        self.aug = self.setup_augmentation()
        self.backgrounds = self.load_bg_images("backgrounds", background_path, 17000,
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
        
    def prepare_renderer(self, obj_path, batch_size):
        # Load the obj and ignore the textures and materials.
        verts, faces_idx, _ = load_obj(obj_path)
        faces = faces_idx.verts_idx

        verts_rgb = torch.ones_like(verts)
        batch_verts_rgb = list_to_padded([verts_rgb for k in np.arange(batch_size)])  # B, Vmax, 3

        batch_textures = Textures(verts_rgb=batch_verts_rgb.to(self.device))
        batch_mesh = Meshes(
            verts=[verts.to(self.device) for k in np.arange(batch_size)],
            faces=[faces.to(self.device) for k in np.arange(batch_size)],
            textures=batch_textures
        )

        # Initialize an OpenGL perspective camera.
        cameras = OpenGLPerspectiveCameras(
            fov=5.0,
            device=self.device)

        # We will also create a phong renderer. This is simpler and only needs to render one face per pixel.
        raster_settings = RasterizationSettings(
            image_size=self.img_size,
            blur_radius=0.0,
            faces_per_pixel=1,
            bin_size=0
        )
        # We can add a point light in front of the object.
        #lights = PointLights(device=device, location=((-1.0, -1.0, -2.0),))
        #"ambient_color", "diffuse_color", "specular_color"
        # 'ambient':0.4,'diffuse':0.8, 'specular':0.3
        lights = DirectionalLights(device=self.device,
                                   ambient_color=[[0.25, 0.25, 0.25]],
                                   diffuse_color=[[0.6, 0.6, 0.6]],
                                   specular_color=[[0.15, 0.15, 0.15]],
                                   direction=[[-1.0, -1.0, 1.0]])
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras,
                raster_settings=raster_settings
            ),
            shader=HardPhongShader(device=self.device, lights=lights)
        )
        return renderer, batch_mesh

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
            #rot_degrees = np.random.uniform(low=0.0, high=360.0, size=1)
            rot_degrees = np.arange(0, 360, 10)
            rot_degrees = rot_degrees[np.random.uniform(low=0,high=rot_degrees.shape[0]+1)]
            rot = scipyR.from_euler('z', rot_degrees, degrees=True)    
            rot_mat = torch.tensor(rot.as_matrix(), dtype=torch.float32)
            R = torch.matmul(R, rot_mat)
        
        t = torch.tensor([0.0, 0.0, self.dist])
        return R,t
    
    def generate_image_batch(self):
        # Generate random poses
        curr_Rs = []
        curr_ts = []
        for k in np.arange(self.batch_size):
            R, t = self.pose_sampling()
            curr_Rs.append(R.squeeze())
            curr_ts.append(t.squeeze())

        batch_R = torch.tensor(np.stack(curr_Rs), device=self.device, dtype=torch.float32)
        batch_T = torch.tensor(np.stack(curr_ts), device=self.device, dtype=torch.float32)

        # Generate random lighting conditions
        random_lights = []
        for l in np.arange(self.batch_size):
            random_light = [random.uniform(-1.0,1.0) for i in np.arange(3)]
            random_lights.append(random_light)
        batch_light = torch.tensor(np.stack(random_lights), device=self.device, dtype=torch.float32)
        self.renderer.shader.lights.direction = batch_light
        
        # Render images
        image_renders = self.renderer(meshes_world=self.batch_mesh, R=batch_R, T=batch_T)

        # Augment data
        image_renders = image_renders.cpu().numpy()
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
                
            image_ref = np.clip(image_ref, 0.0, 1.0) #[:, :, 0:3]

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
