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
from utils.tools import *
from Encoder import Encoder

import imgaug as ia
import imgaug.augmenters as iaa

class DatasetGenerator():

    def __init__(self, background_path, obj_path, obj_distance, batch_size,
                 encoder_weights, device, sampling_method="sphere", random_light=True,
                 num_bgs=17000):
        self.device = device
        self.poses = []
        self.obj_path = obj_path
        self.bg_path = background_path
        self.batch_size = batch_size
        self.dist = obj_distance
        self.aug = self.setup_augmentation()
        if(encoder_weights is not None):
            self.encoder = self.load_encoder(encoder_weights)
        else:
            self.encoder = None

        self.renders = np.load("./data/t-less-obj10/sundermeyer/renders.npz") #obj_path)
        self.bg_imgs = np.load("./data/t-less-obj10/sundermeyer/backgrounds.npy") #background_path)

    def load_encoder(self, weights_path):
        model = Encoder(weights_path).to(self.device)
        model.eval()
        return model
        

    def setup_augmentation(self):
        # Augmentation
        # aug = iaa.Sequential([
        #     #iaa.Sometimes(0.5, iaa.PerspectiveTransform(0.05)),
        #     #iaa.Sometimes(0.5, iaa.CropAndPad(percent=(-0.05, 0.1))),
        #     #iaa.Sometimes(0.5, iaa.Affine(scale=(1.0, 1.2))),
        #     iaa.Sometimes(0.5, iaa.CoarseDropout( p=0.05, size_percent=0.01) ),
        #     iaa.Sometimes(0.5, iaa.GaussianBlur(1.2*np.random.rand())),
        #     iaa.Sometimes(0.5, iaa.Add((-0.1, 0.1), per_channel=0.3)),
        #     iaa.Sometimes(0.3, iaa.Invert(0.2, per_channel=True)),
        #     iaa.Sometimes(0.5, iaa.Multiply((0.6, 1.4), per_channel=0.5)),
        #     iaa.Sometimes(0.5, iaa.Multiply((0.6, 1.4))),
        #     iaa.Sometimes(0.5, iaa.ContrastNormalization((0.5, 2.2), per_channel=0.3))],
        #                      random_order=False)
        # aug = iaa.Sequential([
        #     iaa.Sometimes(0.5, iaa.CoarseDropout( p=0.2, size_percent=0.05) ),
        #     iaa.Sometimes(0.5, iaa.GaussianBlur(1.2*np.random.rand())),
        #     iaa.Sometimes(0.5, iaa.Add((-25, 25), per_channel=0.3))],
        #     iaa.Sometimes(0.5, iaa.Multiply((0.6, 1.4), per_channel=0.5)),
        #     iaa.Sometimes(0.5, iaa.Multiply((0.6, 1.4))),
        #     iaa.Sometimes(0.5, iaa.ContrastNormalization((0.5, 2.2), per_channel=0.3))],
        #                      random_order=False)

        aug = iaa.Sequential([
            iaa.Sometimes(0.5, iaa.Affine(scale=(1.0, 1.2))),
            iaa.Sometimes(1.0, iaa.CoarseDropout( p=0.2, size_percent=0.05) ),
            iaa.Sometimes(0.5, iaa.GaussianBlur(1.2*np.random.rand())),
            iaa.Sometimes(0.5, iaa.Add((-25, 25), per_channel=0.3)),
            iaa.Sometimes(0.3, iaa.Invert(0.2, per_channel=True)),
            iaa.Sometimes(0.5, iaa.Multiply((0.6, 1.4), per_channel=0.5)),
            iaa.Sometimes(0.5, iaa.Multiply((0.6, 1.4))),
            iaa.Sometimes(0.5, iaa.ContrastNormalization((0.5, 2.2), per_channel=0.3))
        ], random_order=False)
        return aug


    def generate_image_batch(self, num_samples):
        # Choose random rendered image
        rand_idcs = np.random.choice(self.renders["train_x"].shape[0], num_samples, replace=False)
        images = self.renders["train_x"][rand_idcs]
        masks = self.renders["mask_x"][rand_idcs]
        Rs = self.renders["train_R"][rand_idcs]

        # Choose random background image
        rand_idcs_bg = np.random.choice(self.bg_imgs.shape[0], num_samples, replace=False)
        rand_bgs = self.bg_imgs[rand_idcs_bg]

        # Insert backgrounds
        images[masks] = rand_bgs[masks]

        # Augment images
        images = self.aug(images=images) / 255.

        for i in np.arange(Rs.shape[0]):
            tmp = Rs[i]

            xy_flip = np.eye(3, dtype=np.float)
            xy_flip[0,0] = -1.0
            xy_flip[1,1] = -1.0
            tmp = np.transpose(tmp)
            tmp = np.dot(tmp,xy_flip)
            Rs[i] = tmp
            
        data = {"images":images,
                "Rs":Rs}
        
        return data

    # def generate_images(self, num_samples):
    #     data = {"images":[],
    #             "Rs":[]}
    #     while(len(data["images"]) < num_samples):
    #         print("Generating images: {0}/{1}".format(len(data["images"]), num_samples))
    #         curr_data = self.generate_image_batch()
    #         data["images"] = data["images"] + curr_data["images"]
    #         data["Rs"] = data["Rs"] + curr_data["Rs"]
    #     data["images"] = data["images"][:num_samples]
    #     data["Rs"] = data["Rs"][:num_samples]
    #     return data

    def generate_samples(self, num_samples):
        data = self.generate_image_batch(num_samples)

        if(self.encoder is not None):
            codes = []
            for img in data["images"]:
                img = torch.from_numpy(img).unsqueeze(0).permute(0,3,1,2).to(self.device)
                code = self.encoder(img.float())
                code = code.detach().cpu().numpy()[0]
                norm_code = code / np.linalg.norm(code)
                codes.append(norm_code)
            data["codes"] = codes
        return data

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == "__main__":
    # Parse parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("obj_path", help="path to the .obj file")
    parser.add_argument("-b", help="batch size, each batch will have the same pose but different augmentations", type=int, default=8)
    parser.add_argument("-d", help="distance to the object", type=float, default=2000.0)
    parser.add_argument("-n", help="number of total samples", type=int, default=1000)
    parser.add_argument("-v", help="visualize the data", default=False)
    parser.add_argument("-o", help="output path", default="")
    parser.add_argument("-bg", help="background images path", default="")
    parser.add_argument("-s", help="pose sampling method", default="tless-simple")
    parser.add_argument("-e", help="path to .npy encoder weights", default=None)
    parser.add_argument("-rl", help="enable random light", default=True)
    parser.add_argument("-ng", help="number of backgrounds", type=int, default=17000)
    arguments = parser.parse_args()

    # Create dataset generator
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    dg = DatasetGenerator(background_path=arguments.bg,
                          obj_path=arguments.obj_path,
                          obj_distance=arguments.d,
                          batch_size=arguments.b,
                          sampling_method=arguments.s,
                          encoder_weights=arguments.e,
                          random_light=str2bool(arguments.rl),
                          num_bgs=arguments.ng,
                          device=device)

    # Generate data
    data = dg.generate_samples(num_samples=arguments.n)

    # Visualize it (optional)
    if(str2bool(arguments.v)):
        for i,img in enumerate(data["images"]):
            window_name = "Sample {0}/{1}".format(i,arguments.n)
            cv2.namedWindow(window_name)
            cv2.moveWindow(window_name,42,42)
            # Flip last axis to convert from RGB to BGR before showing using cv2
            cv2.imshow(window_name, np.flip(img,axis=2))
            key = cv2.waitKey(0)
            cv2.destroyWindow(window_name)
            if(key == ord("q")):
                break

    # Save generated data
    output_path = arguments.o
    if(output_path == ""):
        output_path = "./training-images.p"
    pickle.dump(data, open(output_path, "wb"), protocol=2)
    print("Saved dataset to: ", output_path)
