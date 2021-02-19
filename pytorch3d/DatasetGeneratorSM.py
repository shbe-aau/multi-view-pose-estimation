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
from scipy.spatial.transform import Rotation as scipyR
from utils.utils import *

import configparser
from dataset import Dataset

def build_dataset(dataset_path, args):
    dataset_args = { k:v for k,v in
        args.items('Dataset') +
        args.items('Paths') +
        args.items('Augmentation')+
        args.items('Queue') +
        args.items('Embedding')}
    dataset = Dataset(dataset_path, **dataset_args)
    return dataset

class DatasetGenerator():

    def __init__(self, background_path, obj_path, obj_distance, batch_size,
                 _, device, sampling_method="sphere", random_light=True,
                 num_bgs=15000):
        self.curr_samples = 0
        self.max_samples = 1000
        self.obj_path = obj_path
        self.batch_size = batch_size

        args = configparser.ConfigParser()
        args.read("test.cfg")

        if('reconst' in obj_path):
            args.set('Dataset', 'MODEL', 'reconst')
        elif('cad' in obj_path):
            args.set('Dataset', 'MODEL', 'cad')
        else:
            print("Can determine render type (reconst/cad) from model path: ", obj_path)
        
        args.set('Paths', 'MODEL_PATH', obj_path)
        args.set('Dataset', 'NOOF_TRAINING_IMGS', str(int(batch_size)))
        args.set('Dataset', 'NOOF_BG_IMGS', str(int(num_bgs)))

        self.dataset = build_dataset("./", args)
        self.dataset.load_bg_images("./")


    # Truely random
    # Based on: https://mathworld.wolfram.com/SpherePointPicking.html
    def sphere_wolfram_sampling_fixed(self):
        x1 = np.random.uniform(low=-1.0, high=1.0, size=1)[0]
        x2 = np.random.uniform(low=-1.0, high=1.0, size=1)[0]
        test = x1**2 + x2**2

        while(test >= 1.0):
                x1 = np.random.uniform(low=-1.0, high=1.0, size=1)[0]
                x2 = np.random.uniform(low=-1.0, high=1.0, size=1)[0]
                test = x1**2 + x2**2

        x = 2.0*x1*(1.0 -x1**2 - x2**2)**(0.5)
        y = 2.0*x2*(1.0 -x1**2 - x2**2)**(0.5)
        z = 1.0 - 2.0*(x1**2 + x2**2)

        cam_position = torch.tensor([x, y, z]).unsqueeze(0)
        if(z < 0):
            R = look_at_rotation_fixed(cam_position, up=((0, 0, -1),)).squeeze()
        else:
            R = look_at_rotation_fixed(cam_position, up=((0, 0, 1),)).squeeze()

        # Rotate in-plane
        rot_degrees = np.random.uniform(low=-90.0, high=90.0, size=1)
        rot = scipyR.from_euler('z', rot_degrees, degrees=True)
        rot_mat = torch.tensor(rot.as_matrix(), dtype=torch.float32)
        R = torch.matmul(R, rot_mat)
        R = R.squeeze()
        return R

    def generate_images(self, num_samples):
        data = {"images":[],
                "Rs":[]}

        for i in range(num_samples):
            # Sample rotation matrix
            R = self.sphere_wolfram_sampling_fixed()

            # Convert R matrix from pytorch to opengl format
            # for rendering only!
            xy_flip = np.eye(3, dtype=np.float)
            xy_flip[0,0] = -1.0
            xy_flip[1,1] = -1.0
            R_opengl = np.dot(R,xy_flip)
            R_opengl = np.transpose(R_opengl)

            # Render image
            img = self.dataset.render_training_image(R_opengl)

            # Convert BGR to RGB
            img = np.flip(img,axis=2)

            data["images"].append(img)
            data["Rs"].append(R)
        return data

    def __iter__(self):
        self.curr_samples = 0
        return self

    def __next__(self):
        if(self.curr_samples < self.max_samples):
            self.curr_samples += self.batch_size
            return self.generate_samples(self.batch_size)
        else:
            raise StopIteration

    def generate_samples(self, num_samples):
        data = self.generate_images(num_samples)
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
