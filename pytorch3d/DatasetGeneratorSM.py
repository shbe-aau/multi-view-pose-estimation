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
                 num_bgs=5000):
        self.curr_samples = 0
        self.max_samples = 1000
        self.obj_path = obj_path
        self.batch_size = batch_size

        args = configparser.ConfigParser()
        args.read("test.cfg")

        args.set('Paths', 'MODEL_PATH', obj_path.replace(".obj",".ply"))
        args.set('Dataset', 'NOOF_TRAINING_IMGS', str(int(20000))) #str(int(batch_size*2)))

        self.dataset = build_dataset("./dataset-test", args)
        self.dataset.render_training_images()
        self.dataset.load_bg_images("./dataset-test")


    def opengl2pytorch(self, R):
        # Convert R matrix from opengl to pytorch format
        xy_flip = np.eye(3, dtype=np.float)
        xy_flip[0,0] = -1.0
        xy_flip[1,1] = -1.0
        R_conv = np.transpose(R)
        R_conv = np.dot(R_conv,xy_flip)
        return R_conv

    def generate_images(self, num_samples):
        data = {"images":[],
                "Rs":[]}

        img_x, _, Rs = self.dataset.batch(num_samples)

        # Convert rotation matrices
        for i in range(Rs.shape[0]):
            Rs[i] = self.opengl2pytorch(Rs[i])

        # Convert BGR to RGB
        for i in range(img_x.shape[0]):
            img_x[i] = np.flip(img_x[i],axis=2)

        data["images"] = img_x #data["images"][:num_samples]
        data["Rs"] = Rs #data["Rs"][:num_samples]
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
