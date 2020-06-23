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
import cv2 as cv

import time
import pickle
import random
import yaml
from utils import *


def load_tless_dataset(folder_path):
    img_path = os.path.join(folder_path, "rgb")
    gt_path = os.path.join(folder_path, "gt.yml")
    img_names= glob.glob(img_path + "*.png")

    with open(gt_path, 'r') as fp:
        dataset = yaml.load(fp, Loader=yaml.FullLoader)

    # Add images to the dataset dict
    for k in dataset.keys():
        img_name = "{:04d}.png".format(int(k))
        print("Loading image: {0}".format(img_name))
        curr_img = cv2.imread(os.path.join(img_path, img_name))
        dataset[k][0]['image'] = curr_img/255.0
    return dataset


# Parse parameters
parser = argparse.ArgumentParser()
parser.add_argument("dataset_path", help="path to the T-LESS dataset folder containing .yml groundtruth file and 'rgb' directory with images")
parser.add_argument("-v", help="visualize the data", type=bool, default=False)
parser.add_argument("-o", help="output path", default="")
arguments = parser.parse_args()
visualize = arguments.v
output_path = arguments.o

# Load the T-LESS dataset
dataset = load_tless_dataset(arguments.dataset_path)

# Loop through the T-LESS dataset
Rs = []
ts = []
images = []
for k in dataset.keys():
    # Fetch and crop image
    curr_img = dataset[k][0]["image"]
    obj_bb = np.array(dataset[k][0]["obj_bb"])
    cropped_img = extract_square_patch(curr_img, obj_bb)
    images.append(cropped_img)

    # Format pose data (OpenCV/TLESS format --> pytorch3d format)
    curr_R = np.array(dataset[k][0]["cam_R_m2c"]).reshape(3,3)

    # Rotate 180 degrees around z
    curr_R = np.vstack([-curr_R[0,:],
                   -curr_R[1,:],
                   curr_R[2,:]])

    # Inverse rot matrix
    curr_R = np.linalg.inv(curr_R)

    Rs.append(curr_R)
    curr_t = np.array(dataset[k][0]["cam_t_m2c"]).reshape(3,1)
    ts.append(curr_t)

    if(visualize):
        plt.figure(figsize=(6, 2))
        plt.subplot(1, 3, 1)
        plt.imshow(curr_img)
        plt.title("Original image")

        plt.subplot(1, 3, 2)
        plt.imshow(cropped_img)
        plt.title("Cropped image")
        plt.show()

data={"images":images,
      "Rs":Rs,
      "ts":ts,
      "elevs":[],
      "azims":[],
      "dist":[],
      "light_dir":[]}

if(output_path == ""):
    output_path = "./training-images.p"
pickle.dump(data, open(output_path, "wb"), protocol=2)
