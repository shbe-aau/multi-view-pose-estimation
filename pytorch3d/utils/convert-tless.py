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

import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from BatchRender import BatchRender


def load_tless_dataset(folder_path):
    img_path = os.path.join(folder_path, "rgb")
    gt_path = os.path.join(folder_path, "gt.yml")
    cam_path = os.path.join(folder_path, "info.yml")
    img_names= glob.glob(img_path + "*.png")

    with open(gt_path, 'r') as fp:
        dataset = yaml.load(fp, Loader=yaml.FullLoader)

    with open(cam_path, 'r') as fp:
        camset = yaml.load(fp, Loader=yaml.FullLoader)

    # Add images to the dataset dict
    for k in dataset.keys():
        img_name = "{:04d}.png".format(int(k))
        print("Loading image: {0}".format(img_name))
        curr_img = cv2.imread(os.path.join(img_path, img_name))
        dataset[k][0]['image'] = curr_img/255.0
    return dataset, camset


# Parse parameters
parser = argparse.ArgumentParser()
parser.add_argument("dataset_path", help="path to the T-LESS dataset folder containing .yml groundtruth file and 'rgb' directory with images")
parser.add_argument("-v", help="visualize the data", type=bool, default=False)
parser.add_argument("-o", help="output path", default="")
parser.add_argument("-n", help="object number", type=int, default=19)
arguments = parser.parse_args()
visualize = arguments.v
output_path = arguments.o
obj_num = arguments.n

# Load the T-LESS dataset
dataset, camset = load_tless_dataset(arguments.dataset_path)

# Loop through the T-LESS dataset
Rs = []
ts = []
images = []
for k in dataset.keys():
    # Fetch and crop image
    curr_img = dataset[k][0]["image"]

    cam_K = np.array(camset[k]["cam_K"]).reshape(3,3)
    cam_R_w2c = np.array(camset[k]["cam_R_w2c"]).reshape(3,3)
    cam_t_w2c = np.array(camset[k]["cam_t_w2c"]).reshape(3,1)
    depth_scale = camset[k]["depth_scale"]
    elev = camset[k]["elev"]
    mode = camset[k]["mode"]

    for j in dataset[k]:
        if j["obj_id"] != obj_num:
            continue

        obj_bb = np.array(j["obj_bb"])
        cropped_img = extract_square_patch(curr_img, obj_bb)
        images.append(cropped_img)

        # Format pose data
        cam_R_m2c = np.array(j["cam_R_m2c"]).reshape(3,3)
        Rs.append(cam_R_m2c)
        cam_t_m2c = np.array(j["cam_t_m2c"]).reshape(3,1)
        ts.append(cam_t_m2c)

        if(visualize):
            plt.figure(figsize=(6, 2))
            plt.subplot(1, 3, 1)
            plt.imshow(curr_img)
            plt.title("Original image")

            plt.subplot(1, 3, 2)
            plt.imshow(cropped_img)
            plt.title("Cropped image")

            # Set the cuda device
            device = torch.device("cuda:0")
            torch.cuda.set_device(device)

            #renderer = BatchRender("./data/t-less_v2/models_cad/obj_19.obj",
            renderer = BatchRender("./data/t-less-obj19/cad/obj_19_scaled.obj",
                             device,
                             batch_size=1,
                             faces_per_pixel=1,
                             render_method="hard-phong",
                             image_size=128)

            #print(cam_R_m2c)
            #cam_R_m2c = [[1, 0, 0],
            #          [0, 1, 0],
            #          [0, 0, 1]]
            r = R.from_matrix(cam_R_m2c)
            correction = R.from_euler('xyz', [0, 0, 180], degrees=True)
            r = correction * r
            r = r.inv()
            cam_R_m2c = r.as_matrix()
            print(cam_R_m2c)
            Rs_gt = torch.tensor(np.stack([cam_R_m2c]), device=renderer.device,
                                   dtype=torch.float32)
            views = prepareViews([[0,0,0]])
            rot = R.from_euler('xyz', [0, 0, 0], degrees=True)
            rot_mat = torch.tensor(rot.as_matrix(), dtype=torch.float32)
            print(cam_t_m2c)
            T = torch.tensor(np.array([[ 0.,  0., 15.]]), device=renderer.device,
            #T = torch.tensor(np.array([[0,0,cam_t_m2c[0][2]]]), device=renderer.device,
                                   dtype=torch.float32)

            Rs_new = torch.matmul(Rs_gt, rot_mat.to(renderer.device))
            print(Rs_gt)
            print(Rs_new)

            # print(cam_R_w2c)
            # print(cam_t_w2c)
            # print(cam_R_w2c + cam_t_w2c)
            # print(cam_K * cam_R_w2c)
            # Rs_gt = cam_K * cam_R_m2c
            # Rs_gt = torch.tensor(Rs_gt, device=renderer.device,
            #                         dtype=torch.float32)
            # T = torch.tensor(cam_t_m2c, device=renderer.device)

            gt_images = renderer.renderBatch(Rs_gt, T)
            #print(gt_images.shape)
            gt_images.detach().cpu().numpy()
            image = (gt_images[0]).detach().cpu().numpy()

            plt.subplot(1, 3, 3)
            plt.imshow(image)
            plt.title("Rendered image")

            #plt.subplot(1, 3, 3)
            #plt.imshow(cropped)
            #plt.title("Augmented image")
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
