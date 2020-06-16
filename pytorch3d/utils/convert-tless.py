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

def loadCheckpoint(model_path):
    # Load checkpoint and parameters
    checkpoint = torch.load(model_path)
    epoch = checkpoint['epoch'] + 1
    learning_rate = checkpoint['learning_rate']

    # Load model
    model = Model(output_size=6)
    model.load_state_dict(checkpoint['model'])

    # Load optimizer
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
    optimizer.load_state_dict(checkpoint['optimizer'])
    print("Loaded the checkpoint: \n" + model_path)
    return model, optimizer, epoch, learning_rate

def latestCheckpoint(model_dir):
    checkpoints = glob.glob(os.path.join(model_dir, "*.pt"))
    checkpoints_sorted = sorted(checkpoints, key=os.path.getmtime)
    if(len(checkpoints_sorted) > 0):
        return checkpoints_sorted[-1]
    return None

def load_tless_dataset(folder_path):
    img_path = os.path.join(folder_path, "rgb")
    gt_path = os.path.join(folder_path, "gt.yml")
    img_names= glob.glob(img_path + "*.png")

    with open(gt_path, 'r') as fp:
        dataset = yaml.load(fp, Loader=yaml.FullLoader)

    with open(cam_path, 'r') as fp:
        camset = yaml.load(fp, Loader=yaml.FullLoader)

    # Add images to the dataset dict
    i = 0
    for k in dataset.keys():
        i += 1
        if i > 100:
            #break
            pass
        img_name = "{:04d}.png".format(int(k))
        print("Loading image: {0}".format(img_name))
        curr_img = cv2.imread(os.path.join(img_path, img_name))
        dataset[k][0]['image'] = curr_img/255.0
    return dataset, camset

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
       return v
    return v / norm

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

# Lets use a model to see what we get from it
model_path = latestCheckpoint(os.path.join('./output/depth/aug-obj19-l1-clamped-depth-aug-bg-dataset-fixed-lr/', "models/"))
model, _, _, _ = loadCheckpoint(model_path)

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
        print('cam_K')
        print(cam_K)
        print('cam_t_w2c')
        print(cam_t_w2c)
        print('cam_R_w2c')
        print(cam_R_w2c)
        print('cam_t_m2c')
        print(cam_t_m2c)
        print('cam_R_m2c')
        print(cam_R_m2c)

        t_w2c_norm = normalize(cam_t_m2c.flatten())
        length = np.linalg.norm(cam_t_m2c.flatten())
        print('t_w2c_norm')
        print(t_w2c_norm)
        orig = [0, 0, 1]
        dot = np.dot(t_w2c_norm, orig)
        cross = np.linalg.norm(np.cross(t_w2c_norm, orig))
        mat_corr = np.array([[dot, -cross, 0], [cross, dot, 0], [0, 0, 1]])
        print('mat_corr')
        print(mat_corr)

        cam_R_m2c_corrected = np.dot(cam_R_m2c, mat_corr)

        print('cam_R_m2c_transformed_corrected')
        print(cam_R_m2c_corrected)

        #Rotate 180 degrees around z
        cam_R_m2c_transformed = np.vstack([-cam_R_m2c_corrected[0,:],
                      -cam_R_m2c_corrected[1,:],
                      cam_R_m2c_corrected[2,:]])
        #Inverse rot matrix
        cam_R_m2c_transformed = np.linalg.inv(cam_R_m2c_transformed)

        print('cam_R_m2c_transformed')
        print(cam_R_m2c_transformed)

        # Pm2i = [cam_R_m2c[0].append(cam_t_m2c[0])] \
        #         + [cam_R_m2c[1].append(cam_t_m2c[1])] \
        #         + [cam_R_m2c[2].append(cam_t_m2c[2])] \
        #         + [[0, 0, 0, 1]]

        # temp = np.block([cam_R_m2c, cam_t_m2c])
        # Pm2i = np.block([[temp], [np.array([0, 0, 0, 1])]])
        #
        # Pm2i_inv = np.linalg.inv(Pm2i)
        #
        # temp = np.block([cam_R_w2c, cam_t_w2c])
        # Pw2i = np.block([[temp], [np.array([0, 0, 0, 1])]])
        #
        # Pw = np.block([[cam_t_w2c], [1]])
        #
        # temp = np.dot(Pm2i_inv, Pw2i)
        # Pm = np.dot(temp, Pw)
        #
        # print(Pm)

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

            renderer = BatchRender("./data/t-less_v2/models_cad/obj_19.obj",
            #renderer = BatchRender("./data/t-less-obj19/cad/obj_19_scaled.obj",
                             device,
                             batch_size=1,
                             faces_per_pixel=1,
                             render_method="hard-phong",
                             image_size=128)


            Rs_gt = torch.tensor(np.stack([cam_R_m2c_transformed]), device=renderer.device,
                                   dtype=torch.float32)
            views = prepareViews([[0,0,0]])
            rot = R.from_euler('xyz', [0, 0, 0], degrees=True)
            rot_mat = torch.tensor(rot.as_matrix(), dtype=torch.float32)
            T = torch.tensor(np.array([[ -cam_t_w2c[0][0],  -cam_t_w2c[1][0], cam_t_w2c[2][0]]]), device=renderer.device,
                                   dtype=torch.float32)

            T = torch.tensor(np.array([[0, 0, length]]), device=renderer.device,
                                   dtype=torch.float32)

            Rs_new = torch.matmul(Rs_gt, rot_mat.to(renderer.device))

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
