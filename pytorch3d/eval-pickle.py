import os
import shutil
import torch
import numpy as np
import pickle
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import configparser
import json
import argparse
import glob

import copy

from utils.utils import *
from utils.tools import *

from Model import Model
from Encoder import Encoder
from utils.pytless import inout, misc
from utils.pytless.renderer import Renderer

def loadCheckpoint(model_path):
    # Load checkpoint and parameters
    checkpoint = torch.load(model_path)
    epoch = checkpoint['epoch'] + 1

    # Load model
    model = Model(output_size=6).cuda()
    model.load_state_dict(checkpoint['model'])

    # Load optimizer
    optimizer = torch.optim.Adam(model.parameters())
    optimizer.load_state_dict(checkpoint['optimizer'])
    
    print("Loaded the checkpoint: \n" + model_path)
    return model, optimizer, epoch, None

def correct_trans_offset(R, t_est):
    # Translation offset correction
    d_alpha_x = np.arctan(t_est[0]/t_est[2])
    d_alpha_y = np.arctan(t_est[1]/t_est[2])
    R_corr_x = np.array([[1,0,0],
                         [0,np.cos(d_alpha_y),-np.sin(d_alpha_y)],
                         [0,np.sin(d_alpha_y),np.cos(d_alpha_y)]]) 
    R_corr_y = np.array([[np.cos(d_alpha_x),0,-np.sin(d_alpha_x)],
                         [0,1,0],
                         [np.sin(d_alpha_x),0,np.cos(d_alpha_x)]]) 
    R_corrected = np.dot(R_corr_y,np.dot(R_corr_x,R))
    return R_corrected
    


def main():
    visualize = True
    
    # Read configuration file
    parser = argparse.ArgumentParser()
    parser.add_argument("-mp", help="path to the model checkpoint")
    parser.add_argument("-ep", help="path to the encoder weights")
    parser.add_argument("-pi", help="path to the pickle input file")
    args = parser.parse_args()
    
    # Set the cuda device 
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)                 

    # Initialize a model
    model = Model(output_size=6).to(device)   

    # Load model checkpoint
    model, optimizer, epoch, learning_rate = loadCheckpoint(args.mp)
    model.to(device)
    model.eval()

    # Load and prepare encoder
    encoder = Encoder(args.ep).to(device)
    encoder.eval()

    # Load dataset
    data = pickle.load(open(args.pi,"rb"), encoding="latin1")


    obj_path = "/shared-folder/AugmentedAutoencoder/pytorch3d/data/t-less-obj19/cad/obj_19.ply"
    obj_model = inout.load_ply(obj_path.replace(".obj",".ply"))
    img_size = 320

    K = np.array([1075.65091572, 0.0, 320.0/2.0,
                  0.0, 1073.90347929, 320.0/2.0,
                  0.0, 0.0, 1.0]).reshape(3,3)
    
    renderer = Renderer(obj_model, (img_size,img_size), K,
                        surf_color=(1, 1, 1), mode='rgb', random_light=False)

    # Loop through dataset
    for i,img in enumerate(data["images"]):

        #if(data["visib_fract"][i] < 0.5):
        #    continue
        
        print("Current image: {0}/{1}".format(i+1,len(data["images"])))

        # Run through encoder
        img_torch = torch.from_numpy(img).unsqueeze(0).permute(0,3,1,2).to(device)
        code = encoder(img_torch.float())

        # Run through model
        predicted_poses = model(code)
        Rs_predicted = compute_rotation_matrix_from_ortho6d(predicted_poses)

        R = Rs_predicted.detach().cpu().numpy()[0]
        
        # Invert xy axes
        xy_flip = np.eye(3, dtype=np.float)
        xy_flip[0,0] = -1.0
        xy_flip[1,1] = -1.0
        R = R.dot(xy_flip)

        # Inverse rotation matrix
        R = np.transpose(R)
        
        if(visualize):

            t_gt = np.array(data["ts"][i])
            t = np.array([0,0,500])

            # Render predicted pose
            R = correct_trans_offset(R,t_gt)
            ren_predicted = renderer.render(R, t)

            # Render groundtruth pose
            R_gt = data["Rs"][i]
            R_gt = correct_trans_offset(R_gt,t_gt)
            ren_gt = renderer.render(R_gt, t)
            
            cv2.imshow("input image", np.flip(img,axis=2))
            cv2.imshow("gt render", np.flip(ren_gt,axis=2))
            cv2.imshow("predict render", np.flip(ren_predicted,axis=2))
            key = cv2.waitKey(0)            
            if(key == ord("q")):
                visualize = False
                #break
                continue
    

if __name__ == '__main__':
    main()
