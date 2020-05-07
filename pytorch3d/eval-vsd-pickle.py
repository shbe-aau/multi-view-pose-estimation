import os
import shutil
import torch
import numpy as np
import pickle
import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
import configparser
import json
import argparse
import glob

import copy

from utils.utils import *
from utils.tools import *

from BatchRender import BatchRender 


def main():
    # Read configuration file
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_name")
    arguments = parser.parse_args()
    
    cfg_file_path = os.path.join("./experiments", arguments.experiment_name)
    args = configparser.ConfigParser()
    args.read(cfg_file_path)

    # Set the cuda device 
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Set up batch renderer
    br = BatchRender(args.get('Dataset', 'CAD_PATH'),
                     device,
                     batch_size=args.getint('Evaluation', 'BATCH_SIZE'),
                     render_method="hard-depth",
                     image_size=args.getint('Rendering', 'IMAGE_SIZE'))
                   

    data = pickle.load(open(args.get('Evaluation', 'TEST_DATA_PATH'),"rb"), encoding="latin1")
    output_path = args.get('Training', 'OUTPUT_PATH')
    prepareDir(output_path)
    shutil.copy(cfg_file_path, os.path.join(output_path, cfg_file_path.split("/")[-1]))

    evalEpoch(br, data, output_path,
              t=json.loads(args.get('Rendering', 'T')),
              visualize=args.getboolean('Evaluation', 'SAVE_IMAGES'))
    
def evalEpoch(br, data, output_path, t, visualize=False):
    batch_size = br.batch_size
    num_samples = len(data["Rs"])
    data_indeces = np.arange(num_samples)

    test_dir = os.path.join(output_path, "test")
    test_dir_neg = os.path.join(test_dir, "negative")
    test_dir_pos = os.path.join(test_dir, "positive")
    prepareDir(test_dir_pos)
    prepareDir(test_dir_neg)


    rgb_render = BatchRender(obj_path=br.obj_path,
                             device=br.device,
                             batch_size=br.batch_size,
                             render_method="hard-phong",
                             image_size=br.image_size)

    fig = None
    for i,curr_batch in enumerate(batch(data_indeces, batch_size)):

        Rs_predicted = []
        for b in curr_batch:
            curr = data["Rs_predicted"][b]

            # Inverse rotation matrix
            curr = np.transpose(curr)

            # Invert z axis
            # See auto_pose/meshrenderer/gl_utils/camera.py - line 90
            z_flip = np.eye(3, dtype=np.float)
            z_flip[2,2] = -1.0
            curr = curr.dot(z_flip)

            # Rotate 180 around z axis
            curr = np.vstack([-curr[0,:],
                              -curr[1,:],
                              curr[2,:]])
            
            Rs_predicted.append(curr)
            data["Rs_predicted"][b] = Rs_predicted[-1]

        # Prepare ground truth poses for the loss function
        T = np.array(t, dtype=np.float32)
        Rs = []
        ts = []
        for b in curr_batch:
            curr = data["Rs"][b]
            #curr = np.transpose(curr)
            Rs.append(curr)
            ts.append(T.copy())

        # Render depth images
        predicted_images = br.renderBatch(Rs_predicted, ts).cpu().detach().numpy()
        gt_images = br.renderBatch(Rs, ts).cpu().detach().numpy()

        predicted_rgbs = rgb_render.renderBatch(Rs_predicted, ts).cpu().detach().numpy()
        gt_rgbs = rgb_render.renderBatch(Rs, ts).cpu().detach().numpy()
        
        # Calculate VSD
        thau = 200 / 1000 #20 mm in meters
        sigma = 0.3

        losses = []
        for obj_k in np.arange(len(curr_batch)):
            curr_prediction = predicted_images[obj_k]
            curr_gt = gt_images[obj_k]

            # Calc difference
            diff = np.abs(curr_prediction - curr_gt)

            # Apply visibility masks
            mask_prediction = curr_prediction == -1
            mask_gt = curr_gt == -1
            diff[mask_gt] == 0
            diff[mask_prediction] == 0
            
            outliers = diff[diff > thau]
            total = diff[diff != 0]
            vsd = len(outliers)/(len(total)+1)

            losses.append(vsd)
            
            print("Sample: {0}/{1} - VSD: {2}".format(curr_batch[obj_k],num_samples,vsd))

            if(visualize):
                if(fig is None):
                    fig = plt.figure(figsize=(16,14))
                fig.suptitle("Sample {0} - VSD: {1}".format(curr_batch[obj_k],vsd))

                plt.subplot(2, 3, 1)
                plt.imshow(data["images"][curr_batch[obj_k]])
                plt.title("GT dataset")

                if("codebook_images" in data.keys()):
                    plt.subplot(2, 3, 2)
                    plt.imshow(data["codebook_images"][curr_batch[obj_k]])
                    plt.title("Codebook image")
                else:                    
                    plt.subplot(2, 3, 2)
                    plt.imshow(gt_rgbs[obj_k])
                    plt.title("GT rendered")

                plt.subplot(2, 3, 3)
                plt.imshow(predicted_rgbs[obj_k])
                plt.title("Predicted rendered")

                plt.subplot(2, 3, 4)
                plt.imshow(curr_gt, vmin=14, vmax=16)
                plt.title("GT rendered\n {0}".format(data["Rs"][curr_batch[obj_k]]))
                
                plt.subplot(2, 3, 5)
                plt.imshow(curr_prediction, vmin=14, vmax=16)
                plt.title("Predicted\n {0}".format(data["Rs_predicted"][curr_batch[obj_k]]))

                plt.subplot(2, 3, 6)
                plt1 = plt.imshow(diff, vmin=0, vmax=thau, cmap=plt.get_cmap('jet'))
                #fig.colorbar(plt1, fraction=0.05, pad=0.04)
                plt.title("Difference")

                #fig.tight_layout()

                if(vsd > 0.3):
                    output_dir = test_dir_neg
                else:
                    output_dir = test_dir_pos
                fig.savefig(os.path.join(output_dir, "sample{0}.png".format(curr_batch[obj_k])), dpi=fig.dpi,quality=50)
                plt.clf()
        # Save VSD scores to CSV file
        append2file(losses, os.path.join(test_dir, "vsd.csv"))

if __name__ == '__main__':
    main()
