import torch
from DatasetGeneratorOpenGL import DatasetGenerator
from Pipeline import Pipeline
from Encoder import Encoder
from Model import Model
import json
import pickle
import numpy as np

def main():
    device = torch.device("cuda:0")
    # Need to replace sundermeyer-random with something where we can
    # use predetermined poses, to plot each arch to visualize
    data = DatasetGenerator("",
                            "./data/cad-files/ply-files/obj_10.ply",
                            375,
                            8,
                            "not_used",
                            device,
                            "sundermeyer-random")

    # use fixed data instead
    data = loadDataset(['./data/validationsets/tless-train-obj10.p'], 8)

    # load model
    encoder = Encoder("./data/obj1-18/encoder.npy").to(device)
    encoder.eval()
    checkpoint = torch.load("./output/depth/multi-path-reconst/depth-reconst/obj10-10views/models/model-epoch200.pt") # Change this later
    model = Model(num_views=10).cuda()
    model.load_state_dict(checkpoint['model'])
    model = model.eval()
    pipeline = Pipeline(encoder, model, device)

    # run images through model
    # Predict poses
    for i,curr_batch in enumerate(data):
        if i > 160:
            break
        predicted_poses = pipeline.process(curr_batch["images"])
        Rs = curr_batch["Rs"]

    # evaluate how output confidence and each view changes with input pose
    print(predicted_poses)

# Copied from train.py for now
def loadDataset(file_list, batch_size=2):
    #data = {"codes":[],"Rs":[],"images":[]}
    data = []
    for f in file_list:
        print("Loading dataset: {0}".format(f))
        with open(f, "rb") as f:
            curr_data = pickle.load(f, encoding="latin1")
            curr_batch = {"Rs":[],"images":[]}
            for i in range(len(curr_data["Rs"])):
                curr_pose = curr_data["Rs"][i]

                # Convert from T-LESS to Pytorch3D format
                xy_flip = np.eye(3, dtype=np.float)
                xy_flip[0,0] = -1.0
                xy_flip[1,1] = -1.0
                curr_pose = np.transpose(curr_pose)
                curr_pose = np.dot(curr_pose, xy_flip)
                curr_batch["Rs"].append(curr_pose)

                # Normalize image
                curr_image = curr_data["images"][i]
                curr_image = curr_image/np.max(curr_image)
                curr_batch["images"].append(curr_image)

                if(len(curr_batch["Rs"]) >= batch_size):
                    data.append(curr_batch)
                    curr_batch = {"Rs":[],"images":[]}
            data.append(curr_batch)
    return data

if __name__ == '__main__':
    main()
