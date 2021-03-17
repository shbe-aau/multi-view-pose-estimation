import torch
from DatasetGeneratorOpenGL import DatasetGenerator
from Pipeline import Pipeline
from Encoder import Encoder
from Model import Model
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt

def main():
    device = torch.device("cuda:0")
    # Need to replace sundermeyer-random with something where we can
    # use predetermined poses, to plot each arch to visualize
    datagen = DatasetGenerator("",
                            "./data/cad-files/ply-files/obj_10.ply",
                            375,
                            1,
                            "not_used",
                            device,
                            "sundermeyer-random",
                            random_light=False)

    # use fixed data instead
    # data = loadDataset(['./data/validationsets/tless-train-obj10.p'], 8)

    # load model
    encoder = Encoder("./data/obj1-18/encoder.npy").to(device)
    encoder.eval()
    #checkpoint = torch.load("./output/depth/multi-path-reconst/depth-reconst/obj10-10views/models/model-epoch200.pt") # Change this later
    checkpoint = torch.load("./output/paper-models/10views/obj10/models/model-epoch199.pt")

    model = Model(num_views=10).cuda()
    model.load_state_dict(checkpoint['model'])
    model = model.eval()
    pipeline = Pipeline(encoder, model, device)

    # run images through model
    # Predict poses
    num_datapoints = 100
    # around x
    shiftx = np.eye(3, dtype=np.float)
    theta = np.pi / num_datapoints
    shiftx[1,1] = np.cos(theta)
    shiftx[1,2] = -np.sin(theta)
    shiftx[2,2] = np.cos(theta)
    shiftx[2,1] = np.sin(theta)
    # around y
    shifty = np.eye(3, dtype=np.float)
    theta = np.pi / num_datapoints
    shifty[0,0] = np.cos(theta)
    shifty[0,2] = -np.sin(theta)
    shifty[2,2] = np.cos(theta)
    shifty[2,0] = np.sin(theta)
    # around z
    shiftz = np.eye(3, dtype=np.float)
    theta = np.pi / num_datapoints
    shiftz[0,0] = np.cos(theta)
    shiftz[0,1] = -np.sin(theta)
    shiftz[1,1] = np.cos(theta)
    shiftz[1,0] = np.sin(theta)
    predicted_poses = []
    R_conv = np.eye(3, dtype=np.float)
    #R_conv = np.array([[ 0.5435,  0.1365,  0.8283],
    #                   [ 0.6597,  0.5406, -0.5220],
    #                   [-0.5190,  0.8301,  0.2037]])
    #R_conv = np.array([[-0.7132,  0.0407,  0.6998],
    #                   [ 0.1696, -0.9586,  0.2287],
    #                   [ 0.6802,  0.2818,  0.6767]])
    #R_conv = np.array([[-0.9959,  0.0797,  0.0423],
    #                   [ 0.0444,  0.0249,  0.9987],
    #                   [ 0.0786,  0.9965, -0.0283]])
    for i in range(num_datapoints):
        # get data from fixed R and T vectors
        R_conv = np.matmul(R_conv, shiftx)
        R = torch.from_numpy(R_conv)
        t = torch.tensor([0.0, 0.0, 375])
        curr_batch = datagen.generate_image_batch(R = R, t = t, augment = False)

        predicted_poses.append(pipeline.process(curr_batch["images"]).detach().cpu().numpy()[0])
        Rs = curr_batch["Rs"]

    # evaluate how output confidence and each view changes with input pose
    predicted_poses = np.array(predicted_poses)
    print(predicted_poses[:,6])
    plot_confidences(predicted_poses)

def plot_confidences(predicted_poses):
    num_datapoints = len(predicted_poses)
    views = 10

    x = np.linspace(0, np.pi, num_datapoints)
    #fig, ax = plt.subplots(int(views/2),2)
    fig = plt.figure()
    fig.set_size_inches(15, 10)
    fig.tight_layout(rect=(0,0,0.95,1),pad=1.0)
    for i in range(views):
        #fig.add_subplot(int(views/2),2,i+1)
        plt.plot(x,predicted_poses[:,i], label=i)
        plt.ylim([-0.01, 1.01])
        plt.xlim([-0.01, 1.01])

    plt.legend(loc="upper right")
    # plt.show()
    fig.savefig('confidence.png', bbox_inches='tight')

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
