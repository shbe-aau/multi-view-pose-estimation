import torch
from DatasetGeneratorOpenGL import DatasetGenerator
from Pipeline import Pipeline
from Encoder import Encoder
from Model import Model

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
                               "sundermeyer-random")

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
    for i,curr_batch in enumerate(datagen):
        predicted_poses = pipeline.process(curr_batch["images"])
        Rs = curr_batch["Rs"]

    # evaluate how output confidence and each view changes with input pose
    print(predicted_poses)

if __name__ == '__main__':
    main()
