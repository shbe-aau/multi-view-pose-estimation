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
import gc

from utils.utils import *

from Model import Model
from Encoder import Encoder
from Pipeline import Pipeline
from BatchRender import BatchRender
from losses import Loss
from DatasetGeneratorOpenGL import DatasetGenerator

optimizer = None
lr_reducer = None
pipeline = None
views = []
epoch = 0

dbg_memory = False

def dbg(message, flag):
    if flag:
        print(message)

def latestCheckpoint(model_dir):
    checkpoints = glob.glob(os.path.join(model_dir, "*.pt"))
    checkpoints_sorted = sorted(checkpoints, key=os.path.getmtime)
    if(len(checkpoints_sorted) > 0):
        return checkpoints_sorted[-1]
    return None

def loadCheckpoint(model_path):
    # Load checkpoint and parameters
    checkpoint = torch.load(model_path)
    epoch = checkpoint['epoch'] + 1

    # Load model
    num_views = int(checkpoint['model']['l3.bias'].shape[0]/(6+1))
    model = Model(num_views=num_views).cuda()

    model.load_state_dict(checkpoint['model'])

    # Load optimizer
    optimizer = torch.optim.Adam(model.parameters())
    optimizer.load_state_dict(checkpoint['optimizer'])

    lr_reducer = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.9)
    lr_reducer.load_state_dict(checkpoint['lr_reducer'])

    print("Loaded the checkpoint: \n" + model_path)
    return model, optimizer, epoch, lr_reducer

def loadDataset(file_list, batch_size=2):
    #data = {"codes":[],"Rs":[],"images":[]}
    data = []
    for f in file_list:
        print("Loading dataset: {0}".format(f))
        with open(f, "rb") as f:
            curr_data = pickle.load(f, encoding="latin1")
            curr_batch = {"codes":[],"Rs":[],"images":[]}
            for i in range(len(curr_data["codes"])):
                curr_batch["codes"].append(curr_data["codes"][i])
                curr_batch["Rs"].append(curr_data["Rs"][i])
                curr_batch["images"].append(curr_data["images"][i])
                if(len(curr_batch["codes"]) >= batch_size):
                    data.append(curr_batch)
                    curr_batch = {"codes":[],"Rs":[],"images":[]}
            data.append(curr_batch)
    return data

def main():
    global optimizer, lr_reducer, views, epoch, pipeline
    # Read configuration file
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_name")
    arguments = parser.parse_args()

    cfg_file_path = os.path.join("./experiments", arguments.experiment_name)
    args = configparser.ConfigParser()
    args.read(cfg_file_path)

    # Prepare rotation matrices for multi view loss function
    eulerViews = json.loads(args.get('Rendering', 'VIEWS'))
    views = prepareViews(eulerViews)

    # Set the cuda device
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Set up batch renderer
    br = BatchRender(args.get('Dataset', 'CAD_PATH'),
                     device,
                     batch_size=args.getint('Training', 'BATCH_SIZE'),
                     faces_per_pixel=args.getint('Rendering', 'FACES_PER_PIXEL'),
                     render_method=args.get('Rendering', 'SHADER'),
                     image_size=args.getint('Rendering', 'IMAGE_SIZE'))

    # Set size of model output depending on pose representation - deprecated?
    pose_rep = args.get('Training', 'POSE_REPRESENTATION')
    if(pose_rep == '6d-pose'):
        pose_dim = 6
    elif(pose_rep == 'quat'):
        pose_dim = 4
    elif(pose_rep == 'axis-angle'):
        pose_dim = 4
    elif(pose_rep == 'euler'):
        pose_dim = 3
    else:
        print("Unknown pose representation specified: ", pose_rep)
        pose_dim = -1

    # Initialize a model using the renderer, mesh and reference image
    model = Model(num_views=len(views))
    model.to(device)

    # Create an optimizer. Here we are using Adam and we pass in the parameters of the model
    learning_rate=args.getfloat('Training', 'LEARNING_RATE')
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
    lr_reducer = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.9)

    # Prepare output directories
    output_path = args.get('Training', 'OUTPUT_PATH')
    prepareDir(output_path)
    shutil.copy(cfg_file_path, os.path.join(output_path, cfg_file_path.split("/")[-1]))

    # Not used?
    mean = 0
    std = 1

    # Setup early stopping if enabled
    early_stopping = args.getboolean('Training', 'EARLY_STOPPING', fallback=False)
    if early_stopping:
        window = args.getint('Training', 'STOPPING_WINDOW', fallback=10)
        time_limit = args.getint('Training', 'STOPPING_TIME_LIMIT', fallback=10)
        window_means = []
        lowest_mean = np.inf
        lowest_x = 0
        timer = 0

    # Load checkpoint for last epoch if it exists
    model_path = latestCheckpoint(os.path.join(output_path, "models/"))
    if(model_path is not None):
        model, optimizer, epoch, lr_reducer = loadCheckpoint(model_path)

    if early_stopping:
        validation_csv=os.path.join(output_path, "validation-loss.csv")
        if os.path.exists(validation_csv):
            with open(validation_csv) as f:
                val_reader = csv.reader(f, delimiter='\n')
                val_loss = list(val_reader)
            val_losses = np.array(val_loss, dtype=np.float32).flatten()
            for epoch in range(window,len(val_loss)):
                timer += 1
                w_mean = np.mean(val_losses[epoch-window:epoch])
                window_means.append(w_mean)
                if w_mean < lowest_mean:
                    lowest_mean = w_mean
                    lowest_x = epoch
                    timer = 0


    # Prepare pipeline
    encoder = Encoder(args.get('Dataset', 'ENCODER_WEIGHTS')).to(device)
    encoder.eval()
    pipeline = Pipeline(encoder, model, device)

    # Prepare datasets
    bg_path = "../../autoencoder_ws/data/VOC2012/JPEGImages/"
    training_data = DatasetGenerator(args.get('Dataset', 'BACKGROUND_IMAGES'),
                                     args.get('Dataset', 'CAD_PATH'),
                                     json.loads(args.get('Rendering', 'T'))[-1],
                                     args.getint('Training', 'BATCH_SIZE'),
                                     "not_used",
                                     device,
                                     args.get('Training', 'VIEW_SAMPLING'))
    training_data.max_samples = args.getint('Training', 'NUM_SAMPLES')

    # Load the validationset
    validation_data = loadDataset(json.loads(args.get('Dataset', 'VALID_DATA_PATH')),
                                  args.getint('Training', 'BATCH_SIZE'))
    print("Loaded validation set!")

    # Start training
    np.random.seed(seed=args.getint('Training', 'RANDOM_SEED'))
    while(epoch < args.getint('Training', 'NUM_ITER')):
        # Train on synthetic data
        loss = trainEpoch(mean, std, br, training_data, model, device, output_path,
                          loss_method=args.get('Training', 'LOSS'),
                          pose_rep=args.get('Training', 'POSE_REPRESENTATION'),
                          t=json.loads(args.get('Rendering', 'T')),
                          visualize=args.getboolean('Training', 'SAVE_IMAGES'),
                          loss_params=args.getfloat('Training', 'LOSS_PARAMS'))
        append2file([loss], os.path.join(output_path, "train-loss.csv"))

        # Test on validation data
        val_loss = testEpoch(mean, std, br, validation_data, model, device, output_path,
                             "vsd-predicted-view-log-fixed",
                             pose_rep=args.get('Training', 'POSE_REPRESENTATION'),
                             t=json.loads(args.get('Rendering', 'T')),
                             visualize=args.getboolean('Training', 'SAVE_IMAGES'),
                             loss_params=args.getfloat('Training', 'LOSS_PARAMS'))
        append2file([val_loss], os.path.join(output_path, "validation-loss.csv"))

        # Plot losses
        val_losses = plotLoss(os.path.join(output_path, "train-loss.csv"),
                 os.path.join(output_path, "train-loss.png"),
                 validation_csv=os.path.join(output_path, "validation-loss.csv"))
        print("-"*20)
        print("Epoch: {0} - train loss: {1} - validation loss: {2}".format(epoch,loss,val_loss))
        print("-"*20)
        if early_stopping and epoch >= window:
            timer += 1
            if timer > time_limit:
                # print stuff here
                print()
                print("-"*60)
                print("Validation loss seems to have plateaued, stopping early.")
                print("Best mean loss value over an epoch window of size {} was found at epoch {} ({:.8f} mean loss)".format(window, lowest_x, lowest_mean))
                print("-"*60)
                break
            w_mean = np.mean(val_losses[epoch-window:epoch])
            window_means.append(w_mean)
            if w_mean < lowest_mean:
                lowest_mean = w_mean
                lowest_x = epoch
                timer = 0
        epoch = epoch+1


def testEpoch(mean, std, br, dataset, model,
               device, output_path, loss_method, pose_rep, t,
               visualize=False, loss_params=0.5):
    torch.set_grad_enabled(False)
    loss = runEpoch(mean, std, br, dataset, model.eval(),
                    device, output_path, loss_method, pose_rep, t,
                    visualize, loss_params)
    return loss


def trainEpoch(mean, std, br, dataset, model,
               device, output_path, loss_method, pose_rep, t,
               visualize=False, loss_params=0.5):
    torch.set_grad_enabled(True)
    loss = runEpoch(mean, std, br, dataset, model.train(),
                    device, output_path, loss_method, pose_rep, t,
                    visualize, loss_params)
    return loss


def runEpoch(mean, std, br, dataset, model,
               device, output_path, loss_method, pose_rep, t,
               visualize, loss_params):
    global optimizer, lr_reducer
    dbg("Before train memory: {}".format(torch.cuda.memory_summary(device=device, abbreviated=False)), dbg_memory)

    if(model.training):
        print("Epoch: {0} - current learning rate: {1}".format(epoch, lr_reducer.get_last_lr()))

    losses = []
    batch_size = br.batch_size
    if(model.training):
        dataset.hard_samples = [] # Reset hard samples
    for i,curr_batch in enumerate(dataset):
        if(model.training):
            optimizer.zero_grad()

        # Fetch images
        input_images = curr_batch["images"]

        # Predict poses
        predicted_poses = pipeline.process(input_images)

        # Prepare ground truth poses for the loss function
        T = np.array(t, dtype=np.float32)
        Rs = curr_batch["Rs"]
        ts = [T.copy() for t in Rs]

        # Calculate the loss
        loss, batch_loss, gt_images, predicted_images = Loss(predicted_poses, Rs, br, ts,
                                                             mean, std, loss_method=loss_method, pose_rep=pose_rep, views=views, loss_params=loss_params)

        Rs = torch.tensor(np.stack(Rs), device=device, dtype=torch.float32)

        print("Grad: ", loss.requires_grad)

        if(model.training):
            loss.backward()
            optimizer.step()

            # Save difficult samples
            k = int(len(curr_batch["images"])*(dataset.hard_sample_ratio))
            batch_loss = batch_loss.squeeze()
            top_val, top_ind = torch.topk(batch_loss, k)
            hard_samples = Rs[top_ind]

            # Convert hard samples to a list
            hard_list = []
            for h in np.arange(hard_samples.shape[0]):
                hard_list.append(hard_samples[h])
            dataset.hard_samples = hard_list

        #detach all from gpu
        loss.detach().cpu().numpy()
        gt_images.detach().cpu().numpy()
        predicted_images.detach().cpu().numpy()

        if(model.training):
            print("Batch: {0}/{1} (size: {2}) - loss: {3}".format(i+1,round(dataset.max_samples/batch_size), len(Rs),torch.mean(batch_loss)))
        else:
            print("Test batch: {0}/{1} (size: {2}) - loss: {3}".format(i+1,len(dataset), len(Rs),torch.mean(batch_loss)))
            #print("Test batch: {0}/{1} (size: {2}) - loss: {3}".format(i+1, round(dataset.max_samples/batch_size), len(Rs),torch.mean(batch_loss)))
        losses = losses + batch_loss.data.detach().cpu().numpy().tolist()

        if(visualize):
            if(model.training):
                batch_img_dir = os.path.join(output_path, "images/epoch{0}".format(epoch))
            else:
                batch_img_dir = os.path.join(output_path, "val-images/epoch{0}".format(epoch))
            prepareDir(batch_img_dir)
            gt_img = (gt_images[0]).detach().cpu().numpy()
            predicted_img = (predicted_images[0]).detach().cpu().numpy()

            vmin = np.linalg.norm(T)*0.9
            vmax = max(np.max(gt_img), np.max(predicted_img))

            fig = plt.figure(figsize=(12,3+len(views)*2))
            #for viewNum in np.arange(len(views)):
            plotView(0, len(views), vmin, vmax, input_images, gt_images, predicted_images,
                     predicted_poses, batch_loss, batch_size, threshold=loss_params)
            fig.tight_layout()

            fig.savefig(os.path.join(batch_img_dir, "epoch{0}-batch{1}.png".format(epoch,i)), dpi=fig.dpi)
            plt.close()

    if(model.training):
        # Save current model
        model_dir = os.path.join(output_path, "models/")
        prepareDir(model_dir)
        state = {'model': model.state_dict(),
                 'optimizer': optimizer.state_dict(),
                 'lr_reducer': lr_reducer.state_dict(),
                 'epoch': epoch}
        torch.save(state, os.path.join(model_dir,"model-epoch{0}.pt".format(epoch)))
        #lr_reducer.step()

    # Memory management
    dbg("After train memory: {}".format(torch.cuda.memory_summary(device=device, abbreviated=False)), dbg_memory)
    gc.collect()
    return np.mean(losses)


if __name__ == '__main__':
    main()
