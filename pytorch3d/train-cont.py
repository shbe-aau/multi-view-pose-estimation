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
from BatchRender import BatchRender
from losses import Loss
from DatasetGeneratorOpenGL import DatasetGenerator
#from DatasetGeneratorSM import DatasetGenerator
#from DatasetGeneratorPytorch import DatasetGenerator

optimizer = None
lr_reducer = None
views = []
epoch = 0
dataset_gen = None

dbg_memory = False


from utils.utils import *
from utils.tools import *
from pytorch3d.transforms.transform3d import Rotate
from pytorch3d.loss import chamfer_distance

def ChamferLoss(Rs_gt, prediction, points, device):
    # Convert prediction to rotation matrix
    Rs_predicted = compute_rotation_matrix_from_ortho6d(prediction)

    # Apply gt to mesh points
    R = Rotate(Rs_gt).to(device)
    points_gt = R.transform_points(points)

    # Apply prediction to mesh points
    R = Rotate(Rs_predicted).to(device)
    points_pred = R.transform_points(points)

    # Calc chamfer loss
    loss_chamfer, _ = chamfer_distance(points_gt, points_pred, batch_reduction=None)

    return torch.mean(loss_chamfer), loss_chamfer

from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.io import load_obj, save_obj
from pytorch3d.structures import Meshes
def get_points(path, device):
    # We read the target 3D model using load_obj
    verts, faces, _ = load_obj(path)

    # verts is a FloatTensor of shape (V, 3) where V is the number of vertices in the mesh
    # faces is an object which contains the following LongTensors: verts_idx, normals_idx and textures_idx
    # For this tutorial, normals and textures are ignored.
    faces_idx = faces.verts_idx.to(device)
    verts = verts.to(device)

    # We scale normalize and center the target mesh to fit in a sphere of radius 1 centered at (0,0,0).
    # (scale, center) will be used to bring the predicted mesh to its original center and scale
    # Note that normalizing the target mesh, speeds up the optimization but is not necessary!
    center = verts.mean(0)
    verts = verts - center
    scale = max(verts.abs().max(0)[0])
    verts = verts / scale

    # We construct a Meshes structure for the target mesh
    trg_mesh = Meshes(verts=[verts], faces=[faces_idx])

    return sample_points_from_meshes(trg_mesh, 2000)



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
    output_size = checkpoint['model']['l3.bias'].shape[0]
    model = Model(output_size=output_size).cuda()
    model.load_state_dict(checkpoint['model'])

    # Load optimizer
    optimizer = torch.optim.Adam(model.parameters())
    optimizer.load_state_dict(checkpoint['optimizer'])

    lr_reducer = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.9)
    lr_reducer.load_state_dict(checkpoint['lr_reducer'])

    print("Loaded the checkpoint: \n" + model_path)
    return model, optimizer, epoch, lr_reducer

def loadDataset(file_list):
    data = {"codes":[],"Rs":[],"images":[]}
    for f in file_list:
        print("Loading dataset: {0}".format(f))
        with open(f, "rb") as f:
            curr_data = pickle.load(f, encoding="latin1")
            data["codes"] = data["codes"] + curr_data["codes"].copy()
            data["Rs"] = data["Rs"] + curr_data["Rs"].copy()
            data["images"] = data["images"] + curr_data["images"].copy()
    return data

def main():
    global optimizer, lr_reducer, views, epoch, dataset_gen
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

    # Set size of model output depending on pose representation
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
    model = Model(output_size=pose_dim)
    model.to(device)

    # Create an optimizer. Here we are using Adam and we pass in the parameters of the model
    learning_rate=args.getfloat('Training', 'LEARNING_RATE')
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
    lr_reducer = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.9)

    # Load the dataset
    #data = loadDataset(json.loads(args.get('Dataset', 'TRAIN_DATA_PATH')))
    data = []
    #print("Loaded dataset with {0} samples!".format(len(data["codes"])))

    # Load the validationset
    val_data = loadDataset(json.loads(args.get('Dataset', 'VALID_DATA_PATH')))
    print("Loaded validation set with {0} samples!".format(len(val_data["codes"])))

    output_path = args.get('Training', 'OUTPUT_PATH')
    prepareDir(output_path)
    shutil.copy(cfg_file_path, os.path.join(output_path, cfg_file_path.split("/")[-1]))

    mean = 0
    std = 1

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
        #model, _, _, _ = loadCheckpoint(model_path)

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


    # Prepare image generator
    bg_path = "../../autoencoder_ws/data/VOC2012/JPEGImages/"
    dataset_gen = DatasetGenerator(args.get('Dataset', 'BACKGROUND_IMAGES'),
                                   args.get('Dataset', 'CAD_PATH'),
                                   json.loads(args.get('Rendering', 'T'))[-1],
                                   args.getint('Training', 'BATCH_SIZE'),
                                   args.get('Dataset', 'ENCODER_WEIGHTS'),
                                   device,
                                   args.get('Training', 'VIEW_SAMPLING'))


    np.random.seed(seed=args.getint('Training', 'RANDOM_SEED'))
    while(epoch < args.getint('Training', 'NUM_ITER')):
        loss = trainEpoch(mean, std, br, data, model, device, output_path,
                          loss_method=args.get('Training', 'LOSS'),
                          pose_rep=args.get('Training', 'POSE_REPRESENTATION'),
                          t=json.loads(args.get('Rendering', 'T')),
                          num_samples=args.getint('Training', 'NUM_SAMPLES'),
                          visualize=args.getboolean('Training', 'SAVE_IMAGES'),
                          loss_params=args.getfloat('Training', 'LOSS_PARAMS'))
        append2file([loss], os.path.join(output_path, "train-loss.csv"))
        # val_loss = testEpoch(mean, std, br, val_data, model, device, output_path,
        #                      loss_method=args.get('Training', 'LOSS'),
        #                      pose_rep=args.get('Training', 'POSE_REPRESENTATION'),
        #                      t=json.loads(args.get('Rendering', 'T')),
        #                      visualize=args.getboolean('Training', 'SAVE_IMAGES'),
        #                      loss_params=args.getfloat('Training', 'LOSS_PARAMS'))
        val_loss = loss
        append2file([val_loss], os.path.join(output_path, "validation-loss.csv"))
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

def testEpoch(mean, std, br, val_data, model,
               device, output_path, loss_method, pose_rep, t,
               visualize=False, loss_params=0.5):
    global optimizer
    with torch.no_grad():
        dbg("Before test memory: {}".format(torch.cuda.memory_summary(device=device, abbreviated=False)), dbg_memory)

        model.eval()
        losses = []
        batch_size = br.batch_size
        num_samples = len(val_data["codes"])
        data_indeces = np.arange(num_samples)

        for i,curr_batch in enumerate(batch(data_indeces, batch_size)):
            codes = []
            input_images = []
            for b in curr_batch:
                norm_code = val_data["codes"][b] / np.linalg.norm(val_data["codes"][b])
                codes.append(norm_code)
                input_images.append(val_data["images"][b])
            batch_codes = torch.tensor(np.stack(codes), device=device, dtype=torch.float32) # Bx128

            predicted_poses = model(batch_codes)

            # Prepare ground truth poses for the loss function
            T = np.array(t, dtype=np.float32)
            Rs = []
            ts = []
            for b in curr_batch:
                Rs.append(val_data["Rs"][b])
                ts.append(T.copy())

            if('-random-multiview' in loss_method):
                loss, batch_loss, gt_images, predicted_images = Loss(predicted_poses, Rs, br, ts,
                                                                     mean, std, loss_method=loss_method, pose_rep=pose_rep, views=[views[0]], loss_params=loss_params)
            else:
                loss, batch_loss, gt_images, predicted_images = Loss(predicted_poses, Rs, br, ts,
                                                                     mean, std, loss_method=loss_method, pose_rep=pose_rep, views=views, loss_params=loss_params)

            #detach all from gpu
            batch_codes.detach().cpu().numpy()
            loss.detach().cpu().numpy()
            gt_images.detach().cpu().numpy()
            predicted_images.detach().cpu().numpy()

            print("Test batch: {0}/{1} (size: {2}) - loss: {3}".format(i+1,round(num_samples/batch_size), len(Rs),torch.mean(batch_loss)))
            losses = losses + batch_loss.data.detach().cpu().numpy().tolist()

            if(visualize):
                batch_img_dir = os.path.join(output_path, "val-images/epoch{0}".format(epoch))
                prepareDir(batch_img_dir)
                gt_img = (gt_images[0]).detach().cpu().numpy()
                predicted_img = (predicted_images[0]).detach().cpu().numpy()

                vmin = np.linalg.norm(T)*0.9
                vmax = max(np.max(gt_img), np.max(predicted_img))


                fig = plt.figure(figsize=(12,3+len(views)*2))
                #for viewNum in np.arange(len(views)):
                plotView(0, len(views), vmin, vmax, input_images, gt_images, predicted_images,
                         predicted_poses, batch_loss, batch_size)
                fig.tight_layout()

                #plt.hist(gt_img,bins=20)
                fig.savefig(os.path.join(batch_img_dir, "epoch{0}-batch{1}.png".format(epoch,i)), dpi=fig.dpi)
                plt.close()

                # fig = plt.figure(figsize=(4,4))
                # plt.imshow(data["images"][curr_batch[0]])
                # fig.savefig(os.path.join(batch_img_dir, "epoch{0}-batch{1}-gt.png".format(epoch,i)), dpi=fig.dpi)
                # plt.close()

        dbg("After test memory: {}".format(torch.cuda.memory_summary(device=device, abbreviated=False)), dbg_memory)
        return np.mean(losses)

def trainEpoch(mean, std, br, data, model,
               device, output_path, loss_method, pose_rep, t,
               num_samples, visualize=False, loss_params=0.5):
    global optimizer, lr_reducer
    dbg("Before train memory: {}".format(torch.cuda.memory_summary(device=device, abbreviated=False)), dbg_memory)

    # Generate training data
    data = dataset_gen.generate_samples(num_samples)
    print("Generated {0} samples!".format(len(data["codes"])))

    model.train()
    losses = []
    batch_size = br.batch_size
    data_indeces = np.arange(num_samples)

    print("Epoch: {0} - current learning rate: {1}".format(epoch, lr_reducer.get_last_lr()))

    np.random.shuffle(data_indeces)
    for i,curr_batch in enumerate(batch(data_indeces, batch_size)):
        optimizer.zero_grad()
        codes = []
        input_images = []
        for b in curr_batch:
            codes.append(data["codes"][b])
            input_images.append(data["images"][b])
        batch_codes = torch.tensor(np.stack(codes), device=device, dtype=torch.float32) # Bx128

        predicted_poses = model(batch_codes)

        # Prepare ground truth poses for the loss function
        T = np.array(t, dtype=np.float32)
        Rs = []
        ts = []
        for b in curr_batch:
            Rs.append(data["Rs"][b])
            ts.append(T.copy())

        loss, batch_loss, gt_images, predicted_images = Loss(predicted_poses, Rs, br, ts,
                                                             mean, std, loss_method=loss_method, pose_rep=pose_rep, views=views, loss_params=loss_params)

        points = get_points(br.obj_path, device)

        Rs = torch.tensor(np.stack(Rs), device=device, dtype=torch.float32)

        loss, batch_loss = ChamferLoss(Rs, predicted_poses, points, device)

        loss.backward()
        optimizer.step()

        #print("model weights: ", model.l3.weight[0][:5])
        #print("encoder weights: ", dataset_gen.encoder.autoencoder_dense_MatMul.weight[:5])

        #detach all from gpu
        batch_codes.detach().cpu().numpy()
        loss.detach().cpu().numpy()
        gt_images.detach().cpu().numpy()
        predicted_images.detach().cpu().numpy()

        print("Batch: {0}/{1} (size: {2}) - loss: {3}".format(i+1,round(num_samples/batch_size), len(Rs),torch.mean(batch_loss)))
        losses = losses + batch_loss.data.detach().cpu().numpy().tolist()

        if(visualize):
            batch_img_dir = os.path.join(output_path, "images/epoch{0}".format(epoch))
            prepareDir(batch_img_dir)
            gt_img = (gt_images[0]).detach().cpu().numpy()
            predicted_img = (predicted_images[0]).detach().cpu().numpy()
            input_img = input_images[0]*255

            vmin = np.linalg.norm(T)*0.9
            vmax = max(np.max(gt_img), np.max(predicted_img))


            fig = plt.figure(figsize=(12,3+len(views)*2))
            for viewNum in np.arange(len(views)):
                plotView(viewNum, len(views), vmin, vmax, input_images, gt_images, predicted_images,
                         predicted_poses, batch_loss, batch_size)
            fig.tight_layout()

            #plt.hist(gt_img,bins=20)
            fig.savefig(os.path.join(batch_img_dir, "epoch{0}-batch{1}.png".format(epoch,i)), dpi=fig.dpi)
            plt.close()

            # fig = plt.figure(figsize=(4,4))
            # plt.imshow(data["images"][curr_batch[0]])
            # fig.savefig(os.path.join(batch_img_dir, "epoch{0}-batch{1}-gt.png".format(epoch,i)), dpi=fig.dpi)
            # plt.close()

    model_dir = os.path.join(output_path, "models/")
    prepareDir(model_dir)
    state = {'model': model.state_dict(),
             'optimizer': optimizer.state_dict(),
             'lr_reducer': lr_reducer.state_dict(),
             'epoch': epoch}
    torch.save(state, os.path.join(model_dir,"model-epoch{0}.pt".format(epoch)))
    dbg("After train memory: {}".format(torch.cuda.memory_summary(device=device, abbreviated=False)), dbg_memory)
    lr_reducer.step()
    gc.collect()
    return np.mean(losses)

if __name__ == '__main__':
    main()
