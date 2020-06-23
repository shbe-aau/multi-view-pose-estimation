# -*- coding: utf-8 -*-
import os
import configparser
import argparse
import numpy as np
import signal
import progressbar
import tensorflow as tf
import pickle

import cv2

from . import ae_factory as factory
from . import utils as u

visualize = False

def main():
    workspace_path = os.environ.get('AE_WORKSPACE_PATH')

    if workspace_path == None:
        print('Please define a workspace path:\n')
        print('export AE_WORKSPACE_PATH=/path/to/workspace\n')
        exit(-1)

    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_name")
    parser.add_argument("pickle_path")
    parser.add_argument('--at_step', default=None, required=False)
    arguments = parser.parse_args()


    full_name = arguments.experiment_name.split('/')

    experiment_name = full_name.pop()
    experiment_group = full_name.pop() if len(full_name) > 0 else ''

    codebook,dataset = factory.build_codebook_from_name(experiment_name,experiment_group,return_dataset=True)

    workspace_path = os.environ.get('AE_WORKSPACE_PATH')
    log_dir = u.get_log_dir(workspace_path,experiment_name,experiment_group)
    ckpt_dir = u.get_checkpoint_dir(log_dir)

    train_cfg_file_path = u.get_train_config_exp_file_path(log_dir, experiment_name)
    train_args = configparser.ConfigParser()
    train_args.read(train_cfg_file_path)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
    config = tf.ConfigProto(gpu_options=gpu_options)

    data = pickle.load(open(arguments.pickle_path,"rb"))
    Rs_predicted = []
    images = []

    with tf.Session(config=config) as sess:

        print(ckpt_dir)
        print('#'*20)

        factory.restore_checkpoint(sess, tf.train.Saver(), ckpt_dir)

        for i,img in enumerate(data["images"]):
            if(visualize):
                n = 10
                R = codebook.nearest_rotation(sess, img, top_n=n)
                pred_view = None
                for k in np.arange(n):
                    curr_view = dataset.render_rot(R[k],downSample = 1)
                    if(pred_view is None):
                        pred_view = curr_view
                    else:
                        pred_view = np.concatenate((pred_view, curr_view), axis=1)
                    print(R[k])
                cv2.imshow('pred view rendered', pred_view)
                cv2.imshow('resized webcam input', img)
                k = cv2.waitKey(0)
                if k == 27:
                    break
            else:
                R = codebook.nearest_rotation(sess, img, top_n=1)
                Rs_predicted.append(np.array(R))
                curr_view = dataset.render_rot(R,downSample = 1)
                images.append(curr_view)
                #print(Rs_predicted[-1])
                print(i)

        coded_data = {"images":data["images"],
                      "codebook_images":images,
                      "Rs":data["Rs"],
                      "Rs_predicted":Rs_predicted,
                      "ts":data["ts"]}
        pickle_path_out = (arguments.pickle_path).replace("-images", "-poses")
        print("Saving to: {0}".format(pickle_path_out))
        pickle.dump(coded_data, open(pickle_path_out, "wb"), protocol=2)

        sess.close()

if __name__ == '__main__':
    main()
