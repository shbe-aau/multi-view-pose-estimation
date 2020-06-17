# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

"""Visualizes object models in the ground-truth poses."""

import os
import numpy as np

from bop_toolkit_lib import config
from bop_toolkit_lib import dataset_params
from bop_toolkit_lib import inout
from bop_toolkit_lib import misc

import cv2

# PARAMETERS.
################################################################################
p = {
  # See dataset_params.py for options.
  'dataset': 'tless',

  # Dataset split. Options: 'train', 'val', 'test'.
  'dataset_split': 'test',

  # Dataset split type. None = default. See dataset_params.py for options.
  'dataset_split_type': None,

  # File with a list of estimation targets used to determine the set of images
  # for which the GT poses will be visualized. The file is assumed to be stored
  # in the dataset folder. None = all images.
  # 'targets_filename': 'test_targets_bop19.json',
  'targets_filename': None,

  # Select ID's of scenes, images and GT poses to be processed.
  # Empty list [] means that all ID's will be used.
  'scene_ids': [13],
  'im_ids': [],
  'gt_ids': [],
  'obj_ids': [19],
  
  # Folder containing the BOP datasets.
  'datasets_path': config.datasets_path,
}
################################################################################


# Load dataset parameters.
dp_split = dataset_params.get_split_params(
  p['datasets_path'], p['dataset'], p['dataset_split'], p['dataset_split_type'])

model_type = 'eval'  # None = default.
dp_model = dataset_params.get_model_params(
  p['datasets_path'], p['dataset'], model_type)

# Subset of images for which the ground-truth poses will be rendered.
if p['targets_filename'] is not None:
  targets = inout.load_json(
    os.path.join(dp_split['base_path'], p['targets_filename']))
  scene_im_ids = {}
  for target in targets:
    scene_im_ids.setdefault(
      target['scene_id'], set()).add(target['im_id'])
else:
  scene_im_ids = None

# List of considered scenes.
scene_ids_curr = dp_split['scene_ids']
if p['scene_ids']:
  scene_ids_curr = set(scene_ids_curr).intersection(p['scene_ids'])

for scene_id in scene_ids_curr:
  # Load scene info and ground-truth poses.
  #scene_camera = inout.load_scene_camera(
  #  dp_split['scene_camera_tpath'].format(scene_id=scene_id))
  scene_gt = inout.load_scene_gt( 
    dp_split['scene_gt_tpath'].format(scene_id=scene_id))
  scene_gt_info = inout.load_scene_gt(
    dp_split['scene_gt_info_tpath'].format(scene_id=scene_id))

  # List of considered images.
  if scene_im_ids is not None:
    im_ids = scene_im_ids[scene_id]
  else:
    im_ids = sorted(scene_gt.keys())
  if p['im_ids']:
    im_ids = set(im_ids).intersection(p['im_ids'])

  # Loops through the images
  for im_counter, im_id in enumerate(im_ids):
   
    # List of considered ground-truth poses.
    gt_ids_curr = range(len(scene_gt[im_id]))
    if p['gt_ids']:
      gt_ids_curr = set(gt_ids_curr).intersection(p['gt_ids'])

    # Collect the ground-truth poses.
    gt_poses = []
    for gt_id in gt_ids_curr:    
      gt = scene_gt[im_id][gt_id]
      gt_info = scene_gt_info[im_id][gt_id]

      # Load info
      obj_id = gt['obj_id']
      R = gt['cam_R_m2c']
      t = gt['cam_t_m2c']
      bbox = gt_info['bbox_obj']
      #bbox = gt_info['bbox_visib']
      visible = gt_info['visib_fract']
      print(visible)

      if(obj_id not in p['obj_ids']):
        continue

      # Load the color image
      rgb = inout.load_im(dp_split['rgb_tpath'].format(scene_id=scene_id, im_id=im_id))[:, :, :3]

      # Draw bbox and display
      rgb = cv2.rectangle(rgb, (bbox[0],bbox[1]), (bbox[0]+bbox[2],bbox[1]+bbox[3]), (255,0,0), 2)
      cv2.imshow("bla", np.flip(rgb,axis=2))
      cv2.waitKey(0)
