#!/usr/local/bin/python3
# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

# A script to render 3D object models into the training images. The models are
# rendered at the 6D poses that are associated with the training images.
# The visualizations are saved into the folder specified by "output_dir".

from pytless import inout, renderer, misc
import os
import numpy as np
import imageio

model_path = "./t-less_v2/models_cad/obj_19.ply"

# Get intrinsic camera parameters and object pose
K = np.array([1075.65091572, 0.0, 396.06888344, 0.0, 1073.90347929, 235.72159802, 0.0, 0.0, 1.0]).reshape(3,3)

R = np.array([-0.04952228, 0.64490459, 0.76265755, 0.33259176, -0.70936509, 0.62143685, 0.94176932, 0.28442848, -0.17936052]).reshape(3,3)
t = np.array([-69.35163193, 101.13001501, 737.73502139])
im_size = (720, 540)

model = inout.load_ply(model_path)
surf_color = (1, 1, 1)
ren_rgb = renderer.render(model, im_size, K, R, t,
                          surf_color=surf_color, mode='rgb')

imageio.imwrite("opengl-render.png", ren_rgb.astype(np.uint8))
