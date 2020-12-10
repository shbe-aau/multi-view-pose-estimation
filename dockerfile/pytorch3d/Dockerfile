# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

## Original source:
## https://github.com/tensorflow/tensorflow/blob/v1.13.1/tensorflow/tools/dockerfiles/dockerfiles/gpu.Dockerfile
## Modified by shbe@create.aau.dk

ARG UBUNTU_VERSION=18.04

FROM nvidia/cudagl:10.0-devel-ubuntu${UBUNTU_VERSION} as base

# Required for Python3.6
RUN apt-get update && apt-get -q install -y \
    python3-pip \
    python3 \
    git \
    libfontconfig1-dev

# Upgrade pip3
RUN pip3 --no-cache-dir install --upgrade \
    pip

# Install OpenCV dependencies
RUN apt-get install -y \
    	libglfw3-dev \
	libglfw3 \
	libassimp-dev

# Install cython and numpy
RUN pip3 --no-cache-dir install \
    cython \
    numpy

# Install various libraries
RUN pip3 --no-cache-dir install \
    cyglfw3 \
    imageio==2.6.0 \
    pyassimp==3.3 \
    imgaug \
    progressbar \
    configparser \
    imageio \
    matplotlib \
    scikit-image \
    pypng \
    pytz \
    glumpy

# Install pytless rendering requirements
RUN pip3 --no-cache-dir install \
    pyopengl \
    pyglet \
    vispy \
    ruamel.yaml

# Install pytorch (pytorch3d depends on it...)
RUN pip3 --no-cache-dir install \
    torch==1.4.0 \
    torchvision==0.5.0

# Install OpenCV
RUN pip3 --no-cache-dir --default-timeout=100 install \
    opencv-python==3.4.0.14 \
    opencv-contrib-python

# Install pytorch3d (forked)
ENV FORCE_CUDA=1
RUN pip3 --no-cache-dir install 'git+https://github.com/shbe-aau/pytorch3d.git'
