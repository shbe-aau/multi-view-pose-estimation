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

FROM nvidia/cudagl:10.0-base-ubuntu${UBUNTU_VERSION} as base

# Pick up some TF dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cuda-command-line-tools-10-0 \
        cuda-cublas-10-0 \
        cuda-cufft-10-0 \
        cuda-curand-10-0 \
        cuda-cusolver-10-0 \
        cuda-cusparse-10-0 \
        libcudnn7=7.4.1.5-1+cuda10.0 \
        libfreetype6-dev \
        libhdf5-serial-dev \
        libzmq3-dev \
        pkg-config \
        software-properties-common \
        unzip

RUN apt-get update && \
        apt-get install nvinfer-runtime-trt-repo-ubuntu1804-5.0.2-ga-cuda10.0 \
        && apt-get update \
        && apt-get install -y --no-install-recommends libnvinfer5=5.0.2-1+cuda10.0 \
        && apt-get clean \
        && rm -rf /var/lib/apt/lists/*

# For CUDA profiling, TensorFlow requires CUPTI.
ENV LD_LIBRARY_PATH /usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH

# See http://bugs.python.org/issue19846
ENV LANG C.UTF-8

ENV TORCH_CUDA_ARCH_LIST=7.0

# Required for Python3.6
RUN apt-get update && apt-get -q install -y \
    python3-pip \
    python3 \
    git

RUN pip3 --no-cache-dir install --upgrade \
    pip

# Install dependicies for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
        libsm6 \
	libxext6 \
	libxrender-dev

# Install OpenCV dependencies
RUN apt-get update
RUN apt-get install -y build-essential apt-utils

RUN DEBIAN_FRONTEND="noninteractive" apt-get -y install tzdata
RUN apt-get install -y cmake git libgtk2.0-dev pkg-config libavcodec-dev \
  libavformat-dev libswscale-dev
RUN  apt-get update && apt-get install -y libtbb2 libtbb-dev \
  libjpeg-dev libdc1394-22-dev \
  libatlas-base-dev gfortran webp qt5-default libvtk6-dev zlib1g-dev
RUN pip3 install numpy

# Install OpenCV from repo
RUN cd ~/ &&\
    git clone https://github.com/Itseez/opencv.git &&\
    git clone https://github.com/Itseez/opencv_contrib.git &&\
    cd opencv && mkdir build && cd build && cmake  -DWITH_QT=ON -DWITH_OPENGL=ON -DFORCE_VTK=ON -DWITH_TBB=ON -DWITH_GDAL=ON -DWITH_XINE=ON -DBUILD_EXAMPLES=ON .. && \
    make -j7 && make install && ldconfig
RUN rm -rf ~/opencv && rm -rf ~/opencv_contrib

# Install GLFW for OpenGL and install Assimp
RUN apt-get install -y \
    	libglfw3-dev \
	libglfw3 \
	libassimp-dev

# Pip install stuff
RUN pip3 --no-cache-dir install --pre --upgrade PyOpenGL PyOpenGL_accelerate
RUN pip3 --no-cache-dir install cython

RUN pip3 --no-cache-dir install \
    cyglfw3 \
    imageio==2.6.0 \
    pyassimp==3.3 \
    imgaug \
    progressbar \
    configparser \
    opencv-python==3.4.0.14 \
    opencv-contrib-python

# Install Tensorflow
RUN pip3 install tensorflow-gpu==1.14.0

# Install Keras and RetinaNet
#RUN cd ~/ &&\
#    git clone https://github.com/fizyr/keras-retinanet
#RUN cd keras-retinanet && pip install .
#RUN rm -rf ~/keras-retinanet
RUN pip3 --no-cache-dir install 'git+https://github.com/fizyr/keras-retinanet'

# Install torch and torch vision
RUN pip3 --no-cache-dir install \
    torch \
    torchvision

# Install nice to have stuff
RUN pip3 --no-cache-dir install \
    imageio \
    matplotlib \
    scikit-image

# Install nice to have stuff
RUN pip3 --no-cache-dir install \
    pyglet \
    vispy \
    ruamel.yaml


# Install AAE (AutoPose) - forked version
#RUN pip3 --no-cache-dir install 'git+https://github.com/shbe-aau/AugmentedAutoencoder'
#ENV AE_WORKSPACE_PATH=/shared-folder/autoencoder_ws

# Install pytorch3d - forked version
# NEED TO BE DONE MANUALLY IN THE DOCKER CONTAINER...
#RUN pip3 --no-cache-dir install 'git+https://github.com/shbe-aau/pytorch3d.git'
