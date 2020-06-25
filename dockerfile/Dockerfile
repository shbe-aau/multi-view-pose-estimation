
ARG UBUNTU_VERSION=18.04

FROM nvidia/cuda:10.0-base-ubuntu${UBUNTU_VERSION} as base

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
        unzip \
	git \
	emacs

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

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip

RUN pip3 --no-cache-dir install --upgrade \
    pip \
    setuptools

# Some TF tools expect a "python" binary
RUN ln -s $(which python3) /usr/local/bin/python

# Install Tensorflow
RUN pip3 install tensorflow-gpu==1.14.0

# Install AAE requirements
RUN apt-get install -y \
    	libglfw3-dev \
	libglfw3 \
	libassimp-dev
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
RUN pip3 install tensorflow-gpu==1.14.0

# Install AAE (AutoPose) - forked version
RUN pip3 --no-cache-dir install 'git+https://github.com/shbe-aau/AugmentedAutoencoder'
ENV AE_WORKSPACE_PATH=/shared-folder/autoencoder_ws

# Install pytorch3d requirements
RUN pip3 --no-cache-dir install \
    imageio \
    matplotlib \
    scikit-image

# Install pytorch3d - forked version
# NEED TO BE DONE MANUALLY IN THE DOCKER CONTAINER...
#RUN pip3 --no-cache-dir install 'git+https://github.com/shbe-aau/pytorch3d.git'

# Install BOP toolkit requirements
RUN pip3 --no-cache-dir install \
    pyopengl \
    pyglet \
    vispy \
    ruamel.yaml
