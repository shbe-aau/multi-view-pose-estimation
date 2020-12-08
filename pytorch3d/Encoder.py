import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__weights_dict = dict()

def load_weights(weight_file):
    if weight_file == None:
        return

    try:
        weights_dict = np.load(weight_file, allow_pickle=True).item()
    except:
        weights_dict = np.load(weight_file, encoding='bytes', allow_pickle=True).item()

    return weights_dict

class Encoder(nn.Module):


    def __init__(self, weight_file):
        super(Encoder, self).__init__()
        global __weights_dict
        __weights_dict = load_weights(weight_file)

        base_name = (list(__weights_dict.keys())[0].split('/'))[0]

        self.autoencoder_conv2d_Conv2D = self.__conv(2, name=base_name+'/conv2d/Conv2D', in_channels=3, out_channels=128, kernel_size=(5, 5), stride=(2, 2), groups=1, bias=True)
        self.autoencoder_conv2d_1_Conv2D = self.__conv(2, name=base_name+'/conv2d_1/Conv2D', in_channels=128, out_channels=256, kernel_size=(5, 5), stride=(2, 2), groups=1, bias=True)
        self.autoencoder_conv2d_2_Conv2D = self.__conv(2, name=base_name+'/conv2d_2/Conv2D', in_channels=256, out_channels=512, kernel_size=(5, 5), stride=(2, 2), groups=1, bias=True)
        self.autoencoder_conv2d_3_Conv2D = self.__conv(2, name=base_name+'/conv2d_3/Conv2D', in_channels=512, out_channels=512, kernel_size=(5, 5), stride=(2, 2), groups=1, bias=True)
        self.autoencoder_dense_MatMul = self.__dense(name=base_name+'/dense/MatMul', in_features = 32768, out_features = 128, bias = True)

    def forward(self, x):
        # Normalize images
        #img_mean = torch.mean(torch.flatten(x, start_dim=2), 2).unsqueeze(-1).unsqueeze(-1)
        #img_std = torch.std(torch.flatten(x, start_dim=2), 2).unsqueeze(-1).unsqueeze(-1)
        #x = (x - img_mean) / img_std

        # img_max, _ = torch.max(torch.flatten(x, start_dim=2), 2)
        # img_max = img_max.unsqueeze(-1).unsqueeze(-1)
        # img_min, _ = torch.min(torch.flatten(x, start_dim=2), 2)
        # img_min = img_min.unsqueeze(-1).unsqueeze(-1)
        # x = (x - img_min) / (img_max - img_min)
        
        autoencoder_Flatten_flatten_Reshape_shape_1 = torch.tensor(-1, dtype=torch.int32)
        autoencoder_conv2d_Conv2D_pad = F.pad(x, (1, 2, 1, 2))
        autoencoder_conv2d_Conv2D = self.autoencoder_conv2d_Conv2D(autoencoder_conv2d_Conv2D_pad)
        autoencoder_conv2d_Relu = F.relu(autoencoder_conv2d_Conv2D)
        autoencoder_conv2d_1_Conv2D_pad = F.pad(autoencoder_conv2d_Relu, (1, 2, 1, 2))
        autoencoder_conv2d_1_Conv2D = self.autoencoder_conv2d_1_Conv2D(autoencoder_conv2d_1_Conv2D_pad)
        autoencoder_conv2d_1_Relu = F.relu(autoencoder_conv2d_1_Conv2D)
        autoencoder_conv2d_2_Conv2D_pad = F.pad(autoencoder_conv2d_1_Relu, (1, 2, 1, 2))
        autoencoder_conv2d_2_Conv2D = self.autoencoder_conv2d_2_Conv2D(autoencoder_conv2d_2_Conv2D_pad)
        autoencoder_conv2d_2_Relu = F.relu(autoencoder_conv2d_2_Conv2D)
        autoencoder_conv2d_3_Conv2D_pad = F.pad(autoencoder_conv2d_2_Relu, (1, 2, 1, 2))
        autoencoder_conv2d_3_Conv2D = self.autoencoder_conv2d_3_Conv2D(autoencoder_conv2d_3_Conv2D_pad)
        autoencoder_conv2d_3_Relu = F.relu(autoencoder_conv2d_3_Conv2D)
        autoencoder_Flatten_flatten_Shape = list(autoencoder_conv2d_3_Relu.size())

        # Stefan fix - thanks to Tom:
        # https://discuss.pytorch.org/t/pytorch-convolution-and-tensorflow-convolution-giving-different-results/26863
        intermediate = autoencoder_conv2d_3_Relu.permute(0,2,3,1)
        autoencoder_Flatten_flatten_Reshape = torch.reshape(input = intermediate, shape = (-1,32768))
        #autoencoder_Flatten_flatten_Reshape = torch.reshape(input = autoencoder_conv2d_3_Relu, shape = (-1,32768))

        autoencoder_Flatten_flatten_strided_slice = autoencoder_Flatten_flatten_Shape[0:1]
        autoencoder_dense_MatMul = self.autoencoder_dense_MatMul(autoencoder_Flatten_flatten_Reshape)
        autoencoder_Flatten_flatten_Reshape_shape = [autoencoder_Flatten_flatten_strided_slice,autoencoder_Flatten_flatten_Reshape_shape_1]
        return autoencoder_dense_MatMul


    @staticmethod
    def __dense(name, **kwargs):
        layer = nn.Linear(**kwargs)
        layer.state_dict()['weight'].copy_(torch.from_numpy(__weights_dict[name]['weights']))
        if 'bias' in __weights_dict[name]:
            layer.state_dict()['bias'].copy_(torch.from_numpy(__weights_dict[name]['bias']))
        return layer

    @staticmethod
    def __conv(dim, name, **kwargs):
        if   dim == 1:  layer = nn.Conv1d(**kwargs)
        elif dim == 2:  layer = nn.Conv2d(**kwargs)
        elif dim == 3:  layer = nn.Conv3d(**kwargs)
        else:           raise NotImplementedError()

        layer.state_dict()['weight'].copy_(torch.from_numpy(__weights_dict[name]['weights']))
        if 'bias' in __weights_dict[name]:
            layer.state_dict()['bias'].copy_(torch.from_numpy(__weights_dict[name]['bias']))
        return layer
