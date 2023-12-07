from math import floor
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

# adapted from https://debuggercafe.com/spatial-transformer-network-using-pytorch/

# self.conv_kerns = [7, 5, 5, 5]
# self.pool_kerns = [2, 2, 2, 2]
# self.pool_strides = [2, 2, 2, 2]
# self.conv_out_channels = [16, 32, 64, 128]

# num_conv_layers = 4


class STN(nn.Module):
    # im_shape is tuple...
    def __init__(self, im_shape, n_channels, conv_out_channels, conv_kerns, pool_kerns, pool_strides):
        super(STN, self).__init__()
        self.n_channels = n_channels
        self.im_shape = im_shape

        assert len(conv_out_channels) == len(conv_kerns) == len(pool_kerns) == len(pool_strides), 'conv layer params must all be same length!' 

        # spatial transformer localization network
        self.conv_kerns = conv_kerns
        self.pool_kerns = pool_kerns
        self.pool_strides = pool_strides
        self.conv_out_channels = conv_out_channels

        self.num_conv_layers = len(self.conv_kerns)

        self.localization = nn.Sequential()
        conv_in_channel = self.n_channels
        for i in range(self.num_conv_layers):
            self.localization.add_module('convlayer_' + str(i), ConvPoolReLU(conv_in_channel, 
            self.conv_out_channels[i], self.conv_kerns[i], self.pool_kerns[i], self.pool_strides[i]))
            
            conv_in_channel = self.conv_out_channels[i]
        self.out_shape = self.compute_out_im_shape(self.im_shape)

        # tranformation regressor for theta
        self.fc_loc = nn.Sequential(
            nn.Linear(128*self.out_shape[0]*self.out_shape[1], 256),
            nn.ReLU(True),
            nn.Linear(256, 3 * 2)
        )
        # initializing the weights and biases with identity transformations
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([0.3, 0, -0.2, 0, 0.3, -0.1],  # prior on affine tf params
                                                    dtype=torch.float))

    def compute_out_im_shape(self, im_shape):
        shape = np.array(im_shape)
        for i in range(self.num_conv_layers):
            # conv layer
            shape = np.floor((shape - (self.conv_kerns[i] - 1) - 1) + 1)
            # pool layer
            shape = np.floor(((shape - (self.pool_kerns[i] - 1) - 1)/self.pool_strides[i]) + 1)
        return tuple(shape.astype(int))

    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, xs.size(1)*xs.size(2)*xs.size(3)) # multiply by C x H x W dims
        # calculate the transformation parameters theta
        theta = self.fc_loc(xs)
        # resize theta
        theta = theta.view(-1, 2, 3) 
        # grid generator => transformation on parameters theta

        grid = F.affine_grid(theta, x.size())
        # grid sampling => applying the spatial transformations
        x = F.grid_sample(x, grid)
        return theta.detach(), x

    def forward(self, x):
        return self.stn(x)
        

class ConvPoolReLU(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, conv_kern, pool_kern, pool_stride): #
        super().__init__()
        self.conv_pool_relu = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=conv_kern),
            nn.MaxPool2d(pool_kern, stride=pool_stride),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv_pool_relu(x)