import torch
import torch.nn as nn
import torch.nn.functional as F

from .unet_model import *
from .mlp import MLP
from .coordconv import AddCoords

import torch.nn.functional as F
# when using jupyter notebook, for some reason relative import doesnt work...
# from unet_model import *
# from robot_encoder import MLP

# adapted from https://github.com/milesial/Pytorch-UNet
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

class UNetFusion(nn.Module):
    ''' pose and wrench must be flattened tensors '''
    def __init__(self, 
    context_frame_dict, 
    cropping_dict,
    im_shape, 
    n_channels, 
    pose_dim, 
    pose_enc_dim, 
    wrench_dim, 
    wrench_enc_dim, 
    pose_hidden_list=None, 
    wrench_hidden_list=None, 
    fuse_pose=False, 
    fuse_ft=False, 
    fuse_inputs=False,
    output_channels=1, 
    feature_dim=64, 
    bilinear=False):
        super(UNetFusion, self).__init__()
        self.context_frame_dict = context_frame_dict
        self.cropping_dict = cropping_dict

        self.rank = 2
        self.with_r = True
        if cropping_dict['enable']:
            self.addcoords = AddCoords(self.rank, bb_h=cropping_dict['bb_height'], bb_w=cropping_dict['bb_width'], im_h=im_shape[0], im_w=im_shape[1], with_r = self.with_r) #rank =2(x,y) , True for using R
        
        self.pose_dim = pose_dim
        self.pose_enc_dim = pose_enc_dim
        self.pose_hidden_list = pose_hidden_list
        if pose_hidden_list is None:
            self.pose_hidden_list = [pose_dim, pose_dim]
        self.wrench_dim = wrench_dim
        self.wrench_enc_dim = wrench_enc_dim
        self.wrench_hidden_list = wrench_hidden_list
        if wrench_hidden_list is None:
            self.wrench_hidden_list = [wrench_dim, wrench_dim]

        self.n_channels = n_channels
        self.bilinear = bilinear
        self.feature_dim = feature_dim
        self.output_channels = output_channels

        self.im_shape = im_shape

        self.fuse_pose = fuse_pose
        self.fuse_ft = fuse_ft
        self.fuse_inputs = fuse_inputs

        self.inc = DoubleConv(n_channels, self.feature_dim)
        self.down1 = Down(self.feature_dim, self.feature_dim*2)
        self.down2 = Down(self.feature_dim*2, self.feature_dim*4)
        self.down3 = Down(self.feature_dim*4, self.feature_dim*8)
        factor = 2 if bilinear else 1

        self.encoding_dim = 0
        if fuse_pose:
            self.encoding_dim += pose_enc_dim 
        if fuse_ft:
            self.encoding_dim += wrench_enc_dim
        if self.context_frame_dict['enable']:
            context_frame_channel = 1
            coord_channel = 3
            if self.cropping_dict['enable']:
                self.inc_first = DoubleConv(context_frame_channel + coord_channel, self.feature_dim)
            else:
                self.inc_first = DoubleConv(context_frame_channel, self.feature_dim)
            self.down1_first = Down(self.feature_dim, self.feature_dim*2)
            self.down2_first = Down(self.feature_dim*2, self.feature_dim*4)
            self.down3_first = Down(self.feature_dim*4, self.feature_dim*8)
            # After concatenating two features
            self.downfuse = DownFuse(self.feature_dim*8*2, (self.feature_dim*16) // factor, self.encoding_dim)
        else:
            # Original model
            self.downfuse = DownFuse(self.feature_dim*8, (self.feature_dim*16) // factor, self.encoding_dim)

        self.up1 = Up(self.feature_dim*16, (self.feature_dim*8) // factor, bilinear)
        self.up2 = Up(self.feature_dim*8, (self.feature_dim*4) // factor, bilinear)
        self.up3 = Up(self.feature_dim*4, (self.feature_dim*2) // factor, bilinear)
        self.up4 = Up(self.feature_dim*2, self.feature_dim, bilinear)
        self.outc = OutConv(self.feature_dim, self.output_channels)

        if fuse_pose:
            self.pose_MLP = MLP(pose_dim, pose_hidden_list, pose_enc_dim)
        if fuse_ft:
            self.wrench_MLP = MLP(wrench_dim, wrench_hidden_list, wrench_enc_dim)
    
    def get_init_params_dict(self):
        return {
            'im_shape': self.im_shape,
            'n_channels': self.n_channels,
            'pose_dim': self.pose_dim,
            'pose_enc_dim': self.pose_enc_dim,
            'pose_hidden_list': self.pose_hidden_list,
            'wrench_dim': self.wrench_dim,
            'wrench_enc_dim': self.wrench_enc_dim,
            'wrench_hidden_list': self.wrench_hidden_list,
            'fuse_pose': self.fuse_pose,
            'fuse_ft': self.fuse_ft,
            'output_channels': self.output_channels,
            'feature_dim': self.feature_dim,
            'bilinear': self.bilinear
        }

    def forward(self, image, pose=None, wrench=None, bb_top_left_coordinate=None, context_frame=None, context_bb_top_left_coordinate=None):
        # assumes poses and wrenches are flattened along the window dimension
        # print(pose.get_device())
        # print(wrench.get_device())
        
        if self.fuse_pose:
            pose_enc = self.pose_MLP(pose)
        if self.fuse_ft:
            wrench_enc = self.wrench_MLP(wrench)

        # concatenate along feature dim
        # assuming the feature dimension is after the batch...
        if self.fuse_pose and self.fuse_ft:
            encoding = torch.cat([pose_enc, wrench_enc], dim=1) 
        elif self.fuse_pose:
            encoding = pose_enc 
        elif self.fuse_ft:
            encoding = wrench_enc 
        else:
            encoding = None
        
        # Add coordniate convolution
        if self.cropping_dict['enable']:
            image = self.addcoords.forward(image, bb_top_left_coordinate) # bz, 3, height, width
        x1 = self.inc(image)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        if self.context_frame_dict['enable']:
            if self.cropping_dict['enable']:
                context_frame = self.addcoords.forward(context_frame, context_bb_top_left_coordinate) # bz, 3, height, width
            f1 = self.inc_first(context_frame)
            f2 = self.down1_first(f1)
            f3 = self.down2_first(f2)
            f4 = self.down3_first(f3)
            fuse_frames = torch.cat([x4, f4], dim=1) #concatenate along channel dimension

            x5 = self.downfuse(fuse_frames, encoding) # [16, 1024, 2, 3]
        else:
            x5 = self.downfuse(x4, encoding)

        image = self.up1(x5, x4)
        image = self.up2(image, x3)
        image = self.up3(image, x2)
        image = self.up4(image, x1)
        logits = self.outc(image)

        return logits
    
    def post_processing(self, cropped_image, EE_poses_pxl_coord, bb_width, bb_height, down_scale = 10):
        padded_img = torch.zeros((cropped_image.shape[0], 1, self.im_shape[0], self.im_shape[1]))

        EE_poses_pxl = torch.zeros(EE_poses_pxl_coord.shape)
        EE_poses_pxl[:, 0] = EE_poses_pxl_coord[:, 0] - int(-bb_width/2)   
        EE_poses_pxl[:, 1] = EE_poses_pxl_coord[:, 1] - int(-bb_height/2+down_scale)
        for idx in range(cropped_image.shape[0]):
            x1 = int(EE_poses_pxl[idx][0]-bb_width/2)
            x2 = int(EE_poses_pxl[idx][0]+bb_width/2)
            y1 = int(EE_poses_pxl[idx][1]-bb_height/2)
            y2 = int(EE_poses_pxl[idx][1]+bb_height/2)
            padded_img[idx, 0] = F.pad(input=cropped_image[idx, 0], pad=(x1, self.im_shape[1]-x2, y1, self.im_shape[0]-y2), mode='constant', value=0)
      
        return padded_img.to(device)
    
class DownFuse(nn.Module):
    """Downscaling with maxpool then double conv with fusion with robot encoding"""

    def __init__(self, in_channels, out_channels, enc_dim, kernel=3, pad=1):
        super().__init__()
        self.downsample = nn.MaxPool2d(2)
        self.enc_dim = enc_dim

        self.convrelu = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel, padding=pad, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

        self.fuseconvrelu = nn.Sequential(
            nn.Conv2d(out_channels+enc_dim, out_channels, kernel_size=kernel, padding=pad, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x, encoding):
        x = self.downsample(x) # BxCxHxW
        x = self.convrelu(x)
        # print(x.shape)
        bneck_dim = (x.shape[-2], x.shape[-1])

        if self.enc_dim == 0: # if no fuse at all
            fuse_input = x
        else: # if pose or ft is fused
            tiled_encoding = encoding[:, :, None, None].repeat(1, 1, bneck_dim[0], bneck_dim[1])
            fuse_input = torch.cat([x, tiled_encoding], dim=1) #concatenate along channel dimension

        x = self.fuseconvrelu(fuse_input) 
        return x
        
