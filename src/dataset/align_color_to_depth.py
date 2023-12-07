import os
from contact_dataset_episodic import ContactDatasetEpisodic
from torch.utils.data import DataLoader
import tqdm

import numpy as np
import cv2

import time

import natsort
import glob

import torch
import pickle 

class ColorToDepthAligner():
    def __init__(self, experiment_root_dir_path,
                 align_color, align_flow, align_flow_image, color_depth_are_timesynced=False):
        if not color_depth_are_timesynced:
            print("Color and depth are not time synced. This is not implemented yet.")
            raise NotImplementedError

        self.align_color = align_color
        self.align_flow = align_flow
        self.align_flow_image = align_flow_image
        self.experiment_root_dir_path = experiment_root_dir_path

        # get the episode directory list
        self.episode_dir_list = [os.path.join(self.experiment_root_dir_path, d) for d in os.listdir(self.experiment_root_dir_path) if os.path.isdir(os.path.join(self.experiment_root_dir_path, d))]
        self.episode_dir_list.sort()

        self.color_topic = 'color'
        self.color_channels = 3
        self.color_intrinsic_name = 'color_K.npy'
        self.color_extrinsic_name = 'C_tf_W.npy'

        self.depth_topic = 'depth'
        self.depth_intrinsic_name = 'depth_K.npy'
        self.depth_extrinsic_name = 'D_tf_W.npy'
        self.depth_channels = 1

        self.flow_topic = 'flow'
        self.flow_channels = 2
        self.flow_intrinsic_name = 'color_K.npy'
        self.flow_extrinsic_name = 'C_tf_W.npy'

        self.flow_image_topic = 'flow_image'
        self.flow_image_channels = 3
        self.flow_image_intrinsic_name = 'color_K.npy'
        self.flow_image_extrinsic_name = 'C_tf_W.npy'

    def align_color_to_depth(self, color, depth):
        # color: [H, W, C]
        # depth: [H, W, C]
        assert color.shape[0] == depth.shape[0] and color.shape[1] == depth.shape[1]
        assert color.shape[2] == self.color_channels and depth.shape[2] == self.depth_channels
        if self.align_color:
            # Align color to depth
            color_to_depth = cv2.resize(color, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_NEAREST)
        else:
            color_to_depth = color
        return color_to_depth
    
    def flow_to_depth(self, flow, depth):
        # flow: [H, W, C]
        # depth: [H, W, C]
        assert flow.shape[0] == depth.shape[0] and flow.shape[1] == depth.shape[1]
        assert flow.shape[2] == self.flow_channels and depth.shape[2] == self.depth_channels
        if self.align_flow:
            # Align flow to depth
            flow_to_depth = cv2.resize(flow, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_NEAREST)
        else:
            flow_to_depth = flow
        return flow_to_depth
    
    def flow_image_to_depth(self, flow_image, depth):
        # flow_image: [H, W, C]
        # depth: [H, W, C]
        assert flow_image.shape[0] == depth.shape[0] and flow_image.shape[1] == depth.shape[1]
        assert flow_image.shape[2] == self.flow_image_channels and depth.shape[2] == self.depth_channels
        if self.align_flow_image:
            # Align flow image to depth
            flow_image_to_depth = cv2.resize(flow_image, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_NEAREST)
        else:
            flow_image_to_depth = flow_image
        return flow_image_to_depth

if __name__ == '__main__':
    bag_name = 'teleop_real_480x848_no_clutter_combined'
    dataset_root_dir = '/mnt/hdd/datasetsHDD/contact_estimation/real'

    optical_flow_dict = {'enable': True,
                         'max_flow_norm': 10.0, 
                         'normalize': False,
                         'use_image': False}
    real_path = os.path.join(dataset_root_dir, bag_name)

    real_dset = ContactDatasetEpisodic(real_path, 
        optical_flow_dict = optical_flow_dict, is_real_dataset=True)
    
    val_real_lder = DataLoader(
            real_dset,
            shuffle=False,
            num_workers=4,
            batch_size=1,
            drop_last=False
        )
    # New color
    start = time.time()
    if optical_flow_dict['enable']:
        color_topic = 'flow'
        if optical_flow_dict['use_image']:
            color_channels = 3
            color_topic = 'flow_image'
        else:
            color_channels = 2
            color_topic = 'flow'
    else:
        topic = 'color'
        color_channels = 3
        
    color_write_topic = color_topic + '_aligned_to_depth'
    depth_shape = real_dset.depth_shape
    depth_height = depth_shape[0]
    depth_width = depth_shape[1]

    for batch_idx, return_dict in enumerate(val_real_lder):
        depth_in = return_dict['depth_intrinsics'].numpy()[0]
        depth_in_inv = np.linalg.inv(depth_in)
        depth_ex = return_dict['depth_tf_world'].numpy()[0]
        t = - depth_ex[:3, :3].T @ depth_ex[:3, 3]
        R = depth_ex[:3, :3].T

        color_in = return_dict['color_intrinsics'].numpy()[0]
        color_ex = return_dict['color_tf_world'].numpy()[0]
        t_color = color_ex[:3, 3]
        R_color = color_ex[:3, :3]

        #depth_image = return_dict['images_tensor'].numpy() #Depth
        depth_image = np.array(return_dict['images_tensor']) #1, 1, 480, 848

        # permute to H x W x C
        color_image = torch.permute(return_dict[color_topic][0], (-2, -1, -3)).numpy() # H x W x C
        color_path = return_dict[color_topic + '_paths'][0][0]
        # color_image = cv2.imread(color_path)  # 480, 848, 3

        episode_data_dir = real_dset.experiment_data_dir_path
        s = color_path.split('/')
        new_folder = os.path.join(episode_data_dir, s[-3], color_write_topic)
        if not os.path.exists(new_folder):
            os.mkdir(new_folder)
        new_path = os.path.join(new_folder, s[-1])

        new_color = np.zeros(color_image.shape) 

        depth_image = depth_image.reshape(depth_height, depth_width,1)

        max_depth_clip = real_dset.max_depth_clip
        # get pixel grid

        rows, cols = np.meshgrid(np.arange(depth_height), np.arange(depth_width), indexing='ij')

        # reshape to vectors
        pixel_depth = np.stack((cols.flatten(), rows.flatten(), np.ones((cols.size,))), axis=1)
        pixel_depth = pixel_depth.T

        # transform to world coordinates
        world_coord = R @ ( max_depth_clip * depth_image.flatten()* (depth_in_inv @ pixel_depth) )+ t.reshape(3,1)
        world_coord = world_coord.T

        # transform to color image coordinates
        pixel_color = np.matmul(color_in, np.matmul(R_color, world_coord.T) + t_color.reshape(3,1))
        pixel_color /= pixel_color[-1, :]

        # extract color values from color image
        # get indices of pixels that are within the bounds of the color image
        u_old_color = pixel_color[0]
        v_old_color = pixel_color[1]
        mask = ((u_old_color >= 0) & (u_old_color < color_image.shape[1]) & 
                (v_old_color >= 0) & (v_old_color < color_image.shape[0]))

        # create new color image
        new_color_shape = (depth_height, depth_width, color_channels)
        new_color = np.zeros(new_color_shape)
        
        # index = np.where(mask)[0]

        # for ind in index:
        #     new_color[int(ind/848), ind%848] = color_image[v_old_color[ind].astype(int), u_old_color[ind].astype(int), :]

        index = np.where(mask.reshape(depth_height, depth_width))
        new_color[index[0], index[1], :] = color_image[v_old_color[mask].astype(int), u_old_color[mask].astype(int), :]
        if optical_flow_dict['use_image']:
            new_color = cv2.cvtColor(new_color.astype(np.uint8), cv2.COLOR_RGB2BGR)
            cv2.imwrite(new_path, new_color)
        else:
            # write the flow to pkl
            with open(new_path, 'wb') as f:
                pickle.dump(new_color, f)

    end = time.time()
    print(end-start)
        
