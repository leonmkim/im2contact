import os
import torch
from torch.utils.data import Dataset

import numpy as np
from scipy.spatial.transform import Rotation as R
import pandas as pd

# import bagpy
# from bagpy import bagreader
# import rosbag_pandas.src.rosbag_pandas

import skimage
# import PIL
import torchvision
import cv2
import sys

import glob
import natsort

import math

# import imgaug as ia
# import imgaug.augmenters as iaa
# ci_build_and_not_headless = False
# try:
#     from cv2.version import ci_build, headless
#     ci_and_not_headless = ci_build and not headless
# except:
#     pass
# if sys.platform.startswith("linux") and ci_and_not_headless:
#     os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH")
# if sys.platform.startswith("linux") and ci_and_not_headless:
#     os.environ.pop("QT_QPA_FONTDIR")


import pickle
class FlowDataset(Dataset):
    # if window_size = -1, produce the full episode !
    def __init__(self, experiment_dset,
                 aligned_color_to_depth=False, 
    dyn_cam = False,   
    im_resize=None):
        # self.rosbag = None
        if aligned_color_to_depth:
            self.color_dir_name = 'color_aligned_to_depth'
        else:
            self.color_dir_name = 'color'
            
        self.dyn_cam = dyn_cam
        
        # bag_path = '~/datasets/rosbags/input.bag'
        self.experiment_dset_path = experiment_dset
        self.experiment_dset_name = os.path.basename(os.path.normpath(self.experiment_dset_path))
        self.experiment_dset_path = os.path.expanduser(experiment_dset)
        self.experiment_data_dir_path = os.path.join(self.experiment_dset_path, 'episode_data')
        # get list of episode directories
        if not os.path.exists(self.experiment_dset_path):
            # print non existent bag path
            raise AssertionError(self.experiment_dset_path + ' does not exist!')
        # get list of directory names, excluding those that aren't directories
        episode_data_dir_list = natsort.natsorted(os.listdir(self.experiment_data_dir_path))
        self.episode_data_dir_path_list = [os.path.join(self.experiment_data_dir_path, d) for d in episode_data_dir_list if os.path.isdir(os.path.join(self.experiment_data_dir_path, d))]

        # load the image timestamp and episode multiindex dataframe
        self.depth_episode_timestamp_series = pd.read_pickle(os.path.join(self.experiment_data_dir_path, 'depth_episode_timestamp_series.pkl'))
        self.color_episode_timestamp_series = pd.read_pickle(os.path.join(self.experiment_data_dir_path, 'color_episode_timestamp_series.pkl'))
        self.main_num_msgs = len(self.color_episode_timestamp_series)
        # TODO fix this so that first index gets the first valid history of window size!

        assert self.main_num_msgs > 1, "must be at least one number of msgs!" 
        # adjust the length of the dataset to account for image pairs
        # get number of images per episode
        self.num_img_pairs_per_episode_series = self.color_episode_timestamp_series.groupby(level=0).count() - 1
        self.img_pairs_cumsum_per_episode_series = self.num_img_pairs_per_episode_series.cumsum()
        # get the number of image pairs in the dataset
        self._len = self.num_img_pairs_per_episode_series.sum()

        # check first episode index's image to get the image shape
        first_episode_dir_path = self.episode_data_dir_path_list[0]
        first_episode_first_color_name = os.listdir(os.path.join(first_episode_dir_path, self.color_dir_name))[0]
        self.color_shape = cv2.imread(os.path.join(first_episode_dir_path, self.color_dir_name, first_episode_first_color_name)).shape[:2]

        if im_resize is not None:
            if self.color_shape == im_resize:
                self.im_resize = None
            else:
                self.im_resize = im_resize #HxW
        else:
            self.im_resize = None

        self.im_to_tensor = torchvision.transforms.ToTensor() 
        self.tensor_to_float32 = torchvision.transforms.ConvertImageDtype(torch.float32)

        # print('succesfully loaded bag!')
        print('succesfully read ' + experiment_dset)

    def __len__(self):
        return self._len
    
    def __getitem__(self, idx_accessed):
        if idx_accessed >= self._len:
            raise AssertionError('index out of range')
        # adjust the idx_accessed to account for image pairs
        # get the episode index of the accessed index
        episode_idx = self.img_pairs_cumsum_per_episode_series.index[self.img_pairs_cumsum_per_episode_series > idx_accessed][0]
        adjusted_idx = idx_accessed + (episode_idx) # assuming episode idx starts at 0

        episode_dir_path = self.episode_data_dir_path_list[episode_idx]
        # get the dataframes given an episode index

        # get the color camera params
        color_tf_world = np.load(os.path.join(episode_dir_path, self.color_dir_name, 'C_tf_W.npy'))
        K_color = np.load(os.path.join(episode_dir_path, self.color_dir_name, 'color_K.npy'))

        # get the index of the accessed index in the episode
        idx_in_episode = adjusted_idx - self.color_episode_timestamp_series.index.get_indexer_for([episode_idx])[0]
        prev_idx_in_episode = idx_in_episode
        curr_idx_in_episode = idx_in_episode + 1
        # get color image paths given an episode index
        episode_color_times = self.color_episode_timestamp_series.loc[episode_idx].values 
        episode_color_path_list = natsort.natsorted(glob.glob(os.path.join(episode_dir_path, self.color_dir_name, '*.png')))
        assert prev_idx_in_episode < len(episode_color_times), "prev index out of range"
        assert curr_idx_in_episode < len(episode_color_times), "curr index out of range"
        
        prev_image_filepath = episode_color_path_list[prev_idx_in_episode]
        curr_image_filepath = episode_color_path_list[curr_idx_in_episode]

        # get images
        # prev_image = skimage.io.imread_collection(prev_image_filepath) # H x W 
        # curr_image = skimage.io.imread_collection(curr_image_filepath) # H x W 
        prev_image = cv2.imread(prev_image_filepath, cv2.IMREAD_UNCHANGED) # H x W x C
        prev_image = cv2.cvtColor(prev_image, cv2.COLOR_BGR2RGB)
        curr_image = cv2.imread(curr_image_filepath, cv2.IMREAD_UNCHANGED) # H x W x C
        curr_image = cv2.cvtColor(curr_image, cv2.COLOR_BGR2RGB)
        prev_image_time = episode_color_times[prev_idx_in_episode]
        curr_image_time = episode_color_times[curr_idx_in_episode]

        #resize 
        if self.im_resize is None: # if im resize is none
            prev_image = np.array(prev_image) # now become C x H x W
            curr_image = np.array(curr_image) # now become C x H x W
        else:
            prev_image = np.array([cv2.resize(prev_image, (self.im_resize[1], self.im_resize[0]), interpolation=cv2.INTER_CUBIC)]) #preserves int8 
            curr_image = np.array([cv2.resize(curr_image, (self.im_resize[1], self.im_resize[0]), interpolation=cv2.INTER_CUBIC)]) #preserves int8 
        assert prev_image.dtype == np.uint8, 'color images not np uint8!'
        assert curr_image.dtype == np.uint8, 'color images not np uint8!'

        # permute to be C x H x W
        prev_image = np.moveaxis(prev_image, -1, 0)
        curr_image = np.moveaxis(curr_image, -1, 0)
        
        return_dict = {
        'image_prev': prev_image, #B x C x H x W 
        'image_current': curr_image, #B x C x H x W
        'image_prev_time': prev_image_time, 
        'image_current_time': curr_image_time,
        'image_prev_filepath': prev_image_filepath,
        'image_current_filepath': curr_image_filepath,
        'idx_accessed': idx_accessed,
        'adjusted_idx': adjusted_idx,
        'within_episode_idx': idx_in_episode,
        'episode': episode_idx
        }
        return return_dict

    def invert_transform(self, tf):
        R = tf[0:3, 0:3]
        T = tf[:3, -1]
        tf_inv = np.diag([1.,1.,1.,1.])
        tf_inv[:3, :3] = R.T
        tf_inv[:3, -1] = -R.T @ T
        return tf_inv

    #TODO make this return a list of tfs!
    def get_inv_cam_extrin(self, times, depth=True): # make the assumption we are only dealing with one time...
        # TODO fix the assumption for the time window case
        # use only previous to make sure the extrinsics are not of the next trial where the cam pose has been newly sampled
        nrst_idx = self.get_nearest_idxs(times, self.ep_info_df.index, only_prev=True)[0] #indexing into time dim, assuming len(T)=1
        row = self.ep_info_df.iloc[nrst_idx]
        if depth:
            base_topic = '/episode_info/cam_depth_extrin'
        else:
            base_topic = '/episode_info/cam_color_extrin'

        cam_pos_topic = base_topic + '/position'
        cam_pos_cols = [col for col in row.keys() if cam_pos_topic in col]
        cam_pos = np.array(row[cam_pos_cols].values)

        cam_ori_topic = base_topic + '/orientation'
        cam_ori_cols = [col for col in row.keys() if cam_ori_topic in col]
        cam_ori = np.roll(np.array(row[cam_ori_cols].values), -1) # w,x,y,z to x,y,z,w

        # W_tf_cam = np.diag([0.,0.,0.,1.])
        # W_tf_cam[:3, :3] = (R.from_quat(cam_ori)).as_matrix()
        # W_tf_cam[:3, -1] = cam_pos

        cam_tf_W = np.diag([1.,1.,1.,1.])
        cam_tf_W[:3, :3] = (R.from_quat(cam_ori)).as_matrix().T
        cam_tf_W[:3, -1] = -cam_tf_W[:3, :3] @ cam_pos 
        
        return cam_tf_W
    
    def get_nearest_idxs(self, times, df_index, only_prev=False): #df_index can equivalently be an np array
        # from https://stackoverflow.com/a/26026189
        ## if both elements exactly match, left will give index at the matching value itself and if right, will give index ahead of match
        ## search sorted is like if at that returned idx location, you split the list (inclusive of the given idx) to the right and inserted in the query value in between the split
        ## here times is the query value and we want to insert somwhere into the df_index timestamp list
        ## so left will always return the idx of the time in df_index that is "forward/ahead" of the queried time 
        idxs = df_index.searchsorted(times, side="left") 
        idxs_list = []
        for i in range(len(times)):
            if idxs[i] > 0 and only_prev: # always return the prev index given index is not first 
                idxs_list.append(idxs[i]-1)
            elif idxs[i] > 0 and (idxs[i] == len(df_index) or math.fabs(times[i] - df_index[idxs[i]-1]) < math.fabs(times[i] - df_index[idxs[i]])): # FIXED BUG HERE WHERE I COMPARED TO SAME IDX - 1 ON THE RIGHT SIDE OF INEQ
                idxs_list.append(idxs[i]-1)
            else:
                idxs_list.append(idxs[i])
        return idxs_list
    
    def get_contact_time_label(self, im_times, centered=False):
        if not centered:
            return im_times[-1]
        else:
            return im_times[len(im_times)//2]

    def get_poses_history(self, times_list, proprio_history_dict, in_cam_frame=False, return_pxls=False):
        if self.dyn_cam:
            depth_tf_world = self.get_inv_cam_extrin(times_list, depth=True)
        else:
            depth_tf_world = self.depth_tf_world

        num_samples = int(proprio_history_dict['time_window'] * proprio_history_dict['sample_freq'])
        proprio_hist_times = np.linspace(min(times_list) - proprio_history_dict['time_window'], min(times_list), num_samples, endpoint=True).tolist()
        nrst_proprio_hist_idxs = self.get_nearest_idxs(proprio_hist_times, self.proprio_df.index)
        nrst_tfs_np = np.array(self.proprio_df.iloc[nrst_proprio_hist_idxs][self.pose_topics].values)
        nrst_poses_np = []
        pose_pxls = []
        for i in range(nrst_tfs_np.shape[0]):
            pose = self.affine_tf_to_pose(nrst_tfs_np[i])
            if in_cam_frame:
                pose = self.transform_pose(pose, depth_tf_world)
            nrst_poses_np.append(pose)
            if return_pxls:            
                pose_prj = self.point_proj(self.K_depth, pose[:3], depth_tf_world)
                if self.im_resize is None: # if im_resize is None
                    pose_pxls.append(pose_prj)
                else:
                    ## TODO fix this resize reprojection by scaling the projection matrix...
                    pose_prj_resized = ((self.im_resize[0]/self.depth_shape[0])*pose_prj).astype(int)
                    pose_pxls.append(pose_prj_resized)

        nrst_poses_np = np.array(nrst_poses_np)
        if return_pxls:
            pose_pxls = np.array(pose_pxls) 
            return nrst_poses_np, pose_pxls
        else:
            return nrst_poses_np

    # https://stackoverflow.com/a/61704344
    def lambda_max(self, arr, axis=None, key=None, keepdims=False):
        if callable(key):
            idxs = np.argmax(key(arr), axis)
            if axis is not None:
                idxs = np.expand_dims(idxs, axis)
                result = np.take_along_axis(arr, idxs, axis)
                if not keepdims:
                    result = np.squeeze(result, axis=axis)
                return result
            else:
                return arr.flatten()[idxs]
        else:
            return np.amax(arr, axis)
    #### VISUALIZATION UTILS

    def tensor_to_depth_im(self, im_tensor, colormap, is_np = False, return_BGR=False):
        if not np:
            image_np = np.array(self.tensor_to_float32(im_tensor))
        else: 
            image_np = im_tensor 
        image_np_uint8 = (255 * image_np).astype(np.uint8)
        image_np_color = cv2.applyColorMap(image_np_uint8, colormap)
        if return_BGR:
            return image_np_color
        else:
            return cv2.cvtColor(image_np_color, cv2.COLOR_BGR2RGB)

    ## FOR SPLITTING DATASETS
    def get_episode_name(self, episode_idx):
        # return self.episode_data_dir_path_list[episode_idx].split('/')[-1]
        return os.path.basename(self.episode_data_dir_path_list[episode_idx])
        
    def get_num_episodes(self):
        return len(self.depth_episode_timestamp_series.index)
    
    def get_indices_for_episode_list(self, episode_list):
        return self.depth_episode_timestamp_series.index.get_indexer_for(episode_list)

if __name__ == '__main__':
    root_dir = '/mnt/hdd/datasetsHDD/contact_estimation/real'
    dataset_name = 'teleop_real_480x848_no_clutter_combined'
    dataset_dir_path = os.path.join(root_dir, dataset_name)
    dataset = FlowDataset(dataset_dir_path)
    dataset[0]