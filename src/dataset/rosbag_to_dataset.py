
import rosbag
import os
import numpy as np

import cv2

import argparse

import yaml
from scipy.spatial.transform import Rotation as R

import sys
from pathlib import Path

import pickle

import natsort
import pandas as pd

from tqdm import tqdm
import shutil
import glob


# get current path that the script lives in, not where it is called from
curr_path_dir = Path(__file__).parent

rosbag_pandas_path = Path(curr_path_dir.parent, 'rosbag_pandas', 'src')
Path.exists(rosbag_pandas_path)
# os.path.join(curr_path, '..')
if str(rosbag_pandas_path) not in sys.path:
    sys.path.append(str(rosbag_pandas_path))

import rosbag_pandas

from extract_contact_per_episode import ContactExtractor

from extract_flow import FlowExtractor

import natsort
import glob

import time, datetime

def invert_transform(tf):
    R = tf[0:3, 0:3]
    T = tf[:3, -1]
    tf_inv = np.diag([0.,0.,0.,1.])
    tf_inv[:3, :3] = R.T
    tf_inv[:3, -1] = np.matmul(-R.T, T)
    return tf_inv

class RosbagToDataset():
    def __init__(self, experiment_root_dir, 
                 extract_robot_state, 
                 extract_contact, extract_contact_masks, 
                 extract_depth, extract_color, 
                 extract_bag_info,
                 extract_flow, flow_model_args = None,
                 downsample_images=False, downsampled_image_width=320, downsampled_image_height=240,
                 delete_original_after_downsampling=False,
                 real_dataset=False,
                 l515 = False, align_l515_to_d455_depth=False, d455_depth_K = None, l515_depth_resolution = None, d455_depth_resolution = None,
                 align_color_to_depth=False, align_flow_to_depth=False,
                 from_rosbags=False, ft_calibration_dataset=False, 
                 only_write_image_timestamps=False,
                 overwrite=False, max_depth_clip = 3.0,
                 ):
        self.only_write_image_timestamps = only_write_image_timestamps
        self.overwrite = overwrite 
        self.from_rosbags = from_rosbags
        
        self.real_dataset = real_dataset
        self.downsample_images = downsample_images
        self.downsampled_image_width = downsampled_image_width
        self.downsampled_image_height = downsampled_image_height
        self.delete_original_after_downsampling = delete_original_after_downsampling
        self.align_color_to_depth = align_color_to_depth
        self.align_flow_to_depth = align_flow_to_depth
        self.real_episode = real_dataset
        self.extract_contact_masks = extract_contact_masks
        self.max_depth_clip = max_depth_clip
        self.experiment_root_dir_path = experiment_root_dir
        self.episode_rosbag_dir_path = os.path.join(experiment_root_dir, 'episode_rosbags')
        self.episode_data_dir_path = os.path.join(experiment_root_dir, 'episode_data')
        self.l515 = l515
        self.align_l515_to_d455_depth = align_l515_to_d455_depth
        self.extract_depth = extract_depth
        self.extract_color = extract_color
        self.extract_bag_info = extract_bag_info
        self.extract_robot_state_dataframe = extract_robot_state
        self.extract_contact_dataframe = extract_contact
        self.ft_calibration_dataset = ft_calibration_dataset
        self.extract_flow = extract_flow
        
        if self.real_episode:
            if l515:
                self.depth_topic = '/camera/depth/image_rect_raw_throttled'
            else:
                self.depth_topic = '/camera/depth/image_rect_raw'
            self.color_topic = '/camera/color/image_raw'
            self.depth_extrinsics_filename = 'D_tf_W.npy'
            self.depth_type = 'depth'
            self.color_type = 'color'
            if self.align_l515_to_d455_depth:
                # self.l515_depth_K = l515_depth_K
                # get first episode name from the experiment root directory using listdir and making sure isdir
                first_episode_name = natsort.natsorted([episode_name for episode_name in os.listdir(self.episode_data_dir_path) if os.path.isdir(os.path.join(self.episode_data_dir_path, episode_name))])[0]
                # get the first episode directory
                first_episode_dir = os.path.join(self.episode_data_dir_path, first_episode_name)
                # read the npy intrinsic from the directory of the first episode
                self.l515_depth_K = np.load(os.path.join(first_episode_dir, self.depth_type, 'depth_K.npy'))
                self.d455_depth_K = d455_depth_K
                self.l515_depth_resolution = l515_depth_resolution # (width, height)
                self.d455_depth_resolution = d455_depth_resolution # (width, height)
                # align l515 depth to d455 depth which also different resolution
                # l515 depth is 480 x 640 while d455 depth is 480 x 848
                upper_left = self.d455_depth_K @ np.linalg.inv(self.l515_depth_K) @ np.array([0, 0, 1])
                lower_right = self.d455_depth_K @ np.linalg.inv(self.l515_depth_K) @ np.array([self.l515_depth_resolution[0], self.l515_depth_resolution[1], 1])

                # get the new width and height of the aligned l515 depth image
                self.new_depth_width = int(lower_right[0] - upper_left[0])
                self.new_depth_height = int(lower_right[1] - upper_left[1])

                # get the padding for the aligned l515 depth image
                self.padding_left = int(upper_left[0])
                self.padding_top = int(upper_left[1])
                self.padding_right = self.d455_depth_resolution[0] - self.new_depth_width - self.padding_left
                self.padding_bottom = self.d455_depth_resolution[1] - self.new_depth_height - self.padding_top
        else:
            if self.l515:
                self.depth_topic = '/camera/depth/image_raw'
                self.depth_extrinsics_filename = 'D_tf_W.npy'
                self.depth_type = 'depth'
                self.color_type = 'color_aligned_to_depth'
            else:
                self.depth_topic = '/camera/aligned_depth_to_color/image_raw'
                self.depth_extrinsics_filename = 'C_tf_W.npy'
                self.depth_type = 'aligned_depth_to_color'
                self.color_type = 'color'
            self.color_topic = '/camera/color/image_raw'
        
        if extract_flow:
            assert flow_model_args is not None, 'Must provide flow model arguments if extracting flow'
            self.flow_extractor_model_args = flow_model_args
        
        self.episode_number = 0

        # hardcode the grasped object name 
        self.grasped_object_name = 'EE_object'

        # use pandas multiindex for mapping image indices to episode numbers and timestamps
        self.depth_episode_array = []
        self.depth_timestamp_array = []
        
        self.color_episode_array = []
        self.color_timestamp_array = []

    def trim_dataset(self):
        # assuming just trimming the end of the dataset. TODO extend functionality later
        # get number of episode directories, excluding the pickle files
        episode_list = os.listdir(self.episode_data_dir_path)
        episode_list = [episode_dir for episode_dir in episode_list if os.path.isdir(os.path.join(self.episode_data_dir_path, episode_dir))]
        num_episodes = len(episode_list)
        # load the old color_episode_timestamp_series and depth_episode_timestamp_series
        color_episode_timestamp_series = pd.read_pickle(os.path.join(self.episode_data_dir_path, 'color_episode_timestamp_series.pkl'))
        depth_episode_timestamp_series = pd.read_pickle(os.path.join(self.episode_data_dir_path, 'depth_episode_timestamp_series.pkl'))
        # trim the series
        color_episode_timestamp_series = color_episode_timestamp_series.iloc[color_episode_timestamp_series.index < num_episodes]
        depth_episode_timestamp_series = depth_episode_timestamp_series.iloc[depth_episode_timestamp_series.index < num_episodes]
        # save the trimmed series
        color_episode_timestamp_series.to_pickle(os.path.join(self.episode_data_dir_path, 'color_episode_timestamp_series.pkl'))
        depth_episode_timestamp_series.to_pickle(os.path.join(self.episode_data_dir_path, 'depth_episode_timestamp_series.pkl'))

    def process_dataset(self):
        # get list of episodes
        episode_list = os.listdir(self.episode_data_dir_path)
        # get only those that are directories
        if self.from_rosbags:
            # check if any bags end with .active
            if any([episode_rosbag_path.endswith('.active') for episode_rosbag_path in episode_list]):
                # raise error if any bag is still active and print the dataset name
                raise RuntimeError('Some rosbags in {} are still active. Please close them before processing the dataset.'.format(self.episode_data_dir_path))
            # get list of bags in episode_rosbags
            episode_rosbag_list = glob.glob(os.path.join(self.episode_rosbag_dir_path, '*.bag'))
            # construct list but with .bag stripped
            episode_list = [os.path.basename(episode_rosbag_path)[:-4] for episode_rosbag_path in episode_rosbag_list]
        else:
            episode_list = [episode_dir for episode_dir in episode_list if os.path.isdir(os.path.join(self.episode_data_dir_path, episode_dir))]
        episode_list = natsort.natsorted(episode_list)
        # use tqdm to show progress bar
        for episode_dir in tqdm(episode_list, desc='Processing episodes'):
            self.process_episode(episode_dir)
        
        # save the image indices to episode number and timestamp mapping to a multiindex dataframe
        if self.extract_depth:
            depth_episode_multiindex = pd.Index(self.depth_episode_array, name='episode')
            depth_episode_timestamp_series = pd.Series(self.depth_timestamp_array, index=depth_episode_multiindex, name='timestamp')
            depth_episode_timestamp_series.to_pickle(os.path.join(self.episode_data_dir_path, 'depth_episode_timestamp_series.pkl'))

        if self.extract_color:
            color_episode_multiindex = pd.Index(self.color_episode_array, name='episode')
            color_episode_timestamp_series = pd.Series(self.color_timestamp_array, index=color_episode_multiindex, name='timestamp')
            color_episode_timestamp_series.to_pickle(os.path.join(self.episode_data_dir_path, 'color_episode_timestamp_series.pkl'))
        
        if self.extract_flow:
            if self.l515 and not self.real_dataset: # in sim with l515 color is aligned to depth
                self.flow_extractor = FlowExtractor(self.experiment_root_dir_path, self.flow_extractor_model_args, aligned_color_to_depth=True)
            else:
                self.flow_extractor = FlowExtractor(self.experiment_root_dir_path, self.flow_extractor_model_args, aligned_color_to_depth=False)
            self.flow_extractor.extract_flow()
        
        if self.align_flow_to_depth:
            if self.l515 and self.align_l515_to_d455_depth:
                depth_dir_name = 'depth_aligned_to_d455'
            else:
                depth_dir_name = 'depth'
            for episode_dir in tqdm(episode_list,desc='aligning flow to depth'):
                episode_data_dir_path = os.path.join(self.episode_data_dir_path, episode_dir)
                self.align_to_depth_images(episode_data_dir_path, depth_dir_name, 'flow_image', pkl_images=False)
                self.align_to_depth_images(episode_data_dir_path, depth_dir_name, 'flow', pkl_images=True)
            
        if self.extract_contact_masks:
            if self.l515:
                self.contact_mask_extractor = ContactExtractor(self.experiment_root_dir_path, color_aligned_to_depth=True, is_real_dataset=self.real_dataset)
            else:
                self.contact_mask_extractor = ContactExtractor(self.experiment_root_dir_path, color_aligned_to_depth=False, is_real_dataset=self.real_dataset)
            self.contact_mask_extractor.process_dataset()
        
        # need to downsample after flow and alignment
        if self.downsample_images:
            # loop through all episodes we've processed in this dataset
            for episode_dir in tqdm(episode_list, desc='Downsampling images'):
                self.downsample_images_in_episode(episode_dir)
        
        # write a yaml file with the datetime info as the name and dump the settings used
        self.write_settings_yaml()
    
    def write_settings_yaml(self):
        # get the settings used
        settings_dict = {}
        settings_dict['experiment_root_dir_path'] = self.experiment_root_dir_path
        settings_dict['episode_rosbag_dir_path'] = self.episode_rosbag_dir_path
        settings_dict['episode_data_dir_path'] = self.episode_data_dir_path
        settings_dict['from_rosbags'] = self.from_rosbags
        settings_dict['extract_bag_info'] = self.extract_bag_info
        settings_dict['extract_robot_state_dataframe'] = self.extract_robot_state_dataframe
        settings_dict['extract_contact_dataframe'] = self.extract_contact_dataframe
        settings_dict['extract_depth'] = self.extract_depth
        settings_dict['extract_color'] = self.extract_color
        settings_dict['extract_flow'] = self.extract_flow
        settings_dict['align_color_to_depth'] = self.align_color_to_depth
        settings_dict['align_flow_to_depth'] = self.align_flow_to_depth
        settings_dict['align_l515_to_d455_depth'] = self.align_l515_to_d455_depth
        settings_dict['extract_contact_masks'] = self.extract_contact_masks
        settings_dict['real_dataset'] = self.real_dataset
        settings_dict['l515'] = self.l515
        # settings_dict['flow_extractor_model_args'] = self.flow_extractor_model_args
        settings_dict['downsample_images'] = self.downsample_images
        settings_dict['downsampled_image_width'] = self.downsampled_image_width
        settings_dict['downsampled_image_height'] = self.downsampled_image_height
        settings_dict['max_depth_clip'] = self.max_depth_clip
        settings_dict['ft_calibration_dataset'] = self.ft_calibration_dataset

        # get the current datetime
        now = datetime.datetime.now()
        dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
        settings_yaml_path = os.path.join(self.experiment_root_dir_path, dt_string + '.yaml')
        with open(settings_yaml_path, 'w') as outfile:
            yaml.dump(settings_dict, outfile, default_flow_style=False)

    def process_episode(self, episode_dir):
        rosbag_path = os.path.join(self.episode_rosbag_dir_path, episode_dir + '.bag')
        episode_data_dir_path = os.path.join(self.episode_data_dir_path, episode_dir)
        if not os.path.exists(episode_data_dir_path):
            os.makedirs(episode_data_dir_path)
            
        # load the rosbag
        if (self.extract_bag_info or 
                self.extract_robot_state_dataframe or 
                self.extract_contact_dataframe or 
                self.extract_depth or
                self.extract_color
                ):
            bag = rosbag.Bag(rosbag_path)

        if self.extract_bag_info:
            self.extract_info_dict(bag, episode_data_dir_path)
        if self.extract_robot_state_dataframe:
            self.extract_robot_state(bag, episode_data_dir_path)
        if self.extract_contact_dataframe and not self.real_episode:
            self.extract_contact(bag, episode_data_dir_path)
        if self.extract_depth:
            # ensure depth topic exists in bag
            if not self.depth_topic in bag.get_type_and_topic_info()[1].keys():
                raise ValueError('depth topic {} not found in bag {}'.format(self.depth_topic, rosbag_path))
            self.extract_depth_images(bag, episode_data_dir_path, only_write_timestamps=self.only_write_image_timestamps)
        if self.align_l515_to_d455_depth:
            self.align_l515_to_d455_depth_images(episode_data_dir_path)
        if self.extract_color:
            self.extract_color_images(bag, episode_data_dir_path, only_write_timestamps=self.only_write_image_timestamps)
        # self.correct_bad_l515_color_extrinsics(episode_data_dir_path)
        if self.align_color_to_depth:
            if self.l515 and self.align_l515_to_d455_depth:
                depth_dir_name = 'depth_aligned_to_d455'
            else:
                depth_dir_name = 'depth'
            self.align_to_depth_images(episode_data_dir_path, depth_dir_name, self.color_type, only_write_timestamps=self.only_write_image_timestamps)
        
        if (self.extract_bag_info or 
                self.extract_robot_state_dataframe or 
                self.extract_contact_dataframe or 
                self.extract_depth or
                self.extract_color
                ):
            bag.close()
        self.episode_number += 1

    def downsample_images_in_episode(self, episode_data_dir_path):
        # episode_data_dir_path must start with the experiment root dir path
        if not episode_data_dir_path.startswith(self.episode_data_dir_path):
            # append the experiment root dir path to the episode_data_dir_path
            episode_data_dir_path = os.path.join(self.episode_data_dir_path, episode_data_dir_path)
        # downample the depth images
        depth_dir_path = os.path.join(episode_data_dir_path, 'depth')
        depth_downsampled_dir_path = os.path.join(episode_data_dir_path, 'depth_downsampled')
        if not os.path.exists(depth_downsampled_dir_path):
            os.makedirs(depth_downsampled_dir_path)
        # use glob and natsort to get the depth images in order
        depth_image_paths = natsort.natsorted(glob.glob(os.path.join(depth_dir_path, '*.png')))
        for depth_image_path in depth_image_paths:
            # make sure the datatype uint16 is preserved while reading and writing
            depth_image = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)
            # use cubic interpolation for downsampling
            depth_image_downsampled = cv2.resize(depth_image, (self.downsampled_image_width, self.downsampled_image_height), interpolation=cv2.INTER_CUBIC)
            depth_image_downsampled_path = os.path.join(depth_downsampled_dir_path, os.path.basename(depth_image_path))
            cv2.imwrite(depth_image_downsampled_path, depth_image_downsampled)
        image_width = depth_image.shape[1]
        image_height = depth_image.shape[0]
        # copy over the intrinsics, extrinsics (both are npy files), and finally the timestamp_series.pkl
        # shutil.copy(os.path.join(depth_dir_path, 'depth_K.npy'), os.path.join(depth_downsampled_dir_path, 'depth_K.npy'))
        # resize the intrinsics matrix before copying over
        depth_K = np.load(os.path.join(depth_dir_path, 'depth_K.npy'))
        depth_K[0, :] = depth_K[0, :] * (self.downsampled_image_width / image_width)
        depth_K[1, :] = depth_K[1, :] * (self.downsampled_image_height / image_height)
        np.save(os.path.join(depth_downsampled_dir_path, 'depth_K.npy'), depth_K)
        shutil.copy(os.path.join(depth_dir_path, 'D_tf_W.npy'), os.path.join(depth_downsampled_dir_path, 'D_tf_W.npy'))
        shutil.copy(os.path.join(depth_dir_path, 'timestamp_series.pkl'), os.path.join(depth_downsampled_dir_path, 'timestamp_series.pkl'))

        # downsample the color images
        color_dir_path = os.path.join(episode_data_dir_path, 'color_aligned_to_depth')
        color_downsampled_dir_path = os.path.join(episode_data_dir_path, 'color_aligned_to_depth_downsampled')
        if not os.path.exists(color_downsampled_dir_path):
            os.makedirs(color_downsampled_dir_path)
        # use glob and natsort to get the color images in order
        color_image_paths = natsort.natsorted(glob.glob(os.path.join(color_dir_path, '*.png')))
        for color_image_path in color_image_paths:
            color_image = cv2.imread(color_image_path)
            color_image_downsampled = cv2.resize(color_image, (self.downsampled_image_width, self.downsampled_image_height), interpolation=cv2.INTER_CUBIC)
            color_image_downsampled_path = os.path.join(color_downsampled_dir_path, os.path.basename(color_image_path))
            cv2.imwrite(color_image_downsampled_path, color_image_downsampled)
        image_width = color_image.shape[1]
        image_height = color_image.shape[0]
        # copy over the intrinsics, extrinsics (both are npy files), and finally the timestamp_series.pkl
        # shutil.copy(os.path.join(color_dir_path, 'color_K.npy'), os.path.join(color_downsampled_dir_path, 'color_K.npy'))
        # resize the intrinsics matrix before copying over
        color_K = np.load(os.path.join(color_dir_path, 'color_K.npy'))
        color_K[0, :] = color_K[0, :] * (self.downsampled_image_width / image_width)
        color_K[1, :] = color_K[1, :] * (self.downsampled_image_height / image_height)
        np.save(os.path.join(color_downsampled_dir_path, 'color_K.npy'), color_K)
        shutil.copy(os.path.join(color_dir_path, 'C_tf_W.npy'), os.path.join(color_downsampled_dir_path, 'C_tf_W.npy'))
        shutil.copy(os.path.join(color_dir_path, 'timestamp_series.pkl'), os.path.join(color_downsampled_dir_path, 'timestamp_series.pkl'))

        # downsample the flows which are not images but numpy arrays of dimension (height, width, 2)
        flow_dir_path = os.path.join(episode_data_dir_path, 'flow')
        flow_downsampled_dir_path = os.path.join(episode_data_dir_path, 'flow_downsampled')
        if not os.path.exists(flow_downsampled_dir_path):
            os.makedirs(flow_downsampled_dir_path)
        # use glob and natsort to get the flow files in order
        flow_file_paths = natsort.natsorted(glob.glob(os.path.join(flow_dir_path, '*.pkl')))
        for flow_file_path in flow_file_paths:
            flow = pickle.load(open(flow_file_path, 'rb'))
            flow_downsampled = cv2.resize(flow, (self.downsampled_image_width, self.downsampled_image_height), interpolation=cv2.INTER_CUBIC)
            flow_downsampled_path = os.path.join(flow_downsampled_dir_path, os.path.basename(flow_file_path))
            pickle.dump(flow_downsampled, open(flow_downsampled_path, 'wb'))
        image_width = flow.shape[1]
        image_height = flow.shape[0]
        # copy over the intrinsics, extrinsics (both are npy files)
        # resize the intrinsics matrix before copying over
        flow_K = np.load(os.path.join(flow_dir_path, 'color_K.npy'))
        flow_K[0, :] = flow_K[0, :] * (self.downsampled_image_width / image_width)
        flow_K[1, :] = flow_K[1, :] * (self.downsampled_image_height / image_height)
        np.save(os.path.join(flow_downsampled_dir_path, 'color_K.npy'), flow_K)
        shutil.copy(os.path.join(flow_dir_path, 'C_tf_W.npy'), os.path.join(flow_downsampled_dir_path, 'C_tf_W.npy'))

        # delete the original color, depth, and flow directories if flag is true
        if self.delete_original_after_downsampling:
            shutil.rmtree(depth_dir_path)
            shutil.rmtree(color_dir_path)
            shutil.rmtree(flow_dir_path)

    def correct_bad_l515_color_extrinsics(self, episode_data_dir_path):
        C_tf_D = np.array([[ 0.999984, 0.00504348 , 0.00242612  , 1.80074239324313e-05],
                                    [-0.0050906, 0.99979, 0.0198258, 0.0141367902979255],     
                                    [ -0.00232563, -0.0198378,  0.999801, -0.00364168104715645],
                                    [0.0, 0.0, 0.0, 1.0]])
        # get depth extrinsic
        D_tf_W = np.load(os.path.join(episode_data_dir_path, 'depth', 'D_tf_W.npy'))
        C_tf_W = np.matmul(C_tf_D, D_tf_W)
        np.save(os.path.join(episode_data_dir_path, 'color', 'C_tf_W.npy'), C_tf_W)
    
    def align_l515_to_d455_depth_images(self, episode_data_dir_path):
        # make a new directory called depth_aligned_to_d455
        depth_aligned_to_d455_dir_path = os.path.join(episode_data_dir_path, self.depth_type + '_aligned_to_d455')
        if not os.path.exists(depth_aligned_to_d455_dir_path):
            os.makedirs(depth_aligned_to_d455_dir_path)
        # copy the depth extrinsic 
        shutil.copy(os.path.join(episode_data_dir_path, self.depth_type, 'D_tf_W.npy'), os.path.join(depth_aligned_to_d455_dir_path, 'D_tf_W.npy'))
        # write the d455 depth intrinsics numpy array to the directory
        np.save(os.path.join(depth_aligned_to_d455_dir_path, 'depth_K.npy'), self.d455_depth_K)

        # list the depth images in the episode
        depth_image_list = glob.glob(os.path.join(episode_data_dir_path, self.depth_type, '*.png'))
        # sort the list
        depth_image_list = natsort.natsorted(depth_image_list)
        for depth_image_path in depth_image_list:
            # load the depth image
            depth_image = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED) # make sure this is unsigned int 16
            # resize based on depth new resolution
            depth_image = cv2.resize(depth_image, (self.new_depth_width, self.new_depth_height), interpolation=cv2.INTER_CUBIC)
            # now pad out to the d455 resolution
            depth_image = cv2.copyMakeBorder(depth_image, self.padding_top, self.padding_bottom, self.padding_left, self.padding_right, cv2.BORDER_CONSTANT, value=self.max_depth_clip*1000)# make sure this is unsigned int 16
            # save the depth image with the same name but in new directory
            depth_image_name = os.path.basename(depth_image_path)
            depth_image_path = os.path.join(depth_aligned_to_d455_dir_path, depth_image_name)
            cv2.imwrite(depth_image_path, depth_image)
    
    def align_to_depth_images(self, episode_data_dir_path, depth_dir_name, image_dir_name, pkl_images=False, only_write_timestamps=False):
        # make a new directory called color_aligned_to_depth
        image_aligned_to_depth_dir_path = os.path.join(episode_data_dir_path, image_dir_name + '_aligned_to_' + depth_dir_name)
        if not os.path.exists(image_aligned_to_depth_dir_path):
            os.makedirs(image_aligned_to_depth_dir_path)
        # check that the timestamp series exists
        if os.path.exists(os.path.join(episode_data_dir_path, image_dir_name, 'timestamp_series.pkl')):
            shutil.copy(os.path.join(episode_data_dir_path, image_dir_name, 'timestamp_series.pkl'), os.path.join(image_aligned_to_depth_dir_path, 'timestamp_series.pkl'))
        
        if not only_write_timestamps:
            # copy over the depth non-image files
            shutil.copy(os.path.join(episode_data_dir_path, depth_dir_name, 'D_tf_W.npy'), os.path.join(image_aligned_to_depth_dir_path, 'D_tf_W.npy'))
            shutil.copy(os.path.join(episode_data_dir_path, depth_dir_name, 'depth_K.npy'), os.path.join(image_aligned_to_depth_dir_path, 'depth_K.npy'))
            
            # get the image intrinsic and extrinsic
            image_K = np.load(os.path.join(episode_data_dir_path, image_dir_name, 'color_K.npy'))
            C_tf_W = np.load(os.path.join(episode_data_dir_path, image_dir_name, 'C_tf_W.npy'))
            # get the depth intrinsic and extrinsic
            depth_K = np.load(os.path.join(episode_data_dir_path, depth_dir_name, 'depth_K.npy'))
            D_tf_W = np.load(os.path.join(episode_data_dir_path, depth_dir_name, 'D_tf_W.npy'))

            # list the color images in the episode
            if pkl_images:
                image_list = glob.glob(os.path.join(episode_data_dir_path, image_dir_name, '*.pkl'))
            else:
                image_list = glob.glob(os.path.join(episode_data_dir_path, image_dir_name, '*.png'))

            # also list the depth images in the episode
            depth_image_list = glob.glob(os.path.join(episode_data_dir_path, depth_dir_name, '*.png'))
            # sort the lists
            image_list = natsort.natsorted(image_list)
            depth_image_list = natsort.natsorted(depth_image_list)
            if image_dir_name in ['flow', 'flow_image']:
                # remove the first depth image from list (since it is the first image in the flow sequence)
                depth_image_list.pop(0)
            for image_path, depth_image_path in zip(image_list, depth_image_list):
                # assert that the name (stripped of extension) of the image and depth image are the same
                image_name = os.path.splitext(os.path.basename(image_path))[0]
                depth_image_name = os.path.splitext(os.path.basename(depth_image_path))[0]
                # assert image_name == depth_image_name, 'image and depth image names do not match'
                # load the image
                if pkl_images:
                    with open(image_path, 'rb') as f:
                        image = pickle.load(f)
                else:
                    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
                # load the depth image
                depth_image = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)
                # align the image to the depth image
                if pkl_images:
                    aligned_image_path = os.path.join(image_aligned_to_depth_dir_path, image_name + '.pkl') 
                    self.align_to_depth_image(image_K, C_tf_W, depth_K, D_tf_W, depth_image, image, aligned_image_path, save_as_pkl=True)
                else:
                    aligned_image_path = os.path.join(image_aligned_to_depth_dir_path, image_name + '.png')
                    self.align_to_depth_image(image_K, C_tf_W, depth_K, D_tf_W, depth_image, image, aligned_image_path, save_as_pkl=False)

    def align_to_depth_image(self, image_K, C_tf_W, depth_K, D_tf_W, depth_image, image, aligned_image_path, save_as_pkl=False):
        depth_height, depth_width = depth_image.shape
        rows, cols = np.meshgrid(np.arange(depth_height), np.arange(depth_width), indexing='ij')
        depth_K_inv = np.linalg.inv(depth_K)

        W_trans_D = - D_tf_W[:3, :3].T @ D_tf_W[:3, 3]
        W_R_D = D_tf_W[:3, :3].T
        C_trans_W = C_tf_W[:3, 3]
        C_R_W = C_tf_W[:3, :3]

        image_channels = image.shape[2]

        # reshape to vectors
        pixel_depth = np.stack((cols.flatten(), rows.flatten(), np.ones((cols.size,))), axis=1)
        pixel_depth = pixel_depth.T

        # transform to world coordinates
        # assuming depth is still uint16 in mm
        # convert to meters
        depth_image = depth_image.astype(np.float64) / 1000.0
        pcl_in_world = W_R_D @ (depth_image.flatten()* (depth_K_inv @ pixel_depth) )+ W_trans_D.reshape(3,1)
        pcl_in_world = pcl_in_world.T

        # project to color image coordinates
        valid_color_pixel_coordinates = np.matmul(image_K, np.matmul(C_R_W, pcl_in_world.T) + C_trans_W.reshape(3,1))
        valid_color_pixel_coordinates /= valid_color_pixel_coordinates[-1, :]

        # extract color values from color image
        # get indices of pixels that are within the bounds of the color image
        u_old_color = valid_color_pixel_coordinates[0]
        v_old_color = valid_color_pixel_coordinates[1]
        mask = ((u_old_color >= 0) & (u_old_color < image.shape[1]) & 
                (v_old_color >= 0) & (v_old_color < image.shape[0]))

        # create new color image
        new_image_shape = (depth_height, depth_width, image_channels)
        new_image = np.zeros(new_image_shape)
        
        index = np.where(mask.reshape(depth_height, depth_width))
        new_image[index[0], index[1], :] = image[v_old_color[mask].astype(int), u_old_color[mask].astype(int), :]
        if not save_as_pkl:
            # new_image = cv2.cvtColor(new_image.astype(np.uint8), cv2.COLOR_RGB2BGR)
            cv2.imwrite(aligned_image_path, new_image)
        else:
            # write the flow to pkl
            with open(aligned_image_path, 'wb') as f:
                pickle.dump(new_image, f)

    def extract_info_dict(self, rosbag, episode_data_dir_path):
        # check if info_dict already exists
        if os.path.exists(os.path.join(episode_data_dir_path, 'info_dict.pkl')):
            if self.overwrite:
                print('overwriting info dict!')
            else:
                print('info dict already exists, skipping!')
                return
        # extract the info dict
        info_dict = yaml.safe_load(rosbag._get_yaml_info())
        # save the info dict
        info_dict_path = os.path.join(episode_data_dir_path, 'info_dict.pkl')
        with open(info_dict_path, 'wb') as f:
            pickle.dump(info_dict, f)

    def extract_robot_state(self, rosbag, episode_data_dir_path):
        if os.path.exists(os.path.join(episode_data_dir_path, 'proprio_df.pkl')):
            if self.overwrite:
                print('overwriting robot state df!')
            else:
                print('robot state df exists, skipping!')
                return
        proprio_df = rosbag_pandas.bag_to_dataframe(rosbag, include=['/panda/franka_state_controller_custom/franka_states'])
        proprio_df.to_pickle(os.path.join(episode_data_dir_path, 'proprio_df.pkl'))
        if self.ft_calibration_dataset:
            number_messages = len(proprio_df)
            # write num messages to file as pkl
            with open(os.path.join(episode_data_dir_path, 'num_messages.pkl'), 'wb') as f:
                pickle.dump(number_messages, f)
        del proprio_df
        
    def extract_contact(self, rosbag, episode_data_dir_path):
        if os.path.exists(os.path.join(episode_data_dir_path, 'contact_df.pkl')):
            print('overwriting contact df!')
            # delete the existing contact df
            os.remove(os.path.join(episode_data_dir_path, 'contact_df.pkl'))
        contact_df = rosbag_pandas.bag_to_dataframe(rosbag, include=['/contact_data'])
        ## need to filter out all rows where none of the contact state collision names have the object name
        ## this is in order for the no contact estimation heuristic to work 
        collision_df = contact_df.loc[:, contact_df.columns.str.contains('collision')]
        if not collision_df.empty:
            contact_filtered_df = contact_df[collision_df.astype(str).sum(axis=1).str.contains(self.grasped_object_name)]
            contact_filtered_df.to_pickle(os.path.join(episode_data_dir_path, 'contact_df.pkl'))
        else:
            # write an empty file to indicate that there is no contact data
            Path(os.path.join(episode_data_dir_path, 'no_contact')).touch()
        del contact_df

    def extract_contact_info_from_df(self, contact_df):
        # TODO
        raise NotImplementedError

    def extract_episode_info_dataframe(self, rosbag, episode_data_dir_path):
        if os.path.exists(os.path.join(episode_data_dir_path, 'ep_info_df.pkl')):
            if self.overwrite:
                print('overwriting episode info df!')
            else:  
                print('episode info df exists, skipping!')
                return
        ep_info_df = rosbag_pandas.bag_to_dataframe(rosbag, include=['/episode_info'])
        ep_info_df.to_pickle(os.path.join(episode_data_dir_path, 'ep_info_df.pkl'))
        del ep_info_df

    def extract_depth_images(self, rosbag, episode_data_dir_path, only_write_timestamps=False):
        # episode_number = int(os.path.basename(episode_data_dir_path).split('_')[0])
        count = 0
        timestamp_start_idx = len(self.depth_timestamp_array)
        # wrap this in tqdm
        for topic, msg, t in tqdm(rosbag.read_messages(topics=self.depth_topic), desc='extracting depth images'):
            # cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            timestamp = msg.header.stamp
            if not only_write_timestamps:
                im_name = "{0}-{1}.png".format(count, timestamp) #string formatting timestamp is equivalent to str(to_nsec)
                cv_img = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width) #data is in mm
                cv2.imwrite(os.path.join(episode_data_dir_path, self.depth_type, im_name), cv_img)
            self.depth_timestamp_array.append(timestamp.to_sec()) #equivalent to to_sec()
            self.depth_episode_array.append(self.episode_number)
            # self.image_index_array.append(count)
            count += 1
        
        # write the timestamp array to file
        depth_timestamp_series = pd.Series(self.depth_timestamp_array[timestamp_start_idx:])
        depth_timestamp_series.to_pickle(os.path.join(episode_data_dir_path, self.depth_type, 'timestamp_series.pkl'))
        
    def extract_color_images(self, rosbag, episode_data_dir_path, only_write_timestamps=False):
        # episode_number = int(os.path.basename(episode_data_dir_path).split('_')[0])
        count = 0
        # get starting index for the timestamp array
        timestamp_start_idx = len(self.color_timestamp_array)
        for topic, msg, _ in tqdm(rosbag.read_messages(topics=self.color_topic), desc='extracting color images'):
            # cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            timestamp = msg.header.stamp
            if not only_write_timestamps:
                im_name = "{0}-{1}.png".format(count, timestamp) #string formatting timestamp is equivalent to str(to_nsec)
                cv_img = cv2.cvtColor(np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1), cv2.COLOR_RGB2BGR) # BGR ordering
                cv2.imwrite(os.path.join(episode_data_dir_path, self.color_type, im_name), cv_img)
            self.color_timestamp_array.append(timestamp.to_sec()) #equivalent to to_sec()
            self.color_episode_array.append(self.episode_number)
            count += 1
        
        # write the color timestamp series for this episode to the episode data dir
        color_timestamp_series = pd.Series(self.color_timestamp_array[timestamp_start_idx:])
        color_timestamp_series.to_pickle(os.path.join(episode_data_dir_path, self.color_type, 'timestamp_series.pkl'))

        
if __name__ == '__main__':
    real_dataset = False
    from_rosbags = True
    l515=True
    overwrite = False

    extract_robot_state = True
    extract_depth = True
    extract_color = True
    extract_bag_info = True
    extract_flow = True
    
    extract_contact = True
    extract_contact_masks = True

    downsample_images = True
    delete_original_after_downsampling = True
    
    align_color_to_depth = False
    align_flow_to_depth = False
    only_write_timestamps = False
    align_l515_to_d455_depth = False
    ft_calibration_dataset = False

    if not real_dataset:
        assert align_color_to_depth == False
        assert align_flow_to_depth == False
    else:
        assert extract_contact == False
        assert extract_contact_masks == False
        assert ft_calibration_dataset == False

    # l515_depth_K = np.array([
    #             [456.84765625, 0.0, 322.90625], 
    #             [0.0, 456.9921875, 245.060546875], 
    #             [0.0, 0.0, 1.0]])
    d455_depth_K = np.array([[426.44219971, 0., 422.1725769],
    [0., 426.44219971, 239.74736023],
    [0.,   0.        ,   1.        ]])
    d455_depth_resolution = (848, 480)
    l515_depth_resolution = (640, 480)

    model_args = None
    if extract_flow:
        model_args = argparse.Namespace()
        model_args.model = '../../RAFT/models/raft-sintel.pth'
        model_args.small = False
        model_args.mixed_precision = False
        model_args.alternate_corr = False

    # dset_root_dir = '/mnt/hdd/datasetsHDD/teleop_test'
    if real_dataset:
        dset_root_dir = '/mnt/hdd/datasetsHDD/contact_estimation/real'
    else:
        dset_root_dir = '/mnt/hdd/datasetsHDD/contact_estimation/simulated'
    experiment_dir_name_list = [
        '15_sec_episodes_8_objects_l515_480640_clearview_realfriction_2023-05-31-19-43-03',
        '15_sec_episodes_8_objects_l515_480640_clearview_realfriction_2023-05-31-21-31-58',
        '15_sec_episodes_8_objects_l515_480640_clearview_realfriction_2023-06-01-01-41-30',
        '15_sec_episodes_8_objects_l515_480640_clearview_realfriction_2023-06-01-05-59-00',
        '15_sec_episodes_8_objects_l515_480640_clearview_realfriction_2023-06-01-07-04-52',
    ]
                                
    for experiment_dir_name in experiment_dir_name_list:
        print(experiment_dir_name)
        experiment_root_dir = os.path.join(dset_root_dir, experiment_dir_name)
        
        bag_extractor = RosbagToDataset(experiment_root_dir, 
        downsample_images=downsample_images,
        from_rosbags=from_rosbags, 
        overwrite=overwrite, 
        extract_robot_state=extract_robot_state, 
        extract_contact=extract_contact, extract_contact_masks=extract_contact_masks, 
        extract_depth=extract_depth, extract_color=extract_color, 
        extract_bag_info=extract_bag_info, real_dataset=real_dataset, 
        ft_calibration_dataset=ft_calibration_dataset, 
        extract_flow=extract_flow, flow_model_args=model_args,
        l515=l515, 
        align_l515_to_d455_depth=align_l515_to_d455_depth, d455_depth_K=d455_depth_K, d455_depth_resolution=d455_depth_resolution, l515_depth_resolution=l515_depth_resolution,
        align_color_to_depth=align_color_to_depth, align_flow_to_depth=align_flow_to_depth,
        only_write_image_timestamps=only_write_timestamps,
        delete_original_after_downsampling=delete_original_after_downsampling,)
        bag_extractor.process_dataset()
        # bag_extractor.trim_dataset()

        # TODO add FT filtering in case of real dataset