import os
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset

import numpy as np
from scipy.spatial.transform import Rotation as R
import pandas as pd

import skimage
import torchvision
import cv2
import sys

import glob
import natsort
import time
import math

import imgaug as ia
import imgaug.augmenters as iaa

import pickle
# import pickle5 as pickle
import copy
import json

from dataclasses import asdict
sys.path.append('../')
from matplotlib import pyplot
from src.dataset.contact_dataclasses import BlurContactMapDict, ContextFrameDict, CroppingDict, OpticalFlowDict, CompensateObjGravityDict, ProprioceptionInputDict, InCamFrameDict, ProprioceptionHistoryDict, AddNoiseDict, ProprioFilterDict

def depth_png_to_map(images_np, max_depth_clip=2.0):
    images_np = images_np.astype(np.float32) / 1000.0
    # exchange all zeros to the max depth value 
    # in sim zeros are the background
    # in real zeros are invalid depth
    images_np[images_np == 0] = max_depth_clip
    images_clipped = np.clip(images_np, 0, max_depth_clip)
    images_normalized = images_clipped / max_depth_clip
    return images_normalized

class ContactDatasetEpisodic(Dataset):
    # if window_size = -1, produce the full episode !
    def __init__(self, 
                experiment_dset, 
                episode_idx=None, store_images=False,
                model_type=None, 
                l515=True, 
                downsampled_images=False,
                window_size=1, grasped_obj_name='EE_object', 
                im_resize=(240,320), centered=False, im_time_offset=0.1, max_depth_clip=2.0, 
                max_num_contact=15, contact_persist_time=0.0025, # at 500hz each contact is .002 sec apart
                is_real_dataset=False, 
                proprio_filter_dict = asdict(ProprioFilterDict()), 
                ft_calibration_dict=None, 
                compensate_obj_gravity_dict = asdict(CompensateObjGravityDict()),
                optical_flow_dict=asdict(OpticalFlowDict()),
                context_frame_dict=asdict(ContextFrameDict()),
                cropping_dict=asdict(CroppingDict()),
                blur_contact_map_dict=asdict(BlurContactMapDict()),
                global_proprio_input_dict=asdict(ProprioceptionInputDict()),
                local_proprio_input_dict=asdict(ProprioceptionInputDict()),
                add_noise_dict=asdict(AddNoiseDict()),
                proprio_history_plotting_dict=asdict(ProprioceptionHistoryDict()), 
                is_anno_local = False, is_anno_global = False,
                viz_dataset=False, viz_global_contact=False,
    ):
        if episode_idx is not None:
            self.lazy_load = False
            self.episode_idx = episode_idx
            self.store_images = store_images
        else:
            self.lazy_load = True
            self.store_images = False
        
        self.viz_dataset = viz_dataset
        self.viz_global_contact = viz_global_contact
        self.is_real_dataset = is_real_dataset
        if is_real_dataset:
            assert downsampled_images == False, 'downsampled_images must be False for real dataset'
        self.downsampled_images = downsampled_images
        self.proprio_filter_dict = proprio_filter_dict
        self.l515 = l515
        self.per_episode_timestamp_filename = 'timestamp_series.pkl'

        if model_type in ['local', 'conditional_local']:
            assert local_proprio_input_dict is not None, 'local_proprio_input_dict must be provided for local model'
            self.local_proprio_input_dict = copy.deepcopy(local_proprio_input_dict)
        elif model_type == 'global':
            assert global_proprio_input_dict is not None, 'global_proprio_input_dict must be provided for global model'
            self.global_proprio_input_dict = copy.deepcopy(global_proprio_input_dict)
        elif model_type == 'joint':
            assert local_proprio_input_dict is not None and global_proprio_input_dict is not None, 'local_proprio_input_dict and global_proprio_input_dict must be provided for joint model'
            self.local_proprio_input_dict = copy.deepcopy(local_proprio_input_dict)
            self.global_proprio_input_dict = copy.deepcopy(global_proprio_input_dict)
        else:
            if model_type != None:
                raise RuntimeError('model_type must be local, conditional_local, global, joint, or none')
        self.model_type = model_type

        if compensate_obj_gravity_dict is not None:
            self.compensate_obj_gravity_dict = copy.deepcopy(compensate_obj_gravity_dict)
            self.real_EE_T_objCoM = np.diag([1.,1.,1.,1.])
            self.real_EE_T_objCoM[0, -1] = self.compensate_obj_gravity_dict['EE_pos_x_objCoM']
            self.real_EE_T_objCoM[1, -1] = self.compensate_obj_gravity_dict['EE_pos_y_objCoM']
            self.real_EE_T_objCoM[2, -1] = self.compensate_obj_gravity_dict['EE_pos_z_objCoM']
        else:
            self.compensate_obj_gravity_dict = {'enable': False}
        
        self.add_noise_dict = add_noise_dict
        
        if optical_flow_dict is not None:
            self.optical_flow_dict = copy.deepcopy(optical_flow_dict)
            if self.optical_flow_dict['use_image']:
                self.optical_flow_dict['dir_name'] = 'flow_image'
                self.optical_flow_dict['flow_key'] = 'flow_image'
            else:
                if self.downsampled_images:
                    self.optical_flow_dict['dir_name'] = 'flow_downsampled'

                else:
                    self.optical_flow_dict['dir_name'] = 'flow'
                self.optical_flow_dict['flow_key'] = 'flow'

            if self.is_real_dataset:
                self.optical_flow_dict['dir_name'] = self.optical_flow_dict['dir_name'] + '_aligned_to_depth' 
                # if self.l515:
                    # self.optical_flow_dict['dir_name'] = self.optical_flow_dict['dir_name'] + '_aligned_to_d455'
        else: 
            self.optical_flow_dict = {'enable': False}
        
        if context_frame_dict is not None:
            self.context_frame_dict = copy.deepcopy(context_frame_dict)
        else: 
            self.context_frame_dict = {'enable': False}

        if cropping_dict is not None:
            self.cropping_dict = copy.deepcopy(cropping_dict)
        else: 
            self.cropping_dict = {'enable': False}
        
        if ft_calibration_dict is not None:
            self.ft_calibration_dict = copy.deepcopy(ft_calibration_dict)
            if self.ft_calibration_dict['enable'] and self.ft_calibration_dict['use_history']:
                self.ft_calibration_dict['window_size'] = int(self.ft_calibration_dict['time_window']*self.ft_calibration_dict['sample_freq'])
        else:
            self.ft_calibration_dict = {'enable': False}
        
        if proprio_history_plotting_dict is None:
            self.proprio_history_plotting_dict = {'enable': False}
        else:
            self.proprio_history_plotting_dict = copy.deepcopy(proprio_history_plotting_dict)
        
        self.experiment_dset_path = experiment_dset
        self.experiment_dset_name = os.path.basename(os.path.normpath(self.experiment_dset_path))
        self.experiment_dset_path = os.path.expanduser(experiment_dset)
        self.experiment_data_dir_path = os.path.join(self.experiment_dset_path, 'episode_data')
        if not os.path.exists(self.experiment_dset_path):
            raise AssertionError(self.experiment_dset_path + ' does not exist!')
        if not os.path.exists(self.experiment_data_dir_path):
            raise AssertionError(self.experiment_data_dir_path + ' does not exist!')
        episode_data_dir_list = natsort.natsorted(os.listdir(self.experiment_data_dir_path))
        self.episode_data_dir_path_list = [os.path.join(self.experiment_data_dir_path, d) for d in episode_data_dir_list if os.path.isdir(os.path.join(self.experiment_data_dir_path, d))]

        self.im_time_offset = im_time_offset
        self.depth_intrinsic_filename = 'depth_K.npy'
        
        self.is_anno_local = is_anno_local
        self.is_anno_global = is_anno_global
        if self.is_real_dataset:
            # in real color is aligned to depth
            self.im_time_offset = 0.0
            self.depth_extrinsic_filename = 'D_tf_W.npy'
            self.depth_dir_name = 'depth'
            # if self.l515:
                # self.depth_dir_name = self.depth_dir_name + '_aligned_to_d455'
            self.color_dir_name = 'color_aligned_to_depth'
            # if self.l515:
            #     self.color_dir_name = self.color_dir_name + '_aligned_to_d455'
            self.color_extrinsic_base_filepath = os.path.join(self.color_dir_name, 'D_tf_W.npy')
            self.color_intrinsic_base_filepath = os.path.join(self.color_dir_name, 'depth_K.npy')

            # check if dataset is annotated
            if is_anno_local:
                if self.lazy_load:
                    if not os.path.exists(os.path.join(self.experiment_data_dir_path, 'annotated_local')):
                        raise AssertionError('annotated_local does not exist!')
                else:
                    # check in the specific episode folders
                    if not os.path.exists(os.path.join(self.episode_data_dir_path_list[episode_idx], 'is_anno_local')):
                        raise AssertionError('annotated_local does not exist!')
            if is_anno_global:
                if self.lazy_load:
                    if not os.path.exists(os.path.join(self.experiment_data_dir_path, 'annotated_global')):
                        raise AssertionError('annotated_global does not exist!')
                else:
                    # check in the specific episode folders
                    if not os.path.exists(os.path.join(self.episode_data_dir_path_list[episode_idx], 'is_anno_global')):
                        raise AssertionError('annotated_global does not exist!')
            # Annotation label folder
            self.is_anno_local_dir = 'is_anno_local'
            self.is_anno_global_dir = 'is_anno_global'
        else: # sim dataset
            assert not is_anno_local and not is_anno_global, 'Annotation is not supported for sim dataset' 
            if self.l515:
                # in sim color is aligned with depth
                self.depth_extrinsic_filename = 'D_tf_W.npy'
                if self.downsampled_images:
                    print('downsampled images turned on!')
                    self.depth_dir_name = 'depth_downsampled'
                    self.color_dir_name = 'color_aligned_to_depth_downsampled'
                else:
                    self.depth_dir_name = 'depth'
                    self.color_dir_name = 'color_aligned_to_depth'
            else:
                # in sim depth is aligned to color
                self.depth_extrinsic_filename = 'C_tf_W.npy'
                self.depth_dir_name = 'aligned_depth_to_color'
                self.color_dir_name = 'color'
            self.color_extrinsic_base_filepath = os.path.join(self.color_dir_name, 'C_tf_W.npy')
            self.color_intrinsic_base_filepath = os.path.join(self.color_dir_name, 'color_K.npy')
        
        self.max_depth_clip = max_depth_clip
        self.max_num_contact = max_num_contact
        self.contact_persist_time = contact_persist_time
        if blur_contact_map_dict is None:
            self.blur_contact_prob_dict = {'enable': False} 
        else:
            self.blur_contact_prob_dict = copy.deepcopy(blur_contact_map_dict) 
        ## for now hardcode the main topic and topics of interest...
        ## WINDOW SIZE IS THE SAME FOR ALL STREAMS FOR NOW
        # max_window = max(list(time_window_dict.values()))
        self.window_size = window_size
        self.grasped_obj_name = grasped_obj_name
        self.centered = centered 
        # if episode_idx is provided, we can store the episode into memory instead of lazy loading

        
        if self.lazy_load:
            # load the image timestamp and episode multiindex dataframe
            self.depth_episode_timestamp_series = pd.read_pickle(os.path.join(self.experiment_data_dir_path, 'depth_episode_timestamp_series.pkl'))
            self.color_episode_timestamp_series = pd.read_pickle(os.path.join(self.experiment_data_dir_path, 'color_episode_timestamp_series.pkl'))
            episode_idx = 0
        else:
            self.depth_episode_timestamp_series = pd.read_pickle(os.path.join(self.experiment_data_dir_path, self.episode_data_dir_path_list[episode_idx], self.depth_dir_name, 'timestamp_series.pkl'))
            self.color_episode_timestamp_series = pd.read_pickle(os.path.join(self.experiment_data_dir_path, self.episode_data_dir_path_list[episode_idx], self.color_dir_name, 'timestamp_series.pkl'))
           
        self.main_num_msgs = len(self.depth_episode_timestamp_series)
        if self.window_size == -1:
            self._len = 1
        else:
            self._len = self.main_num_msgs 

        self.base_proprio_topic = '/panda/franka_state_controller_custom/franka_states/'

        self.main_num_msgs = len(self.depth_episode_timestamp_series)
        assert self.main_num_msgs > 1, "must be at least one number of msgs!" 
        assert self.main_num_msgs >= self.window_size, "number of msgs must be geq window size!" 
      
        # check first episode index's image to get the image shape
        first_episode_dir_path = self.episode_data_dir_path_list[episode_idx]
        # use glob and natural sort to get the first image name
        first_episode_first_depth_name = os.path.basename(natsort.natsorted(glob.glob(os.path.join(first_episode_dir_path, self.depth_dir_name, '*.png')))[0])
        assert first_episode_first_depth_name is not None, "first episode first depth name is None!"
        # assert that image start with 0
        assert first_episode_first_depth_name.split('-')[0] == '0', "first episode first depth name does not start with 0!"
        # first_episode_first_depth_name = os.listdir(os.path.join(first_episode_dir_path, self.depth_dir_name))[episode_idx]
        self.depth_shape = cv2.imread(os.path.join(first_episode_dir_path, self.depth_dir_name, first_episode_first_depth_name)).shape[:2]
        # use glob and natural sort to get the first color image name
        first_episode_first_color_name = os.path.basename(natsort.natsorted(glob.glob(os.path.join(first_episode_dir_path, self.color_dir_name, '*.png')))[0])
        assert first_episode_first_color_name is not None, "first episode first color name is None!"
        # assert that image start with 0
        assert first_episode_first_color_name.split('-')[0] == '0', "first episode first color name does not start with 0!"
        self.color_shape = cv2.imread(os.path.join(first_episode_dir_path, self.color_dir_name, first_episode_first_color_name)).shape[:2]

        self.joint_position_topics = []
        for i in range(7):
            topic = self.base_proprio_topic + 'q/' + str(i)
            self.joint_position_topics.append(topic)

        self.joint_velocity_topics = []
        for i in range(7):
            topic = self.base_proprio_topic + 'dq/' + str(i)
            self.joint_velocity_topics.append(topic)
        
        self.external_joint_torques_topics = []
        for i in range(7):
            topic = self.base_proprio_topic + 'tau_ext_hat_filtered/' + str(i)
            self.external_joint_torques_topics.append(topic)
        
        self.measured_joint_torques_topics = []
        for i in range(7):
            topic = self.base_proprio_topic + 'tau_J/' + str(i)
            self.measured_joint_torques_topics.append(topic)
        
        self.desired_joint_torques_topics = []
        for i in range(7):
            topic = self.base_proprio_topic + 'tau_J_d/' + str(i)
            self.desired_joint_torques_topics.append(topic)

        self.pose_topics = []
        for i in range(16):
            topic = self.base_proprio_topic + 'O_T_EE/' + str(i)
            self.pose_topics.append(topic)

        # to add EE vel??
        self.EE_vel_topics = []
        for i in range(6):
            topic = self.base_proprio_topic + 'O_dP_EE/' + str(i)
            self.EE_vel_topics.append(topic)
        # O_dP_EE # EE vel computed as J*dq

        self.wrench_topics = []
        # If we use filtered data in pre-processed proprio, this should be another topic 'O_F_ext_hat_K_filtered'
        for i in range(6):
            topic = self.base_proprio_topic + 'O_F_ext_hat_K/' + str(i) #change from K_F_ext_hat_K to match other modality frames
            self.wrench_topics.append(topic)

        # TODO consider adding a separate topic for the filtered force data
        # If we use filtered force and ft data
        if self.is_real_dataset:
            if self.proprio_filter_dict['enable']:
                filter_name = self.proprio_filter_dict['filter_name'] + '_'
                self.wrench_topics = []
                for i in range(6):
                    topic = self.base_proprio_topic + filter_name + 'O_F_ext_hat_K/' + str(i) #change from K_F_ext_hat_K to match other modality frames
                    self.wrench_topics.append(topic)
                
                self.joint_velocity_topics = []
                for i in range(7):
                    topic = self.base_proprio_topic + filter_name + 'dq/' + str(i)
                    self.joint_velocity_topics.append(topic)
                
                self.external_joint_torques_topics = []
                for i in range(7):
                    topic = self.base_proprio_topic + filter_name + 'tau_ext_hat_filtered/' + str(i)
                    self.external_joint_torques_topics.append(topic)
                
                self.measured_joint_torques_topics = []
                for i in range(7):
                    topic = self.base_proprio_topic + filter_name + 'tau_J/' + str(i)
                    self.measured_joint_torques_topics.append(topic)
                
                self.desired_joint_torques_topics = []
                for i in range(7):
                    topic = self.base_proprio_topic + filter_name + 'tau_J_d/' + str(i)
                    self.desired_joint_torques_topics.append(topic)

        
        # Only exist in pre-processed proprio            
        if self.is_real_dataset:
            self.real_global_start_end_topic = 'real_global_contact_start_end'
            self.real_global_continuous_topic = 'real_global_contact_continous'

        if im_resize is not None:
            if self.depth_shape == im_resize:
                self.im_resize = None
                self.im_size = self.depth_shape
            else:
                self.im_resize = im_resize #HxW
                self.im_size = im_resize
        else:
            self.im_resize = None
            self.im_size = self.depth_shape
        self.im_to_tensor = torchvision.transforms.ToTensor() 
        self.tensor_to_float32 = torchvision.transforms.ConvertImageDtype(torch.float32)

        # assume contact frequency is same across all episodes
        if not self.is_real_dataset: #contact data only avail in sim!
            # get the info dict of first episode
            first_episode_dir_path = self.episode_data_dir_path_list[0]
            info_dict = pickle.load(open(os.path.join(first_episode_dir_path, 'info_dict.pkl'), 'rb'))
            for topic_dict in info_dict['topics']:
                if 'contact_data' in topic_dict['topic']:
                    # if no frequency information (sometimes doesnt store), just hardcode default to 500hz
                    if 'frequency' not in topic_dict:
                        self.contact_freq = 500.
                    else:
                        self.contact_freq = topic_dict['frequency']
                    break
            self.contact_dt = 1./self.contact_freq
        
        if not self.lazy_load:
            self.load_episode_data(self.episode_idx)

        if self.lazy_load:
            print('succesfully read ' + experiment_dset)
        else:
            # also print the episode idx data name
            print('succesfully read ' + experiment_dset + ' episode ' + str(self.episode_idx))

    def __len__(self):
        return self._len

    def load_episode_data(self, episode_idx):
        episode_data_dict = {}
        episode_dir_path = self.episode_data_dir_path_list[episode_idx]

        # get the dataframes given an episode index
        if not self.is_real_dataset: #contact data only avail in sim!
            # check first if episode has no contact file
            if not os.path.exists(os.path.join(episode_dir_path, 'no_contact')):
                contact_df = pd.read_pickle(os.path.join(episode_dir_path, 'contact_df.pkl'))
            else:
                contact_df = None 
        else:
            contact_df = None
        proprio_df = self.get_proprio_df(episode_idx)

        grasped_object_params_dict = {}
        # if self.compensate_obj_gravity_dict['enable']:
        grasped_object_params_dict['EE_T_objCoM'] = self.real_EE_T_objCoM    
        if self.is_real_dataset:
            # Assumption on real_EE_T_objCoM (identity for rotation and a displacement on +z axis)
            first_several_wrench_np = np.array(proprio_df.iloc[self.compensate_obj_gravity_dict['start_idx']:self.compensate_obj_gravity_dict['start_idx']+self.compensate_obj_gravity_dict['num_idxs']][self.wrench_topics].values)  # Get the first 10 data for estimating the mass of the object
            mass = np.mean(first_several_wrench_np[:, 2]) / 9.81                                 
            # Get average Fz and devided by g=9.81
            grasped_object_params_dict['mass'] = mass 
        else:
            grasped_object_params_dict_gt = pd.read_pickle(os.path.join(episode_dir_path, 'grasped_object_params.pkl')) #'EE_T_objCoM', 'mass'
            grasped_object_params_dict['mass'] = grasped_object_params_dict_gt['mass']

        # get the depth camera params
        depth_tf_world = np.load(os.path.join(episode_dir_path, self.depth_dir_name, self.depth_extrinsic_filename))
        K_depth = np.load(os.path.join(episode_dir_path, self.depth_dir_name, self.depth_intrinsic_filename))
        # get the color camera params
        color_tf_world = np.load(os.path.join(episode_dir_path, self.color_extrinsic_base_filepath))
        K_color = np.load(os.path.join(episode_dir_path, self.color_intrinsic_base_filepath))

        if self.lazy_load:
            episode_data_dict['contact_df'] = contact_df
            episode_data_dict['proprio_df'] = proprio_df
            episode_data_dict['grasped_object_params_dict'] = grasped_object_params_dict

            episode_data_dict['depth_tf_world'] = depth_tf_world
            episode_data_dict['K_depth'] = K_depth

            episode_data_dict['color_tf_world'] = color_tf_world
            episode_data_dict['K_color'] = K_color

            return episode_data_dict
        else:
            self.episode_dir_path = episode_dir_path

            self.contact_df = contact_df
            self.proprio_df = proprio_df
            self.grasped_object_params_dict = grasped_object_params_dict
            
            self.depth_tf_world = depth_tf_world
            self.K_depth = K_depth

            self.color_tf_world = color_tf_world
            self.K_color = K_color

            self.episode_depth_times = self.depth_episode_timestamp_series.values - self.im_time_offset

            self.episode_color_times = self.color_episode_timestamp_series.values - self.im_time_offset
            self.episode_color_path_list = natsort.natsorted(glob.glob(os.path.join(episode_dir_path, self.color_dir_name, '*.png')))

            # if self.store_images:
            episode_depth_path_list = natsort.natsorted(glob.glob(os.path.join(episode_dir_path, self.depth_dir_name, '*.png')))
            # load all images in the episode to memory
            ## get images
            images = skimage.io.imread_collection(episode_depth_path_list) # H x W x T
            #resize 
            if self.im_resize is None: # if im resize is none
                images_np = np.array([img for img in images]) # now become T x H x W
            else:
                images_np = np.array([cv2.resize(img, (self.im_resize[1], self.im_resize[0]), interpolation=cv2.INTER_CUBIC) for img in images]) #preserves uint16 
            assert images_np.dtype == np.uint16, 'depth images not np uint16!'
            self.depth_images_np = images_np
    
    def __getitem__(self, idx_accessed):
        if idx_accessed >= self._len:
            raise AssertionError('index out of range')
        
        if self.lazy_load:
            # get the episode index of the accessed index
            episode_idx = self.depth_episode_timestamp_series.index[idx_accessed]
            episode_data_dict = self.load_episode_data(episode_idx)
            episode_dir_path = self.episode_data_dir_path_list[episode_idx]

            proprio_df = episode_data_dict['proprio_df']
            contact_df = episode_data_dict['contact_df']
            grasped_object_params_dict = episode_data_dict['grasped_object_params_dict']
            depth_tf_world = episode_data_dict['depth_tf_world']
            K_depth = episode_data_dict['K_depth']
            color_tf_world = episode_data_dict['color_tf_world']
            K_color = episode_data_dict['K_color']
            
            episode_depth_times = self.depth_episode_timestamp_series.loc[episode_idx].values - self.im_time_offset
            episode_depth_path_list = natsort.natsorted(glob.glob(os.path.join(episode_dir_path, self.depth_dir_name, '*.png')))

            # get image given an episode index
            # get the index of the accessed index in the episode
            idx_in_episode = idx_accessed - self.depth_episode_timestamp_series.index.get_indexer_for([episode_idx])[0]
       
            # get color image paths given an episode index
            episode_color_times = self.color_episode_timestamp_series.loc[episode_idx].values - self.im_time_offset

        else:
            episode_dir_path = self.episode_dir_path
            episode_idx = self.episode_idx
            idx_in_episode = idx_accessed

            episode_depth_times = self.episode_depth_times
            episode_color_times = self.episode_color_times
            proprio_df = self.proprio_df
            contact_df = self.contact_df
            grasped_object_params_dict = self.grasped_object_params_dict
            depth_tf_world = self.depth_tf_world
            K_depth = self.K_depth
            color_tf_world = self.color_tf_world
            K_color = self.K_color

            episode_dir_path_annotation = self.episode_data_dir_path_list[episode_idx]
            episode_depth_path_list = natsort.natsorted(glob.glob(os.path.join(episode_dir_path_annotation, self.depth_dir_name, '*.png')))
            
        ## indexing for non-overlapping indexing
        if self.window_size == -1:
            start_idx = 0
            end_idx = self.main_num_msgs
        else:
            start_idx = max(0, idx_in_episode-self.window_size + 1)
            end_idx = min(len(episode_depth_times), idx_in_episode+1)
        
        if self.lazy_load:
            ## get images
            images = skimage.io.imread_collection(episode_depth_path_list[start_idx:end_idx]) # H x W x T
            #resize 
            if self.im_resize is None: # if im resize is none
                images_np = np.array([img for img in images]) # now become T x H x W
            else:
                images_np = np.array([cv2.resize(img, (self.im_resize[1], self.im_resize[0]), interpolation=cv2.INTER_CUBIC) for img in images]) #preserves uint16 
            assert images_np.dtype == np.uint16, 'depth images not np uint16!'
        else:
            images_np = self.depth_images_np[start_idx:end_idx]

        # converting depth from mm to m
        # images_np = (images_np*1.0e-3)
        images_normalized = depth_png_to_map(images_np, self.max_depth_clip)
        # expects np array of dim N x H x W x C
        
        im_times = episode_depth_times[start_idx:end_idx]
        # im_max_vals = np.amax(images_np, axis=(-2, -1))

        # get color
        ## need to sort by nearest index here because color and depth timestamps arent aligned!
        # TODO for visualization datasets, actually return the color image instead of the path
        episode_color_path_list = natsort.natsorted(glob.glob(os.path.join(episode_dir_path, self.color_dir_name, '*.png')))
        nrst_color_idxs = self.get_nearest_idxs(im_times, episode_color_times)
        # set max and min on the nrst_color_idxs to make sure they are within the indices of the length of episode_color_path_list
        nrst_color_idxs = np.clip(nrst_color_idxs, 0, len(episode_color_path_list)-1)
        color_im_paths = [episode_color_path_list[idx] for idx in nrst_color_idxs]

        ## get proprioceptive info
        ## get current pose, wrench,
        nrst_proprio_idxs = self.get_nearest_idxs(im_times, proprio_df.index)
        nrst_tf_np = np.array(proprio_df.iloc[nrst_proprio_idxs[0]][self.pose_topics].values)
        nrst_wrench_np = np.array(proprio_df.iloc[nrst_proprio_idxs[0]][self.wrench_topics].values)
        if self.compensate_obj_gravity_dict['enable']:
            O_T_EE = np.reshape(nrst_tf_np, (-1,4,4), order='F')   # 1,4,4
            EEO_grav_wrench = self.get_EEO_grav_wrench(O_T_EE, grasped_object_params_dict["EE_T_objCoM"], grasped_object_params_dict["mass"])
            nrst_wrench_np = (nrst_wrench_np + EEO_grav_wrench).astype(float) #broadcasting grav_wrench from (6,) to (T,6) 

        # nrst_desired_torques_np = np.array(proprio_df.iloc[nrst_proprio_idxs[0]][self.joint_desired_torques_topics].values)
        nrst_pose_np = self.affine_tf_to_pose(nrst_tf_np)
        pose_pxl_np = self.point_proj(K_depth, nrst_pose_np[:3], C_tf_W = depth_tf_world)
        if self.im_resize is not None:
            ## THIS ONLY WORKS BECAUSE THE RESIZE IS BY THE SAME FACTOR ON BOTH H AND W! AND ALSO THE RESIZE IS MADE TO BE A NICE INT FACTOR (4 in this case...)
            ## TODO fix this resize reprojection by scaling the projection matrix...
            pose_pxl_np = ((self.im_resize[0]/self.depth_shape[0])*pose_pxl_np)
        pose_pxl_np = pose_pxl_np.astype(int)

        return_dict = {
        'grasped_object_mass': np.float32(grasped_object_params_dict['mass']),
        'current_pose': nrst_pose_np.astype(np.float32), # B x 7
        'current_pose_pxl': pose_pxl_np, # B x T x 2
        'current_wrench': nrst_wrench_np.astype(np.float32), # B x 6
        # 'joint_desired_torques': nrst_desired_torques_np.astype(np.float32), # B x T x 7
        'color_paths': color_im_paths, # T dim list of B dim tuples
        'im_times': im_times, # B x T
        'depth_intrinsics': K_depth, # B x 3 x 3
        'depth_tf_world': depth_tf_world, 
        'color_intrinsics': K_color, # B x 3 x 3
        'color_tf_world': color_tf_world,
        'len_samples': self._len,
        'idx_accessed': idx_accessed,
        'within_episode_idx': idx_in_episode,
        'episode': episode_idx
        }

        # get length of currently accessed episode
        episode_length = len(episode_depth_times)
        # return whether this item is the last in the episode
        return_dict.update({'is_last_in_episode': idx_in_episode == (episode_length-1)})

        ## RETURN IMAGE DATA 
        if self.model_type != 'global' or self.viz_dataset:

            # only need to return full images if we arent cropping or if we're visualizing this dataset
            if not self.cropping_dict['enable'] or self.viz_dataset:
                return_dict.update({'images_tensor': images_normalized.astype(np.float32),}) # need to convert to dim T x H x W (im using Color channel as time...)

            # TODO consider moving cropping to datamodule
            if self.cropping_dict['enable']:
                cropping_output_tuple = self.cropping(self.im_size, self.cropping_dict,images_normalized, pose_pxl_np)
                if len(cropping_output_tuple) == 2:
                    cropped_images, bb_top_left_coordinate = cropping_output_tuple
                elif len(cropping_output_tuple) == 3:
                    cropped_images, bb_top_left_coordinate, _ = cropping_output_tuple
                    # log the dataset, episode and index so I know where the error is
                    print(f'Error in cropping! Dataset: {self.experiment_dset_name}, Episode: {episode_idx}, Index in episode: {idx_in_episode}')
                
                cropped_images[cropped_images == 0] = 1.0 
                return_dict.update({
                    'cropped_images_tensor': cropped_images.astype(np.float32),
                    'bb_top_left_coordinate': bb_top_left_coordinate,
                })

            # get optical flow
            if self.optical_flow_dict['enable']:
                # get optical flow image paths given an episode index
                flow_dir_path = os.path.join(episode_dir_path, self.optical_flow_dict['dir_name'])
                assert os.path.exists(flow_dir_path), f'Optical flow directory {flow_dir_path} does not exist!'
                episode_optical_flow_path_list = natsort.natsorted(glob.glob(os.path.join(flow_dir_path, '*.png' if self.optical_flow_dict['use_image'] else '*.pkl')))
                # theres one less optical flow image than color image due to pairing
                # the minus one aligns such that flow corresponding to second color image is the first flow image
                # TODO, replace this maximum with zero flow for the first image
                nrst_flow_idxs = np.maximum(np.array(nrst_color_idxs) - 1, 0)
                flow_paths = [episode_optical_flow_path_list[idx] for idx in nrst_flow_idxs]
                if self.optical_flow_dict['use_image']:
                    flow = skimage.io.imread_collection(flow_paths) # T x H x W x 3
                    if self.im_resize is None: # if im resize is none
                        flow_np = np.array([img for img in flow]) # now become T x H x W x C
                    else:
                        flow_np = np.array([cv2.resize(img, (self.im_resize[1], self.im_resize[0]), interpolation=cv2.INTER_CUBIC) for img in flow]) 
                    if self.is_real_dataset and self.optical_flow_dict['fill_holes']:
                        flow_np = self.fill_flow_image_holes(flow_np)
                    if self.optical_flow_dict['normalize']:
                        flow_np = flow_np / 255.0
                else:
                    flow = [pickle.load(open(path, 'rb')) for path in flow_paths] # T x H x W x 2
                    if self.im_resize is None: # if im resize is none
                        flow_np = np.array(flow) # now become T x H x W x 2
                    else:
                        flow_np = np.array([cv2.resize(img, (self.im_resize[1], self.im_resize[0]), interpolation=cv2.INTER_CUBIC) for img in flow]) 
                    if self.optical_flow_dict['normalize']:
                        # TODO fix this so its actually normalizing... Need to clip before normalizing in the 2-norm sense
                        max_flow_norm = self.optical_flow_dict['max_flow_norm']
                        flow_np = np.clip(flow_np, -max_flow_norm, max_flow_norm) / self.optical_flow_dict['max_flow_norm']
                flow_np = flow_np[0] # change T x H x W x C to H x W x C
                # permute dimensions from H x W x C to C x H x W
                flow_np = np.transpose(flow_np, (2, 0, 1)) 

                # only return these if not cropping or if we need to visualize the dataset
                if not self.cropping_dict['enable'] or self.viz_dataset:
                    return_dict[self.optical_flow_dict['flow_key']] = flow_np.astype(np.float32) # key either flow or flow_image
                    return_dict[self.optical_flow_dict['flow_key'] + '_paths'] = flow_paths
                    
                if self.cropping_dict['enable']:
                    cropped_flow_np = np.zeros((flow_np.shape[0], self.cropping_dict['bb_height'], self.cropping_dict['bb_width']))
                    for i in range(flow_np.shape[0]):
                        cropping_output_tuple = self.cropping(self.im_size,self.cropping_dict, flow_np[i][np.newaxis, :, :], pose_pxl_np)
                        if len(cropping_output_tuple) == 2:
                            cp, _ = cropping_output_tuple
                        elif len(cropping_output_tuple) == 3:
                            cp, _, _ = cropping_output_tuple
                            # log the dataset, episode and index so I know where the error is
                            print(f'Error in flow cropping! Dataset: {self.experiment_dset_name}, Episode: {episode_idx}, Index in episode: {idx_in_episode}')
                        cropped_flow_np[i] = cp[0]
                    
                    return_dict.update({
                    'cropped_optical_flow_images_tensor':cropped_flow_np              
                    })

            if self.context_frame_dict['enable']:
                ## get images 
                if self.lazy_load:
                    first_epi_image = skimage.io.imread_collection(episode_depth_path_list[0])
                    #resize 
                    if self.im_resize is None: # if im resize is none
                        first_epi_images_np = np.array(first_epi_image) # now become T=1 x H x W
                    else:
                        first_epi_images_np = np.array([cv2.resize(img, (self.im_resize[1], self.im_resize[0]), interpolation=cv2.INTER_CUBIC) for img in first_epi_image]) #preserves uint16 
                        # first_epi_images_np = cv2.resize(first_epi_image, (self.im_resize[1], self.im_resize[0]), interpolation=cv2.INTER_CUBIC)  #preserves uint16 
                    assert first_epi_images_np.dtype == np.uint16, 'depth images not np uint16!'
                else:
                    # index by slice so that we still get (1, H, W) shape instead of (H, W)
                    first_epi_images_np = self.depth_images_np[0:1]
                first_epi_images_np = first_epi_images_np.astype(np.float32) / 1000.0
                # exchange all zeros to the max depth value 
                # in sim zeros are the background
                # in real zeros are invalid depth
                first_epi_images_np[first_epi_images_np == 0] = self.max_depth_clip
                first_epi_images_clipped = np.clip(first_epi_images_np, 0, self.max_depth_clip)
                first_epi_images_normalized = first_epi_images_clipped / self.max_depth_clip
                # expects np array of dim H x W x C
                
                first_epi_im_times = episode_depth_times[0]

                ## get proprioceptive info
                ## get current pose, wrench,
                first_epi_nrst_proprio_idxs = self.get_nearest_idxs([first_epi_im_times], proprio_df.index)
                first_epi_nrst_tf_np = np.array(proprio_df.iloc[first_epi_nrst_proprio_idxs[0]][self.pose_topics].values)
                first_epi_nrst_pose_np = self.affine_tf_to_pose(first_epi_nrst_tf_np)
                first_epi_pose_pxl_np = self.point_proj(K_depth, first_epi_nrst_pose_np[:3], C_tf_W = depth_tf_world)
                # first_epi_nrst_wrench_np = np.array(proprio_df.iloc[first_epi_nrst_proprio_idxs[0]][self.wrench_topics].values)
                # nrst_desired_torques_np = np.array(proprio_df.iloc[nrst_proprio_idxs[0]][self.joint_desired_torques_topics].values)
                if self.im_resize is not None:
                    ## THIS ONLY WORKS BECAUSE THE RESIZE IS BY THE SAME FACTOR ON BOTH H AND W! AND ALSO THE RESIZE IS MADE TO BE A NICE INT FACTOR (4 in this case...)
                    ## TODO fix this resize reprojection by scaling the projection matrix...
                    first_epi_pose_pxl_np = ((self.im_resize[0]/self.depth_shape[0])*first_epi_pose_pxl_np)
                first_epi_pose_pxl_np = first_epi_pose_pxl_np.astype(int)

                return_dict.update({'first_epi_pose_pxl': first_epi_pose_pxl_np.astype(np.float32),}) 
                if not self.cropping_dict['enable'] or self.viz_dataset:
                    return_dict.update({
                    'first_epi_images_tensor': first_epi_images_normalized.astype(np.float32),
                    'context_color_path': episode_color_path_list[0],
                    # 'first_epi_im_times':first_epi_im_times,
                    # 'first_epi_wrench': first_epi_nrst_wrench_np.astype(np.float32), # this will be useful to calibrate out the weight of the object
                    # 'first_epi_pose': first_epi_nrst_pose_np.astype(np.float32), 
                    })
                
                if self.cropping_dict['enable']:
                    cropping_output_tuple = self.cropping(self.im_size,self.cropping_dict,first_epi_images_normalized, first_epi_pose_pxl_np)
                    if len(cropping_output_tuple) == 2:
                        cropped_first_epi, context_bb_top_left_coordinate = cropping_output_tuple
                    else:
                        cropped_first_epi, context_bb_top_left_coordinate, _ = cropping_output_tuple
                        print(f'Error in context frame cropping! Dataset: {self.experiment_dset_name}, Episode: {episode_idx}, Index in episode: {idx_in_episode}')
                    cropped_first_epi[cropped_first_epi == 0] = 1.0
                    return_dict.update({
                    'cropped_first_epi_images_tensor':cropped_first_epi.astype(np.float32), 
                    'context_bb_top_left_coordinate': context_bb_top_left_coordinate
                    })

        # get local proprioceptive info
        if self.model_type in ['local', 'conditional_local', 'joint']:
            # if no local proprioceptive inputs, then just return an empty dict
            if not (self.local_proprio_input_dict['external_wrench'] or self.local_proprio_input_dict['EE_pose'] or self.local_proprio_input_dict['desired_joint_torques']):
                local_proprio_dict = {}
            else:
                local_proprio_dict = self.get_proprioceptive_history(times_list = im_times, key_prefix='local_',
                                                                            proprio_input_dict=self.local_proprio_input_dict,
                                                                            proprio_df = proprio_df, calib_grav = self.compensate_obj_gravity_dict, objpara = grasped_object_params_dict,
                                                                            add_noise_dict = self.add_noise_dict)
            return_dict.update(local_proprio_dict)
        
        if self.model_type in ['global', 'joint']:
            global_proprio_dict = self.get_proprioceptive_history(times_list = im_times, key_prefix='global_',
                                                                            proprio_input_dict=self.global_proprio_input_dict, 
                                                                            proprio_df = proprio_df,  calib_grav = self.compensate_obj_gravity_dict, objpara = grasped_object_params_dict,
                                                                            add_noise_dict = self.add_noise_dict)
            return_dict.update(global_proprio_dict)
        
        ## get contact info
        if not self.is_real_dataset: #contact data only avail in sim!
            # output the contact location prob map
            # output the contact forces map
            if self.im_resize is None:
                contact_prob_map = np.zeros(self.im_size)
                contact_prob_map_blurred = np.zeros(self.im_size)
            else: 
                contact_prob_map = np.zeros(self.im_resize)
                contact_prob_map_blurred = np.zeros(self.im_resize)
            
            # get contact data
            if not contact_df is None:
                # get target contact label
                contact_time_label = self.get_contact_time_label(im_times, centered=self.centered)

                contact_dict = self.get_contact_data(contact_time_label, self.contact_dt, contact_df)
                contact_pxls = []

                if contact_dict['num_contacts'] != 0:
                    global_contact = 1.
                    for idx in range(contact_dict['num_contacts']): 
                        contact_pos = contact_dict['positions'][idx]

                        contact_pos_prj = self.point_proj(K_depth, contact_pos, depth_tf_world)

                        if self.im_resize is None: # if im_resize is None
                            contact_pxls.append(contact_pos_prj)
                            contact_prob_map[contact_pos_prj[1], contact_pos_prj[0]] = 1.0
                        else:
                            ## THIS ONLY WORKS BECAUSE THE RESIZE IS BY THE SAME FACTOR ON BOTH H AND W! AND ALSO THE RESIZE IS MADE TO BE A NICE INT FACTOR (4 in this case...)
                            ## TODO fix this resize reprojection by scaling the projection matrix...
                            contact_pos_prj_resized = ((self.im_resize[0]/self.depth_shape[0])*contact_pos_prj).astype(int)
                            contact_pxls.append(contact_pos_prj_resized)
                            contact_prob_map[contact_pos_prj_resized[1], contact_pos_prj_resized[0]] = 1.0

                    # now pad each array to fit into max num contacts

                    # 2D np array of num contact x feature dim
                    # for some really odd reason these arrays were being converted to type object... had to explicitly convert them to float dtype...
                    num_pad_contacts = self.max_num_contact - contact_dict['num_contacts']
                    assert num_pad_contacts >= 0, 'number of contacts is bigger than max padding!!!'

                    padded_contact_positions = np.pad(contact_dict['positions'], ((0, num_pad_contacts), (0,0)), mode='constant', constant_values=(np.nan))
                    ## need to convert to float bc np.nan is a float and need to pad with it 
                    padded_contact_pxls_flt = np.pad(np.array(contact_pxls, dtype=float), ((0, num_pad_contacts), (0,0)), mode='constant', constant_values=(np.nan))
                    # create a padded list of strings
                    grasped_object_geometry = contact_dict['grasped_object_geometry'] 
                    contact_object_geometry = contact_dict['contact_object_geometry'] 
               
                    if self.blur_contact_prob_dict['enable']:
                        contact_prob_map_blurred = cv2.GaussianBlur(contact_prob_map, (self.blur_contact_prob_dict['kernel_size'], self.blur_contact_prob_dict['kernel_size']), self.blur_contact_prob_dict['sigma'])
                else:
                    global_contact = 0.
                    padded_contact_positions = np.full((self.max_num_contact, 3), np.nan)
                    padded_contact_pxls_flt = np.full((self.max_num_contact, 2), np.nan)
                    grasped_object_geometry = '' #* self.max_num_contact
                    contact_object_geometry = '' #* self.max_num_contact
            else:
                padded_contact_positions = np.full((self.max_num_contact, 3), np.nan)
                padded_contact_pxls_flt = np.full((self.max_num_contact, 2), np.nan)
                grasped_object_geometry = '' #* self.max_num_contact
                contact_object_geometry = '' #* self.max_num_contact
                contact_dict = {}
                contact_dict['num_contacts'] = 0
                global_contact = 0.
                contact_dict['time'] = np.nan
                contact_dict['time_diff'] = np.nan
           
            return_dict.update({
            'global_contact': np.float32(global_contact), # B
            'contact_prob_map': contact_prob_map, # B x H x W
            'contact_positions': padded_contact_positions, # B x max_num_contact x 3
            'num_contacts': contact_dict['num_contacts'], # B
            'contact_pxls_float': padded_contact_pxls_flt, # need to apply astype(int) after loading
            'contact_time': contact_dict['time'], # B
            'contact_time_diff': contact_dict['time_diff'],
            'grasped_object_geometry': grasped_object_geometry,
            'contact_object_geometry': contact_object_geometry})
            
            if self.blur_contact_prob_dict['enable']:
                return_dict['contact_prob_map_blurred'] = contact_prob_map_blurred
            
            if self.cropping_dict['enable']:
                cropping_output_tuple = self.cropping(self.im_size,self.cropping_dict,contact_prob_map, pose_pxl_np)
                if len(cropping_output_tuple) == 2:
                    cropped_contact_prob_map, _ = cropping_output_tuple
                elif len(cropping_output_tuple) == 3:
                    cropped_contact_prob_map, _, _ = cropping_output_tuple
                    print(f'Error in contact frame cropping! Dataset: {self.experiment_dset_name}, Episode: {episode_idx}, Index in episode: {idx_in_episode}')
                cropped_contact_prob_map = cropped_contact_prob_map[0]
                return_dict['cropped_contact_prob_map'] = cropped_contact_prob_map
                if self.blur_contact_prob_dict['enable']:
                    cropping_output_tuple = self.cropping(self.im_size,self.cropping_dict,contact_prob_map_blurred, pose_pxl_np)
                    if len(cropping_output_tuple) == 2:
                        cropped_contact_prob_map_blurred, _ = cropping_output_tuple
                    elif len(cropping_output_tuple) == 3:
                        cropped_contact_prob_map_blurred, _, _ = cropping_output_tuple
                        print(f'Error in contact frame cropping! Dataset: {self.experiment_dset_name}, Episode: {episode_idx}, Index in episode: {idx_in_episode}')
                    cropped_contact_prob_map_blurred = cropped_contact_prob_map_blurred[0]
                    return_dict['cropped_contact_prob_map_blurred'] = cropped_contact_prob_map_blurred
        else: # real dataset
            # get real data set global contact labels (two types of contact)
            # should use the pre-processed proprio with "real_global_contact" topic
            if self.is_anno_local:
                # replace the folder name of episode_depth_path_list
                json_anno_file_path = [ p.replace(self.color_dir_name, self.is_anno_local_dir).replace('png', 'json') for p in episode_color_path_list[start_idx:end_idx]] # H x W x T
                # Assume window size = 1
                if len(json_anno_file_path) != 0:
                    json_anno_file_path = json_anno_file_path[0]
                else:
                    json_anno_file_path = 'notexist'

                if self.im_resize is None:
                    contact_prob_map = np.zeros(self.im_size)
                    contact_prob_map_blurred = np.zeros(self.im_size)
                else: 
                    contact_prob_map = np.zeros(self.im_resize)
                    contact_prob_map_blurred = np.zeros(self.im_resize)

                if os.path.exists(json_anno_file_path):
                    #real_global_contact = np.array(proprio_df.iloc[nrst_proprio_idxs][self.real_anno_topic].values)
                    real_global_contact = 1
                    # Opening JSON file
                    contact_prob_map, number_contacts, contact_positions = self.get_real_contact_label(contact_prob_map, json_anno_file_path)

                    num_pad_contacts = self.max_num_contact - number_contacts
                    assert num_pad_contacts >= 0, 'number of contacts is bigger than max padding!!!'
                    
                    padded_contact_pxls_flt = np.pad(np.array(contact_positions , dtype=float), ((0, num_pad_contacts), (0,0)), mode='constant', constant_values=(np.nan))

                    if self.blur_contact_prob_dict['enable']:
                        contact_prob_map_blurred = cv2.GaussianBlur(contact_prob_map, (self.blur_contact_prob_dict['kernel_size'], self.blur_contact_prob_dict['kernel_size']), self.blur_contact_prob_dict['sigma'])
                else:
                    real_global_contact = 0
                    number_contacts = 0
                    padded_contact_pxls_flt = np.full((self.max_num_contact, 2), np.nan)
                
                return_dict.update({
                'global_contact': np.float32(real_global_contact),
                'contact_prob_map': contact_prob_map, # B x H x W
                'num_contacts': number_contacts, # B
                'contact_pxls_float': padded_contact_pxls_flt, # need to apply astype(int) after loading
                })

                if self.blur_contact_prob_dict['enable']:
                    return_dict['contact_prob_map_blurred'] = contact_prob_map_blurred

                if self.cropping_dict['enable']:
                    cropping_output_tuple = self.cropping(self.im_size,self.cropping_dict,contact_prob_map, pose_pxl_np)
                    if len(cropping_output_tuple) == 2:
                        cropped_contact_prob_map, _ = cropping_output_tuple 
                    elif len(cropping_output_tuple) == 3:
                        cropped_contact_prob_map, _, _ = cropping_output_tuple
                        print(f'Error in contact frame cropping! Dataset: {self.experiment_dset_name}, Episode: {episode_idx}, Index in episode: {idx_in_episode}')
                    cropped_contact_prob_map = cropped_contact_prob_map[0]
                    return_dict['cropped_contact_prob_map'] = cropped_contact_prob_map
                    if self.blur_contact_prob_dict['enable']:
                        cropping_output_tuple = self.cropping(self.im_size,self.cropping_dict,contact_prob_map_blurred, pose_pxl_np)
                        if len(cropping_output_tuple) == 2:
                            cropped_contact_prob_map_blurred, _ = cropping_output_tuple
                        elif len(cropping_output_tuple) == 3:
                            cropped_contact_prob_map_blurred, _, _ = cropping_output_tuple
                            print(f'Error in contact frame cropping! Dataset: {self.experiment_dset_name}, Episode: {episode_idx}, Index in episode: {idx_in_episode}')
                        cropped_contact_prob_map_blurred = cropped_contact_prob_map_blurred[0]
                        return_dict['cropped_contact_prob_map_blurred'] = cropped_contact_prob_map_blurred
            elif self.is_anno_global:
                real_global_contact = np.array(proprio_df.iloc[nrst_proprio_idxs[0]][self.real_global_continuous_topic])
                return_dict['global_contact'] = np.float32(real_global_contact)
        
        if self.proprio_history_plotting_dict['enable'] and self.viz_dataset:
            return_dict.update(self.get_proprioceptive_history(times_list = im_times, key_prefix='plotting_',
                                                                            proprio_input_dict=asdict(ProprioceptionInputDict(external_wrench=True, history_dict=self.proprio_history_plotting_dict)),
                                                                            return_global_labels=self.viz_global_contact,
                                                                            proprio_df = proprio_df,
                                                                            calib_grav = False,
                                                                            )
            )
        
        if self.ft_calibration_dict['enable']:
            return_dict.update(self.get_ft_calibration_data(im_times, proprio_df = proprio_df))

        return return_dict


    
    def get_real_contact_label(self, contact_label, json_path):
        f = open(json_path)
        data = json.load(f)
        contact_position = []
        for i in range(len(data['shapes'])):
            [w, h] = data['shapes'][i]['points'][0]
            # Resizing according to current image_size
            w = int(w / self.depth_shape[1] * contact_label.shape[1])
            h = int(h / self.depth_shape[0] * contact_label.shape[0])
            contact_position.append([w, h])
            contact_label[h][w] = 1
        
        return contact_label, len(data['shapes']), contact_position
    
    # image processing
    def fill_flow_image_holes(self, flow_np):
        nan_mask = flow_np == 0
        flow_np[nan_mask] = 255 # set to white
        return flow_np
    
    def affine_tf_to_pose(self, tf):
        ## returns 7d vector of trans, quat (x,y,z,w) format
        tf_np = np.reshape(tf, (4,4), order='F')
        # pose_np[0:4, 3]
        rot = tf_np[0:3, 0:3]
        # R @ R.T
        rot = R.from_matrix(rot)
        quat = rot.as_quat()
        quat = np.divide(quat, np.linalg.norm(quat))

        trans = tf_np[0:3, -1]
        pose = np.concatenate((trans, quat))
        return pose 
    
    def affine_tf_to_pos_quat_rpy_rotmax(self, tf): 
        ## returns 7d vector of trans, quat (x,y,z,w) format
        tf_np = np.reshape(tf, (4,4), order='F')
        # pose_np[0:4, 3]
        rot = tf_np[0:3, 0:3]
        # R @ R.T
        rot = R.from_matrix(rot)
        quat = rot.as_quat()
        quat = np.divide(quat, np.linalg.norm(quat))

        trans = tf_np[0:3, -1]
        return trans, quat, rot.as_euler('xyz'), rot.as_matrix()
    
    def get_ft_calibration_data(self, im_times, proprio_df=None, episode_idx = None):
        curr_time = im_times[0]
        if self.ft_calibration_dict['use_history']:
            proprio_hist_times = np.linspace(curr_time, curr_time - self.ft_calibration_dict['time_window'], 
                                             self.ft_calibration_dict['window_size'], endpoint=True).tolist()
        else:
            proprio_hist_times = [curr_time]
        nrst_proprio_idxs = self.get_nearest_idxs(proprio_hist_times, proprio_df.index)

        # get EE posquats        
        nrst_tfs_np = np.array(proprio_df.iloc[nrst_proprio_idxs][self.pose_topics].values)
        nrst_pos_np, nrst_quat_np, nrst_rpy_np, nrst_rotmax_np = [], [], [], []
        for i in range(nrst_tfs_np.shape[0]):
            pos, quat, rpy, rotmax = self.affine_tf_to_pos_quat_rpy_rotmax(nrst_tfs_np[i])
            # posquat = np.concatenate((pos, quat))
            # posrpy = np.concatenate((pos, rpy))
            # flatten the rotation matrix
            rotmax = rotmax.flatten()
            # posrotmax = np.concatenate((pos, rotmax))
            nrst_pos_np.append(pos)
            nrst_quat_np.append(quat)
            nrst_rpy_np.append(rpy)
            nrst_rotmax_np.append(rotmax)
        nrst_pos_np = np.array(nrst_pos_np).astype(np.float32)
        nrst_quat_np = np.array(nrst_quat_np).astype(np.float32)
        nrst_rpy_np = np.array(nrst_rpy_np).astype(np.float32)
        nrst_rotmax_np = np.array(nrst_rotmax_np).astype(np.float32)

        # get EE velocities
        nrst_EE_vels_np = np.array(proprio_df.iloc[nrst_proprio_idxs][self.EE_vel_topics].values).astype(np.float32)

        ## get wrenches
        current_wrenches_np = np.array(proprio_df.iloc[nrst_proprio_idxs[0]][self.wrench_topics].values).astype(np.float32)

        ## get joint positions
        nrst_joint_positions_np = np.array(proprio_df.iloc[nrst_proprio_idxs][self.joint_position_topics].values).astype(np.float32)

        ## get joint velocities
        nrst_joint_velocities_np = np.array(proprio_df.iloc[nrst_proprio_idxs][self.joint_velocity_topics].values).astype(np.float32)
        
        ## get joint desired torques
        nrst_joint_desired_torques_np = np.array(proprio_df.iloc[nrst_proprio_idxs][self.desired_joint_torques_topics].values).astype(np.float32)

        ## get joint external torques
        current_joint_external_torques_np = np.array(proprio_df.iloc[nrst_proprio_idxs[0]][self.external_joint_torques_topics].values).astype(np.float32)

        return_dict = {
        'EE_positions': nrst_pos_np, # B x T x 3
        'EE_quaternions': nrst_quat_np, # B x T x 4
        'EE_rpys': nrst_rpy_np, # B x T x 3
        'EE_rotmaxs': nrst_rotmax_np, # B x T x 9
        'EE_velocities': nrst_EE_vels_np, # B x T x 7
        'joint_positions': nrst_joint_positions_np, # B x T x 7
        'joint_velocities': nrst_joint_velocities_np, # B x T x 7
        'joint_desired_torques': nrst_joint_desired_torques_np, # B x T x 7
        'joint_external_torque': current_joint_external_torques_np, # B x 7
        'EE_external_wrench': current_wrenches_np, # B x 6
        'history_times': np.array(proprio_hist_times), # B x T
        }
        return return_dict

    def invert_transform(self, tf):
        R = tf[0:3, 0:3]
        T = tf[:3, -1]
        tf_inv = np.diag([1.,1.,1.,1.])
        tf_inv[:3, :3] = R.T
        tf_inv[:3, -1] = -R.T @ T
        return tf_inv
    
    def transform_pose_into_frame(self, pose_W, C_tf_W):
        C_rot_W = R.from_matrix(C_tf_W[:3, :3])

        pos = pose_W[:3]
        W_rot_EE = R.from_quat(pose_W[3:])
        pos_tfed = (C_tf_W @ np.concatenate((pos, np.array([1.,]))))[:-1]
        
        ori_tfed = C_rot_W * W_rot_EE

        return np.concatenate((pos_tfed, ori_tfed.as_quat())) 

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
    
    def get_EEO_grav_wrench(self, O_T_EE, EE_T_CoM, obj_mass):

        # EEO is the K/EE frame but rotated to align with the O frame
        # Similarly, CoMO is the CoM frame but rotated to align with the O frame
        t_span = O_T_EE.shape[0]
        CoM_pos_EE = -EE_T_CoM[:3, :3].T @ EE_T_CoM[:3, -1] 

        O_R_CoM = O_T_EE[:, :3, :3] @ EE_T_CoM[:3, :3]     # Time, 3, 3
        CoMO_pos_EEO = O_R_CoM @ CoM_pos_EE                # Time, 3

        #CoMO_pos_EEO_skew = np.zeros((t_span, 3, 3))
        CoMO_pos_EEO_skew = np.stack([np.array([
            [0, -C[2], C[1]],
            [C[2], 0, -C[0]],
            [-C[1], C[0], 0],
        ]) for C in CoMO_pos_EEO] )

        # hence why the rotation matrix here is identity
        CoMO_adj_EEO = np.stack([np.eye(6) for _ in range(t_span)])  # suppose to be t,6,6
        # CoMO_adj_EEO[:3, :3] = EEO_R_CoMO 
        # CoMO_adj_EEO[3:, 3:] = EEO_R_CoMO
        CoMO_adj_EEO[:, 3:, :3] = - CoMO_pos_EEO_skew # because the rotation matrix here is identity just need the negative of the skew sym matrix

        # this is the accel applied by gravity at the CoM frame but oriented with world/origin
        CoMO_grav_wrench  =  np.array([0, 0, -9.81, 0, 0, 0]) * obj_mass
        EEO_grav_wrench = CoMO_adj_EEO @ CoMO_grav_wrench

        return EEO_grav_wrench.squeeze()

    # TODO return times list
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
    
    #current_proprio_idx = None, 
    # TODO move cam frame transform to the datamodule, NOT HERE!
    def get_proprioceptive_history(self, key_prefix = '', times_list=None, 
                                proprio_input_dict = asdict(ProprioceptionInputDict()),
                                episode_index = None, proprio_df = None, contact_df = None, ft_predictions_df = None, 
                                calib_grav = False, objpara = None,
                                return_global_labels=False,
                                add_noise_dict = None,):
        '''get wrench history as T x 6 array where T is ordered by most recent to oldest (0 index is most recent)'''
        
        if ft_predictions_df is not None:
            assert not (proprio_input_dict['EE_pose'] or proprio_input_dict['desired_joint_torques']), "cannot specify ft_predictions_df and return_poses or return_desired_joint_torques"
        return_dict = {}
        # if current_proprio_idx is not None:
        #     nearest_proprioceptive_time = proprio_df.index[current_proprio_idx]
        # else: 
        nearest_proprioceptive_time = times_list[0] # TODO remove the use of im time lists
        in_cam_frame_dict = proprio_input_dict['in_cam_frame_dict']
        history_dict = proprio_input_dict['history_dict']

        if proprio_df is None: # then this is for plotting wrench history and we should get the proprio dataframe for the episode
            cam_tf_world = np.load(os.path.join(episode_dir_path, self.depth_extrinsic_filename))
            # get the proprio dataframe for the episode
            if episode_index is not None:
                episode_dir_path = self.episode_data_dir_path_list[episode_index]
                proprio_df = pd.read_pickle(os.path.join(episode_dir_path, 'proprio_df.pkl'))
            elif ft_predictions_df is not None:
                proprio_df = ft_predictions_df
        else:
            # compose the camera tf world matrix from the C_R_W (column major) and C_t_W lists
            cam_tf_world = np.eye(4)
            cam_tf_world[:3, :3] = np.array(in_cam_frame_dict['C_R_W'])#.reshape(3,3, order='F')
            cam_tf_world[:3, 3] = np.array(in_cam_frame_dict['C_t_W'])

        if history_dict['enable']:
            num_samples = int(history_dict['time_window'] * history_dict['sample_freq'])
            proprio_times = np.linspace(nearest_proprioceptive_time, nearest_proprioceptive_time - history_dict['time_window'], num_samples, endpoint=True).tolist()
        else:
            proprio_times = [nearest_proprioceptive_time]
        nrst_proprio_hist_idxs = self.get_nearest_idxs(proprio_times, proprio_df.index)
        nrst_proprio_hist_times = proprio_df.index[nrst_proprio_hist_idxs]
        return_dict[key_prefix + 'proprio_history_times'] = nrst_proprio_hist_times.to_numpy()
        # return_dict[key_prefix + 'proprio_hist_idxs'] = nrst_proprio_hist_idxs

        if proprio_input_dict['external_wrench']:
            if ft_predictions_df is not None:
                nrst_wrench_np = np.array(proprio_df.iloc[nrst_proprio_hist_idxs].values)
            else:
                nrst_wrench_np = np.array(proprio_df.iloc[nrst_proprio_hist_idxs][self.wrench_topics].values)
            if in_cam_frame_dict['enable']:
                nrst_wrench_np = self.transform_wrenches_into_frame(nrst_wrench_np, cam_tf_world)      # Time, 6
            if calib_grav:
                nrst_tfs_np = np.array(proprio_df.iloc[nrst_proprio_hist_idxs][self.pose_topics].values)
                O_T_EE = np.reshape(nrst_tfs_np, (-1,4,4), order='F')   # Time, 4,4
                EEO_grav_wrench = self.get_EEO_grav_wrench(O_T_EE, objpara["EE_T_objCoM"], objpara["mass"])
                nrst_wrench_np = (nrst_wrench_np + EEO_grav_wrench).astype(np.float32) #broadcasting grav_wrench from (6,) to (T,6) 
            return_dict[key_prefix + 'wrenches'] = nrst_wrench_np.astype(np.float32)

        if proprio_input_dict['EE_pose']:
            nrst_tfs_np = np.array(proprio_df.iloc[nrst_proprio_hist_idxs][self.pose_topics].values)
            nrst_poses_np = []
            nrst_poses_in_cam_frame_np = []
            for tf in nrst_tfs_np:
                pose = self.affine_tf_to_pose(tf)
                nrst_poses_np.append(pose)
                if in_cam_frame_dict['enable']:
                    nrst_poses_in_cam_frame_np.append(self.transform_pose_into_frame(pose, cam_tf_world))
            nrst_poses_np = np.array(nrst_poses_np) # T x 7
            return_dict[key_prefix + 'poses'] = nrst_poses_np
            if in_cam_frame_dict['enable']:
                nrst_poses_in_cam_frame_np = np.array(nrst_poses_in_cam_frame_np)
                return_dict[key_prefix + 'poses'] = nrst_poses_in_cam_frame_np
            return_dict[key_prefix + 'poses'] = return_dict[key_prefix + 'poses'].astype(np.float32)

        if proprio_input_dict['desired_joint_torques']:
            nrst_desired_joint_torques_np = np.array(proprio_df.iloc[nrst_proprio_hist_idxs][self.desired_joint_torques_topics].values)
            return_dict[key_prefix + 'desired_joint_torques'] = nrst_desired_joint_torques_np.astype(np.float32)

        if proprio_input_dict['joint_velocities']:
            nrst_joint_velocities_np = np.array(proprio_df.iloc[nrst_proprio_hist_idxs][self.joint_velocity_topics].values)
            return_dict[key_prefix + 'joint_velocities'] = nrst_joint_velocities_np.astype(np.float32)
        
        if proprio_input_dict['joint_positions']:
            nrst_joint_positions_np = np.array(proprio_df.iloc[nrst_proprio_hist_idxs][self.joint_position_topics].values)
            return_dict[key_prefix + 'joint_positions'] = nrst_joint_positions_np.astype(np.float32)
        
        if proprio_input_dict['external_joint_torques']:
            nrst_external_joint_torques_np = np.array(proprio_df.iloc[nrst_proprio_hist_idxs][self.external_joint_torques_topics].values)
            return_dict[key_prefix + 'external_joint_torques'] = nrst_external_joint_torques_np.astype(np.float32)
        
        if proprio_input_dict['measured_joint_torques']:
            nrst_measured_joint_torques_np = np.array(proprio_df.iloc[nrst_proprio_hist_idxs][self.measured_joint_torques_topics].values)
            return_dict[key_prefix + 'measured_joint_torques'] = nrst_measured_joint_torques_np.astype(np.float32)
        
        if add_noise_dict is not None:
            if add_noise_dict['enable']:    
                return_dict = self.add_noise_to_FT(return_dict, add_noise_dict, key_prefix, proprio_input_dict)
        
        if return_global_labels:
            if self.is_real_dataset:
                assert self.is_anno_global, "This dataset does not have global labels!"
                nrst_global_labels_np = np.array(proprio_df.iloc[nrst_proprio_hist_idxs][self.real_global_continuous_topic].values)
                return_dict[key_prefix + 'global_labels'] = nrst_global_labels_np.astype(np.float32)
            else:
                if not contact_df is None:
                    nrst_global_num_labels_np = np.zeros(len(proprio_times))
                    for i in range(len(proprio_times)):
                        nrst_global_num_labels_np[i] = self.get_contact_data(proprio_times[i], self.contact_dt, contact_df)['num_contacts']
                    nrst_global_labels_np = np.where(nrst_global_num_labels_np > 0, 1, 0)
                    return_dict[key_prefix + 'global_labels'] = nrst_global_labels_np.astype(np.float32)
                else:
                    nrst_global_labels_np = np.zeros(len(proprio_times))
                    return_dict[key_prefix + 'global_labels'] = nrst_global_labels_np.astype(np.float32)
                    
        return return_dict

    def add_noise_to_FT(self, return_dict, add_noise_dict, key_prefix, proprio_input_dict):
        # Time, 6 (3 force + 3 torque)
        # Double check if the dataset is real or not, won't add noise to real dataset
        if self.is_real_dataset:
            return return_dict

        # Add noise to different timestamp
        if add_noise_dict['noise_type'] == "gaussian_per_sample":
            if proprio_input_dict['external_wrench']:
                for t in range(return_dict[key_prefix + 'wrenches'].shape[0]):
                    return_dict[key_prefix + 'wrenches'][t] += np.random.normal(add_noise_dict['FT_dict']["mean_gaussian"], add_noise_dict['FT_dict']['std_gaussian'])
            if proprio_input_dict['desired_joint_torques']: 
                for t in range(return_dict[key_prefix + 'wrenches'].shape[0]):
                    return_dict[key_prefix + 'desired_joint_torques'][t] += np.random.normal(add_noise_dict['desired_joint_torque_dict']["mean_gaussian"], add_noise_dict['desired_joint_torque_dict']['std_gaussian'])
            if proprio_input_dict['joint_velocities']:
                for t in range(return_dict[key_prefix + 'wrenches'].shape[0]):
                    return_dict[key_prefix + 'joint_velocities'][t] += np.random.normal(add_noise_dict['joint_velo_dict']["mean_gaussian"], add_noise_dict['joint_velo_dict']['std_gaussian'])
            if proprio_input_dict['external_joint_torques']:
                for t in range(return_dict[key_prefix + 'wrenches'].shape[0]):
                    return_dict[key_prefix + 'external_joint_torques'][t] += np.random.normal(add_noise_dict['external_joint_torque_dict']["mean_gaussian"], add_noise_dict['external_joint_torque_dict']['std_gaussian'])
            if proprio_input_dict['measured_joint_torques']:
                for t in range(return_dict[key_prefix + 'wrenches'].shape[0]):
                    return_dict[key_prefix + 'measured_joint_torques'][t] += np.random.normal(add_noise_dict['measured_joint_torque_dict']["mean_gaussian"], add_noise_dict['measured_joint_torque_dict']['std_gaussian'])
        # Add same noise to all timestamps
        elif add_noise_dict['noise_type'] == "bias_per_batch":
            if proprio_input_dict['external_wrench']:
                return_dict[key_prefix + 'wrenches'] += np.random.normal(add_noise_dict['FT_dict']["mean_gaussian"], add_noise_dict['FT_dict']['std_gaussian'])
            if proprio_input_dict['desired_joint_torques']:
                return_dict[key_prefix + 'desired_joint_torques'] += np.random.normal(add_noise_dict['desired_joint_torque_dict']["mean_gaussian"], add_noise_dict['desired_joint_torque_dict']['std_gaussian'])
            if proprio_input_dict['joint_velocities']:
                return_dict[key_prefix + 'joint_velocities'] += np.random.normal(add_noise_dict['joint_velo_dict']["mean_gaussian"], add_noise_dict['joint_velo_dict']['std_gaussian'])
            if proprio_input_dict['external_joint_torques']:
                return_dict[key_prefix + 'external_joint_torques'] += np.random.normal(add_noise_dict['external_joint_torque_dict']["mean_gaussian"], add_noise_dict['external_joint_torque_dict']['std_gaussian'])
            if proprio_input_dict['measured_joint_torques']:
                return_dict[key_prefix + 'measured_joint_torques'] += np.random.normal(add_noise_dict['measured_joint_torque_dict']["mean_gaussian"], add_noise_dict['measured_joint_torque_dict']['std_gaussian']) 
        # Add a sinusoidal noise to all timestamps
        elif add_noise_dict['noise_type'] == "sin_per_batch":
            # Iterate over 3 forcr and 3 torque
            if proprio_input_dict['external_wrench']:
                for i in range(6):
                    amp = np.abs(np.random.normal(add_noise_dict['FT_dict']["amp_sin_mean"][i], add_noise_dict['FT_dict']['amp_sin_std'][i]))
                    freq = np.abs(np.random.normal(add_noise_dict['FT_dict']["freq_sin_mean"][i], add_noise_dict['FT_dict']['freq_sin_std'][i]))
                    phase = np.random.normal(add_noise_dict['FT_dict']["phase_sin_mean"][i], add_noise_dict['FT_dict']['phase_sin_std'][i])
                    return_dict[key_prefix + 'wrenches'][:,i] += amp * np.sin(2 * np.pi * freq * return_dict[key_prefix + 'proprio_history_times'] + phase)
            if proprio_input_dict['desired_joint_torques']:
                for i in range(7):
                    amp = np.abs(np.random.normal(add_noise_dict['desired_joint_torque_dict']["amp_sin_mean"][i], add_noise_dict['desired_joint_torque_dict']['amp_sin_std'][i]))
                    freq = np.abs(np.random.normal(add_noise_dict['desired_joint_torque_dict']["freq_sin_mean"][i], add_noise_dict['desired_joint_torque_dict']['freq_sin_std'][i]))
                    phase = np.random.normal(add_noise_dict['desired_joint_torque_dict']["phase_sin_mean"][i], add_noise_dict['desired_joint_torque_dict']['phase_sin_std'][i])
                    return_dict[key_prefix + 'desired_joint_torques'][:,i] += amp * np.sin(2 * np.pi * freq * return_dict[key_prefix + 'proprio_history_times'] + phase)
            if proprio_input_dict['joint_velocities']:
                for i in range(7):
                    amp = np.abs(np.random.normal(add_noise_dict['joint_velo_dict']["amp_sin_mean"][i], add_noise_dict['joint_velo_dict']['amp_sin_std'][i]))
                    freq = np.abs(np.random.normal(add_noise_dict['joint_velo_dict']["freq_sin_mean"][i], add_noise_dict['joint_velo_dict']['freq_sin_std'][i]))
                    phase = np.random.normal(add_noise_dict['joint_velo_dict']["phase_sin_mean"][i], add_noise_dict['joint_velo_dict']['phase_sin_std'][i])
                    return_dict[key_prefix + 'joint_velocities'][:,i] += amp * np.sin(2 * np.pi * freq * return_dict[key_prefix + 'proprio_history_times'] + phase)
            if proprio_input_dict['external_joint_torques']:
                for i in range(7):
                    amp = np.abs(np.random.normal(add_noise_dict['external_joint_torque_dict']["amp_sin_mean"][i], add_noise_dict['external_joint_torque_dict']['amp_sin_std'][i]))
                    freq = np.abs(np.random.normal(add_noise_dict['external_joint_torque_dict']["freq_sin_mean"][i], add_noise_dict['external_joint_torque_dict']['freq_sin_std'][i]))
                    phase = np.random.normal(add_noise_dict['external_joint_torque_dict']["phase_sin_mean"][i], add_noise_dict['external_joint_torque_dict']['phase_sin_std'][i])
                    return_dict[key_prefix + 'external_joint_torques'][:,i] += amp * np.sin(2 * np.pi * freq * return_dict[key_prefix + 'proprio_history_times'] + phase)
            if proprio_input_dict['measured_joint_torques']:
                for i in range(7):
                    amp = np.abs(np.random.normal(add_noise_dict['measured_joint_torque_dict']["amp_sin_mean"][i], add_noise_dict['measured_joint_torque_dict']['amp_sin_std'][i]))
                    freq = np.abs(np.random.normal(add_noise_dict['measured_joint_torque_dict']["freq_sin_mean"][i], add_noise_dict['measured_joint_torque_dict']['freq_sin_std'][i]))
                    phase = np.random.normal(add_noise_dict['measured_joint_torque_dict']["phase_sin_mean"][i], add_noise_dict['measured_joint_torque_dict']['phase_sin_std'][i])
                    return_dict[key_prefix + 'measured_joint_torques'][:,i] += amp * np.sin(2 * np.pi * freq * return_dict[key_prefix + 'proprio_history_times'] + phase)
        return return_dict
    
    def transform_wrenches_into_frame(self, wrenches, depth_tf_world):
        tfed_wrenches = []
        for i in range(wrenches.shape[0]):
            # print(tfs_np[i])
            wrench = wrenches[i]
            R = depth_tf_world[:3, :3]
            wrench = np.concatenate((R@wrench[:3], R@wrench[3:]))
            tfed_wrenches.append(wrench)
        return np.array(tfed_wrenches)
    
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
    
    def get_contact_data(self, im_time, contact_dt, contact_df, clump=False):
        # IVE REMOVED THE CONTACT NORMALS AND WRENCHES FOR NOW!!! 04/21/23

        # These are all 2d num_contacts x number of feature dims
        contact_dict = {}
        contact_positions = []
        # contact_wrenches = []
        # contact_normals = []
        contact_object_geometry = []

        total_num_contacts = 0

        nrst_contact_idx = self.get_nearest_idxs([im_time], contact_df.index)[0]
        contact_dict['contact_idx'] = nrst_contact_idx

        row = contact_df.iloc[nrst_contact_idx].loc[contact_df.iloc[nrst_contact_idx].notnull()]
        contact_time = row.name
        contact_dict['time'] = contact_time

        ## again need to filter per timestamp, only the contact states that have object name in collisions
        collision_row = row[row.keys().str.contains('collision')]
        ## filter out to only collisions which contain object name
        ## need to handle when some collision fields are None by setting na flag!
        filtered_collision_row = collision_row[collision_row.str.contains(self.grasped_obj_name, na=False)]
        ## get the state index after the state substring
        filtered_state_idxs = filtered_collision_row.keys().str.split('/').str.get(3)

        contact_time_diff = abs(contact_time - im_time)
        contact_dict['time_diff'] = contact_time_diff

        ## if there is no contact ...
        ## inflating by dt factor greater than 1 allows checking multiple local timestamps for contact
        ## in the case where the contact dissapears intermittently!
        # if contact_time_diff > (contact_dt*1.5): #1.5
        if (contact_time_diff > (self.contact_persist_time)): #1.5
            contact_dict['positions'] = np.nan
            # contact_dict['wrenches'] = np.nan
            # contact_dict['normals'] = np.nan
        ## if there is contact temporally locally
        else:
            for state_idx_str in filtered_state_idxs:
                state_num_contacts = len(row[row.keys().str.contains(state_idx_str + '/depths', na=False)])
                total_num_contacts += state_num_contacts

                base_contact_name = '/contact_data/states/' + state_idx_str
                for contact_idx in range(state_num_contacts):

                    contact_pos_idx = base_contact_name + '/contact_positions/' + str(contact_idx) 
                    contact_pos_cols = [col for col in row.keys() if contact_pos_idx in col]
                    contact_pos = row[contact_pos_cols].values.astype(np.float64)
                    contact_positions.append(contact_pos)

                    # long ugly code to get the contact object geometry
                    contact_name_idx = base_contact_name + '/collision' 
                    contact_name_cols = [col for col in row.keys() if contact_name_idx in col]
                    contact_names = row[contact_name_cols].values.tolist()
                    for contact_name in contact_names:
                        if self.grasped_obj_name in contact_name:
                            if 'box' in contact_name:
                                grasped_object_geometry = 'box'
                            elif 'sphere' in contact_name:
                                grasped_object_geometry = 'sphere'
                            elif 'cylinder' in contact_name:
                                grasped_object_geometry = 'cylinder'
                        else:
                            if 'table' in contact_name:
                                contact_object_geometry.append('table')
                            elif 'box' in contact_name:
                                contact_object_geometry.append('box')
                            elif 'sphere' in contact_name:
                                contact_object_geometry.append('sphere')
                            elif 'cylinder' in contact_name:
                                contact_object_geometry.append('cylinder')

            contact_dict['positions'] = np.array(contact_positions, dtype=float)
            # contact_dict['wrenches'] = np.array(contact_wrenches, dtype=float)
            # contact_dict['normals'] = np.array(contact_normals, dtype=float)
            contact_dict['grasped_object_geometry'] = grasped_object_geometry
            # join list of strings with comma
            contact_dict['contact_object_geometry'] = ','.join(contact_object_geometry)

        contact_dict['num_contacts'] = total_num_contacts
        return contact_dict

    def point_proj(self, K, pos, C_tf_W = None):
        if not C_tf_W is None:
            contact_pos_in_depth = (C_tf_W @ np.append(pos, 1))[:-1]
        else: 
            contact_pos_in_depth = pos
        project_coords = K @ (contact_pos_in_depth)
        return (project_coords[:2]/project_coords[-1]).astype(int)  
    
    #### VISUALIZATION UTILS
    # pink color by default
    def viz_contact_pos(self, image_np, contact_pxls_flt, num_contacts, radius=1, color=(255, 133, 233)): # expects contact positions in world frame
        contact_pxls = contact_pxls_flt[:num_contacts, ...].astype(int)
        for idxs in contact_pxls:
            image_np = cv2.circle(image_np, tuple(idxs), radius=radius, color=color, thickness=-1)
        return image_np

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
        return self.depth_episode_timestamp_series.index.max() + 1
    
    def get_indices_for_episode_list(self, episode_list):
        return self.depth_episode_timestamp_series.index.get_indexer_for(episode_list)
    
    def get_contact_indices_for_episode_list(self, episode_list):
        # generate the contact mask across all episodes by concatenating the masks
        for episode in episode_list:
            # load the contact mask from the episode data dir
            episode_data_dir = self.episode_data_dir_path_list[episode]
            contact_mask = np.load(os.path.join(episode_data_dir, 'contact_idxs_mask.npy'))
            if episode == episode_list[0]:
                contact_mask_all_episodes = contact_mask
            else:
                contact_mask_all_episodes = np.concatenate((contact_mask_all_episodes, contact_mask))
        
        return self.get_indices_for_episode_list(episode_list)[contact_mask_all_episodes]

    # def get_half_randomized_half_indices_for_episode_list():
    def get_half_split_indices_for_episode_list(self, episode_list):
        episode_indices = self.depth_episode_timestamp_series.index.get_indexer_for(episode_list)
        num_samples_per_episode = self.depth_episode_timestamp_series.groupby('episode').count().iloc[episode_list]
        # create a mask of first half true lasat half false
        mask = np.zeros(len(episode_list), dtype=bool)
        mask[:len(episode_list)//2] = True
        np.random.shuffle(mask)

        # interleave mask with opposite boolean
        train_mask = np.repeat(mask, 2)
        train_mask[1::2] = ~train_mask[1::2]

        # construct repeat number array for each episode
        num_elements_per_episode_first_half = num_samples_per_episode.values // 2
        num_elements_per_episode_second_half = num_samples_per_episode.values - num_elements_per_episode_first_half

        # interleave the two arrays
        repeat_mask = np.repeat(num_elements_per_episode_first_half, 2)
        repeat_mask[1::2] = num_elements_per_episode_second_half

        # construct an array of indices using the mask
        train_mask_idxs = np.repeat(train_mask, repeat_mask)
        train_indices = episode_indices[train_mask_idxs]
        valid_mask_idxs = ~train_mask_idxs
        valid_indices = episode_indices[valid_mask_idxs]

        return train_indices, valid_indices

    def get_proprio_df(self, episode_idx):
        # get the episode index of the accessed index
        episode_dir_path = self.episode_data_dir_path_list[episode_idx]
        return pd.read_pickle(os.path.join(episode_dir_path, 'proprio_df.pkl'))
    
    def processing_startend2contin(self, Binary):
        idx = np.where(Binary)[0]
        for i in range(int(idx.shape[0]/2)):
            Binary[idx[i*2]:idx[i*2+1]] = 1
        return Binary
    
    @staticmethod
    def cropping(im_size, cropping_dict, image, EE_poses_pxl):
        no_error = True

        bb_height = cropping_dict['bb_height']
        bb_width = cropping_dict['bb_width']
        down_scale = cropping_dict['down_scale']

        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=0)
        
        assert image.shape == (1, im_size[0], im_size[1]), 'image shape is not 1x{}x{}'.format(im_size[0], im_size[1])

        image_height = image.shape[1]
        image_width = image.shape[2]

        bb_image = np.zeros((1, bb_height, bb_width))

        x1 = int(EE_poses_pxl[0]-bb_width/2)
        x2 = int(EE_poses_pxl[0]+bb_width/2)
        y1 = int(EE_poses_pxl[1]-(bb_height/2)+down_scale) 
        y2 = int(EE_poses_pxl[1]+(bb_height/2)+down_scale)

        # if lower bounds x1, y1 are greater than image size, then we need to keep bb_image as all zeros
        # similarly if upper bounds x2, y2 are less than 0, then we need to keep bb_image as all zeros
        if x1 >= image_width or y1 >= image_height or x2 <= 0 or y2 <= 0:
            no_error = False
        else:
            x1_clamped = max(x1, 0)
            y1_clamped = max(y1, 0)
            x2_clamped = min(x2, image_width)
            y2_clamped = min(y2, image_height)
            
            cropped_image = image[0][y1_clamped:y2_clamped, x1_clamped:x2_clamped]
            assert not (cropped_image.shape[0] == 0 or cropped_image.shape[1] == 0), 'cropped image shape is 0'
            
            # can never be negative
            x1_bb = max(x1, 0) - x1
            y1_bb = max(y1, 0) - y1
            # can never be greater than bb_width or bb_height
            x2_bb = bb_width - max(0, x2-image_width)
            y2_bb = bb_height - max(0, y2-image_height)

            # make sure the slice is the same shape as the cropped image
            assert cropped_image.shape[0] == y2_bb - y1_bb, 'cropped image shape is not the same as bb_image shape'
            assert cropped_image.shape[1] == x2_bb - x1_bb, 'cropped image shape is not the same as bb_image shape'
            bb_image[0][y1_bb:y2_bb, x1_bb:x2_bb] = cropped_image
            #image_cropped[idx][0] = np.squeeze(image)[idx][y1:int(y1+bb_height), x1:x1+int(bb_width)]

        # x_list.append(x1)
        # y_list.append(y1)
        bb_top_left_coordinate = np.zeros(EE_poses_pxl.shape)
        bb_top_left_coordinate[0] = x1  # W
        bb_top_left_coordinate[1] = y1  # H

        if no_error:
            return bb_image, bb_top_left_coordinate
        else:
            return bb_image, bb_top_left_coordinate, no_error
