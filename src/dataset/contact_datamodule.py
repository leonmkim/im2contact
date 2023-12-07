import pytorch_lightning as pl
from torch.utils.data import DataLoader, Subset, ConcatDataset
from .contact_dataset_episodic import ContactDatasetEpisodic
from torch.utils.data import random_split
import os
import torch
import numpy as np
import ast
from typing import Optional, List, Dict, Any, Union, Tuple
from dataclasses import dataclass, field, asdict

import copy
import sys
sys.path.append('../')

from src.dataset.contact_dataclasses import OpticalFlowDict, CroppingDict, ContextFrameDict, BlurContactMapDict, CuratedValidationDatasetDict, NormalizeDict, CompensateObjGravityDict, TestDatasetDict, FitDatasetDict, InCamFrameDict, ProprioceptionInputDict, ProprioceptionHistoryDict, AddNoiseDict, DatasetCollection

def normalize_wrench(wrench, normalize_dict):
    # A batch of data just use the first one
    # force_min, force_max, torque_min, torque_max = normalize_dict['force_min_max'][0][0], normalize_dict['force_min_max'][1][0], normalize_dict['torque_min_max'][0][0], normalize_dict['torque_min_max'][1][0]  
    max_wrench = torch.tensor(normalize_dict['O_F_ext_EE_max']).unsqueeze(0).unsqueeze(0).repeat(wrench.shape[0], wrench.shape[1], 1).float() # B x T x 6
    min_wrench = torch.tensor(normalize_dict['O_F_ext_EE_min']).unsqueeze(0).unsqueeze(0).repeat(wrench.shape[0], wrench.shape[1], 1).float()
    # B x T x 6
    wrench = torch.clamp(wrench, min=min_wrench, max=max_wrench)
    wrench = (wrench - min_wrench) / (max_wrench - min_wrench)
    return wrench

def normalize_pose(pose, normalize_dict):
    max_position = torch.tensor(normalize_dict['O_position_EE_max']).unsqueeze(0).unsqueeze(0).repeat(pose.shape[0], pose.shape[1], 1).float() # B x T x 3
    min_position = torch.tensor(normalize_dict['O_position_EE_min']).unsqueeze(0).unsqueeze(0).repeat(pose.shape[0], pose.shape[1], 1).float()
    # B x T x 7 (position and quaternion)
    # quaternion is already normalized
    pose[:, :, 0:3] = torch.clamp(pose[:, :, 0:3], min=min_position, max=max_position)
    pose[:, :, 0:3] = (pose[:, :, 0:3] - min_position) / (max_position - min_position)
    return pose

def normalize_desired_torque(desired_torques, normalize_dict):
    torque_min, torque_max = normalize_dict['desired_torques_min'], normalize_dict['desired_torques_max']
    # B x T x 7 (joints)
    max_torque = torch.tensor(torque_max).unsqueeze(0).unsqueeze(0).repeat(desired_torques.shape[0], desired_torques.shape[1], 1).float() # B x T x 7
    min_torque = torch.tensor(torque_min).unsqueeze(0).unsqueeze(0).repeat(desired_torques.shape[0], desired_torques.shape[1], 1).float()
    desired_torques = torch.clamp(desired_torques, min=min_torque, max=max_torque)
    desired_torques = (desired_torques - min_torque) / (max_torque - min_torque)
    return desired_torques

def normalize_joint_positions(joint_positions, normalize_dict):
    position_min, position_max = normalize_dict['joint_positions_min'], normalize_dict['joint_positions_max']
    # B x T x 7 (joints)
    max_position = torch.tensor(position_max).unsqueeze(0).unsqueeze(0).repeat(joint_positions.shape[0], joint_positions.shape[1], 1).float() # B x T x 7
    min_position = torch.tensor(position_min).unsqueeze(0).unsqueeze(0).repeat(joint_positions.shape[0], joint_positions.shape[1], 1).float()
    joint_positions = torch.clamp(joint_positions, min=min_position, max=max_position)
    joint_positions = (joint_positions - min_position) / (max_position - min_position)
    return joint_positions

def normalize_joint_velocities(joint_velocities, normalize_dict):
    velocity_min, velocity_max = normalize_dict['joint_velocities_min'], normalize_dict['joint_velocities_max']
    # B x T x 7 (joints)
    max_velocity = torch.tensor(velocity_max).unsqueeze(0).unsqueeze(0).repeat(joint_velocities.shape[0], joint_velocities.shape[1], 1).float() # B x T x 7
    min_velocity = torch.tensor(velocity_min).unsqueeze(0).unsqueeze(0).repeat(joint_velocities.shape[0], joint_velocities.shape[1], 1).float()
    joint_velocities = torch.clamp(joint_velocities, min=min_velocity, max=max_velocity)
    joint_velocities = (joint_velocities - min_velocity) / (max_velocity - min_velocity)
    return joint_velocities

def normalize_external_joint_torques(external_joint_torques, normalize_dict):
    torque_min, torque_max = normalize_dict['external_torques_min'], normalize_dict['external_torques_max']
    # B x T x 7 (joints)
    max_torque = torch.tensor(torque_max).unsqueeze(0).unsqueeze(0).repeat(external_joint_torques.shape[0], external_joint_torques.shape[1], 1).float() # B x T x 7
    min_torque = torch.tensor(torque_min).unsqueeze(0).unsqueeze(0).repeat(external_joint_torques.shape[0], external_joint_torques.shape[1], 1).float()
    external_joint_torques = torch.clamp(external_joint_torques, min=min_torque, max=max_torque)
    external_joint_torques = (external_joint_torques - min_torque) / (max_torque - min_torque)
    return external_joint_torques

def normalize_measured_joint_torques(measaured_joint_torques, normalize_dict):
    torque_min, torque_max = normalize_dict['measured_torques_min'], normalize_dict['measured_torques_max']
    # B x T x 7 (joints)
    max_torque = torch.tensor(torque_max).unsqueeze(0).unsqueeze(0).repeat(measaured_joint_torques.shape[0], measaured_joint_torques.shape[1], 1).float() # B x T x 7
    min_torque = torch.tensor(torque_min).unsqueeze(0).unsqueeze(0).repeat(measaured_joint_torques.shape[0], measaured_joint_torques.shape[1], 1).float()
    measaured_joint_torques = torch.clamp(measaured_joint_torques, min=min_torque, max=max_torque)
    measaured_joint_torques = (measaured_joint_torques - min_torque) / (max_torque - min_torque)
    return measaured_joint_torques

def get_global_input(batch, hparams, batch_keys_dict):
    model_input_dict = {}
    if hparams.global_proprio_input_dict['external_wrench']:
        external_wrenches = batch[batch_keys_dict['global_wrench_key']]
        if hparams.global_normalize_dict['enable']:
            external_wrenches = normalize_wrench(external_wrenches, hparams.global_normalize_dict)
        model_input_dict['wrenches'] = external_wrenches
    if hparams.global_proprio_input_dict['joint_positions']:
        joint_positions = batch[batch_keys_dict['global_joint_position_key']]
        if hparams.global_normalize_dict['enable']:
            joint_positions = normalize_joint_positions(joint_positions, hparams.global_normalize_dict)
        model_input_dict['joint_positions'] = joint_positions
    if hparams.global_proprio_input_dict['joint_velocities']:
        joint_velocities = batch[batch_keys_dict['global_joint_velocity_key']]
        if hparams.global_normalize_dict['enable']:
            joint_velocities = normalize_joint_velocities(joint_velocities, hparams.global_normalize_dict)
        model_input_dict['joint_velocities'] = joint_velocities
    if hparams.global_proprio_input_dict['desired_joint_torques']:
        desired_torques = batch[batch_keys_dict['global_desired_joint_torque_key']]
        if hparams.global_normalize_dict['enable']:
            desired_torques = normalize_desired_torque(desired_torques, hparams.global_normalize_dict)
        model_input_dict['desired_joint_torques'] = desired_torques
    if hparams.global_proprio_input_dict['external_joint_torques']:
        external_joint_torques = batch[batch_keys_dict['global_external_joint_torque_key']]
        if hparams.global_normalize_dict['enable']:
            external_joint_torques = normalize_external_joint_torques(external_joint_torques, hparams.global_normalize_dict)
        model_input_dict['external_joint_torques'] = external_joint_torques
    if hparams.global_proprio_input_dict['measured_joint_torques']:
        measured_joint_torques = batch[batch_keys_dict['global_measured_joint_torque_key']]
        if hparams.global_normalize_dict['enable']:
            measured_joint_torques = normalize_measured_joint_torques(measured_joint_torques, hparams.global_normalize_dict)
        model_input_dict['measured_joint_torques'] = measured_joint_torques
    return model_input_dict #, y

def get_global_target(batch, hparams, batch_keys_dict):
    target_dict = {}
    target_dict['y'] = batch['global_contact']
    return target_dict # (batch_size, 6)

def get_local_input(batch, hparams, batch_keys_dict):
    input_dict = {}
    if hparams.local_proprio_input_dict['external_wrench'] or hparams.local_proprio_input_dict['desired_joint_torques']:
        wrench = []
        if hparams.local_proprio_input_dict['external_wrench']:
            external_wrench = batch[batch_keys_dict['local_wrench_key']]
            if hparams.local_normalize_dict['enable']:
                external_wrench = normalize_wrench(external_wrench, hparams.local_normalize_dict)
            wrench.append(external_wrench)
        if hparams.local_proprio_input_dict['desired_joint_torques']: # just massage the desired joint torques in with the external wrenches
            desired_joint_torques = batch[batch_keys_dict['local_desired_joint_torque_key']]
            if hparams.local_normalize_dict['enable']:
                desired_joint_torques = normalize_desired_torque(desired_joint_torques, hparams.local_normalize_dict)
            wrench.append(desired_joint_torques)
        wrench = torch.cat(tuple(wrench), dim=-1)
        # flatten the time and input dim, while preserving batch dimensions
        wrench = wrench.reshape(wrench.shape[0], -1)
        input_dict['wrench'] = wrench

    if hparams.local_proprio_input_dict['EE_pose']:
        pose = batch[batch_keys_dict['local_pose_key']]
        if hparams.local_normalize_dict['enable']:
            pose = normalize_pose(pose, hparams.local_normalize_dict)
        # flatten the time and input dim, while preserving batch dimensions
        pose = pose.reshape(pose.shape[0], -1)
        input_dict['pose'] = pose

    # These are not going through 'get_proprioceptive_history' function
    input_dict['EE_pose_pxl'] = batch[batch_keys_dict['EE_pxl_key']]
    if not hparams.cropping_dict['enable']:
        image = batch[batch_keys_dict['depth_image_key']] # B x 1 x H x W
        if hparams.context_frame_dict['enable']:
            input_dict['first_epi_image']=batch[batch_keys_dict['context_frame_key']]
        if hparams.optical_flow_dict['enable']:
            if hparams.optical_flow_dict['use_image']:
                flow_key = hparams.flow_image_key
                # if real_dataset:
                    # flow_key += '_aligned_to_depth'
            else:
                flow_key = 'flow'
                # if real_dataset:
                    # flow_key += '_aligned_to_depth'
            image = torch.cat((image, batch[flow_key]), dim=1)
        input_dict['image'] = image
    else:
        input_dict['pose_pxl_for_coord']= batch[batch_keys_dict['pose_pxl_for_coordinate_conv_key']]
        image = batch[batch_keys_dict['cropped_depth_image_key']] # B x 1 x H x W
        if hparams.context_frame_dict['enable']:
            input_dict['first_epi_image']=batch[batch_keys_dict['cropped_context_frame_key']]
            input_dict['first_epi_pose_pxl_for_coord']= batch[batch_keys_dict['cropped_context_pose_pxl_coordinate_key']]
        if hparams.optical_flow_dict['enable']: # whether use_image or not isn't important
            image = torch.cat((image, batch[batch_keys_dict['cropped_flow_image_key']]), dim=1)
        input_dict['image'] = image       
    return input_dict

def get_local_target(batch, hparams, batch_keys_dict):
    # y is for actual training, while unmodified_y is for consistent evaluation across models
    target_dict = {}
    target_dict['y'] = batch[batch_keys_dict['train_target_key']] # B x H x W
    target_dict['unmodified_y'] = batch[batch_keys_dict['unmodified_target_key']]
    return target_dict

def get_joint_input(batch, hparams, real_dataset=False):
    raise NotImplementedError

class ContactDataModule(pl.LightningDataModule):
    def __init__(self, 
                 model_type: Optional[str] = None,
                 fit_dataset_dict: Optional[FitDatasetDict] = None,
                 curated_val_dataset_list: Optional[List[CuratedValidationDatasetDict]] = None,
                 test_dataset_dict_list: Optional[List[TestDatasetDict]] = None,
                 max_depth_clip: float=2.0,
                 optical_flow_dict: OpticalFlowDict = OpticalFlowDict(), 
                 context_frame_dict: ContextFrameDict = ContextFrameDict(), 
                 cropping_dict: CroppingDict = CroppingDict(), 
                 local_proprio_input_dict: ProprioceptionInputDict = ProprioceptionInputDict(),
                 global_proprio_input_dict: ProprioceptionInputDict = ProprioceptionInputDict(),
                 global_normalize_dict:  NormalizeDict = NormalizeDict(),
                 local_normalize_dict:  NormalizeDict = NormalizeDict(),
                 im_resize_tuple: Tuple[int, int] = (240, 320), 
                 blur_contact_map_dict: BlurContactMapDict = BlurContactMapDict(),
                 compensate_obj_gravity_dict: CompensateObjGravityDict=CompensateObjGravityDict(), # TODO split this off into local and global!!! 
                 proprio_history_plotting_dict: ProprioceptionHistoryDict = ProprioceptionHistoryDict(), 
                 add_noise_dict: AddNoiseDict = AddNoiseDict(),
                **kwargs):
        super().__init__()
        if model_type is not None:
            assert model_type in ['global', 'local', 'conditional_local', 'joint'], 'model_type must be one of global, local, joint'

        # not sure I even need these.... 
        # self.save_hyperparameters() seems to do the job....

        if isinstance(fit_dataset_dict, FitDatasetDict):
            fit_dataset_dict = asdict(fit_dataset_dict)
        if curated_val_dataset_list is not None:
            for i, curated_valid_dataset_dict in enumerate(curated_val_dataset_list):
                if isinstance(curated_valid_dataset_dict, CuratedValidationDatasetDict):
                    curated_val_dataset_list[i] = asdict(curated_valid_dataset_dict)
        if test_dataset_dict_list is not None:
            for i, test_dataset_dict in enumerate(test_dataset_dict_list):
                if isinstance(test_dataset_dict, TestDatasetDict):
                    test_dataset_dict_list[i] = asdict(test_dataset_dict)
        if isinstance(local_proprio_input_dict, ProprioceptionInputDict):
            local_proprio_input_dict = asdict(local_proprio_input_dict)
        if isinstance(global_proprio_input_dict, ProprioceptionInputDict):
            global_proprio_input_dict = asdict(global_proprio_input_dict)
        if isinstance(proprio_history_plotting_dict, ProprioceptionHistoryDict):
            proprio_history_plotting_dict = asdict(proprio_history_plotting_dict)
        if isinstance(optical_flow_dict, OpticalFlowDict):
            optical_flow_dict = asdict(optical_flow_dict)
        if isinstance(context_frame_dict, ContextFrameDict):
            context_frame_dict = asdict(context_frame_dict)
        if isinstance(cropping_dict, CroppingDict):
            cropping_dict = asdict(cropping_dict)
        if isinstance(blur_contact_map_dict, BlurContactMapDict):
            blur_contact_map_dict = asdict(blur_contact_map_dict)
        if isinstance(local_normalize_dict, NormalizeDict):
            local_normalize_dict = asdict(local_normalize_dict)
        if isinstance(global_normalize_dict, NormalizeDict):
            global_normalize_dict = asdict(global_normalize_dict)
        if isinstance(compensate_obj_gravity_dict, CompensateObjGravityDict):
            compensate_obj_gravity_dict = asdict(compensate_obj_gravity_dict)

        if isinstance(add_noise_dict, AddNoiseDict):
            add_noise_dict = asdict(add_noise_dict)
        # ------------------------------------------------------------------------------------

        # set up dataset keys
        # TODO set this in the dataset class and get it here!
        self.batch_keys_dict = {}
        self.batch_keys_dict['depth_image_key'] = 'images_tensor'
        self.batch_keys_dict['color_image_path_key'] = 'color_paths'
        self.batch_keys_dict['EE_pxl_key'] = 'current_pose_pxl'
        self.batch_keys_dict['target_contact_map_key'] = 'contact_prob_map'
        self.batch_keys_dict['contact_pxls_key'] = 'contact_pxls_float'
        self.batch_keys_dict['num_contacts_key'] = 'num_contacts'
        self.batch_keys_dict['current_time_key'] = 'im_times'
        self.batch_keys_dict['episode_idx_key'] = 'episode'
        self.batch_keys_dict['wrench_history_plotting_key'] = 'plotting_wrenches'
        self.batch_keys_dict['wrench_history_plotting_times_key'] = 'plotting_proprio_history_times'
        self.batch_keys_dict['global_label_plotting_key'] = 'plotting_global_labels'
        self.batch_keys_dict['normalize_key'] = 'updated_normalize_dict'
        self.batch_keys_dict['grasped_object_mass_key'] = 'grasped_object_mass'

        # ------------------------------------------------------------------------------------

        ## set up datasets returns!

        self.im_resize = im_resize_tuple
        # prep datasets
        if model_type in ['global', 'joint']:
            self.batch_keys_dict['global_wrench_key'] = 'global_wrenches'
            self.batch_keys_dict['global_pose_key'] = 'global_poses'
            self.batch_keys_dict['global_joint_velocity_key'] = 'global_joint_velocities'
            self.batch_keys_dict['global_joint_position_key'] = 'global_joint_positions'
            self.batch_keys_dict['global_desired_joint_torque_key'] = 'global_desired_joint_torques'
            self.batch_keys_dict['global_external_joint_torque_key'] = 'global_external_joint_torques'
            self.batch_keys_dict['global_measured_joint_torque_key'] = 'global_measured_joint_torques'

        # ------------------------------------------------------------------------------------

        self.only_contact = False
        if model_type in ['local', 'conditional_local', 'joint']:
            self.batch_keys_dict['unmodified_target_key'] = 'contact_prob_map'
            if blur_contact_map_dict['enable']:
                if cropping_dict['enable']:
                    self.batch_keys_dict['train_target_key'] = 'cropped_contact_prob_map_blurred' #
                else:
                    self.batch_keys_dict['train_target_key'] = 'contact_prob_map_blurred' #
            else:
                if cropping_dict['enable']:
                    self.batch_keys_dict['train_target_key'] = 'cropped_contact_prob_map'
                else:
                    self.batch_keys_dict['train_target_key'] = 'contact_prob_map' # B x H x W

            self.batch_keys_dict['local_wrench_key'] = 'local_wrenches'
            self.batch_keys_dict['local_pose_key'] = 'local_poses'
            self.batch_keys_dict['local_desired_joint_torque_key'] = 'local_desired_joint_torques'

            if context_frame_dict['enable']:
                self.batch_keys_dict['context_frame_key'] = 'first_epi_images_tensor'
                self.batch_keys_dict['context_color_path_key'] = 'context_color_path'
                self.batch_keys_dict['context_pose_pxl_coordinate_key'] = 'first_epi_pose_pxl'
                
            self.batch_keys_dict['cropped_depth_image_key'] = 'cropped_images_tensor'
            self.batch_keys_dict['pose_pxl_for_coordinate_conv_key'] = 'bb_top_left_coordinate'
            self.batch_keys_dict['cropped_context_frame_key'] = 'cropped_first_epi_images_tensor'
            self.batch_keys_dict['cropped_context_pose_pxl_coordinate_key'] = 'context_bb_top_left_coordinate'
            self.batch_keys_dict['cropped_flow_image_key'] = 'cropped_optical_flow_images_tensor'
            if optical_flow_dict['enable']:
                if optical_flow_dict['use_image']:
                    self.batch_keys_dict['flow_image_key'] = 'flow_image'
                else:
                    self.batch_keys_dict['flow_image_key'] = 'flow'

            if model_type == 'conditional_local':
                self.only_contact = True 
            
            # handle transform of normalize dict here:
            if local_normalize_dict['enable'] and local_proprio_input_dict['in_cam_frame_dict']['enable']:
                # cam_tf_world = np.load(os.path.join(self.fit_dataset.episode_data_dir_path_list[0], self.fit_dataset.depth_dir_name, self.fit_dataset.depth_extrinsic_filename))
                cam_tf_world = np.eye(4)
                cam_tf_world[:3, :3] = np.array(local_proprio_input_dict['in_cam_frame_dict']['C_R_W'])#.reshape(3, 3, order='F')
                cam_tf_world[:3, 3] = np.array(local_proprio_input_dict['in_cam_frame_dict']['C_t_W'])
                pose_min = np.array(local_normalize_dict['O_position_EE_min'] + [1])
                pose_min_in_cam = cam_tf_world @ pose_min
                pose_max = np.array(local_normalize_dict['O_position_EE_max'] + [1])
                pose_max_in_cam = cam_tf_world @ pose_max
                # swap min and max if necessary
                for i in range(3):
                    if pose_min_in_cam[i] > pose_max_in_cam[i]:
                        pose_min_in_cam[i], pose_max_in_cam[i] = pose_max_in_cam[i], pose_min_in_cam[i]

                local_normalize_dict["O_position_EE_min"] = list(pose_min_in_cam[:3])
                local_normalize_dict["O_position_EE_max"] = list(pose_max_in_cam[:3])
                # dont need to handle orientation since quaternions are already normalized

                external_wrench_min = np.array(local_normalize_dict['O_F_ext_EE_min'])
                external_wrench_max = np.array(local_normalize_dict['O_F_ext_EE_max'])
                external_force_min_in_cam = cam_tf_world[:3,:3] @ external_wrench_min[:3]
                external_force_max_in_cam = cam_tf_world[:3,:3] @ external_wrench_max[:3]
                external_torque_min_in_cam = cam_tf_world[:3,:3] @ external_wrench_min[3:]
                external_torque_max_in_cam = cam_tf_world[:3,:3] @ external_wrench_max[3:]
                # swap min and max if necessary
                for i in range(3):
                    if external_force_min_in_cam[i] > external_force_max_in_cam[i]:
                        external_force_min_in_cam[i], external_force_max_in_cam[i] = external_force_max_in_cam[i], external_force_min_in_cam[i]
                for i in range(3):
                    if external_torque_min_in_cam[i] > external_torque_max_in_cam[i]:
                        external_torque_min_in_cam[i], external_torque_max_in_cam[i] = external_torque_max_in_cam[i], external_torque_min_in_cam[i]
                
                local_normalize_dict['O_F_ext_EE_min'] = list(np.concatenate([external_force_min_in_cam, external_torque_min_in_cam]))
                local_normalize_dict['O_F_ext_EE_max'] = list(np.concatenate([external_force_max_in_cam, external_torque_max_in_cam]))
            
        self.save_hyperparameters()
    
    def prepare_data(self):
        pass

    def setup(self, stage=None):
        # run all this if in the fit stage
        if stage == 'fit':
            # assert that fit dataset dict is not none
            assert self.hparams.fit_dataset_dict is not None, "fit_dataset_dict is None"
            self._setup_fit_datasets()
        elif stage == 'test':
            self.test_dataset_dict_list = []
            for test_dataset_dict in self.hparams.test_dataset_dict_list:
                dataset_episodes_list = []
                dataset_episode_names_list = []
                dataset_path_basename_list = []
                settings_dict = test_dataset_dict['settings_dict']
                for episode_collection_dict in test_dataset_dict['dataset_episode_collection_dict_list']:
                    if episode_collection_dict['episode_idx_list'] == -1:
                        # then we want to use all episodes in the dataset
                        episode_dir_path = os.path.join(episode_collection_dict['dataset_dir_path'], 'episode_data')
                        # get the number of directories in the dataset directory under episode_data
                        num_episodes = len([name for name in os.listdir(episode_dir_path) if os.path.isdir(os.path.join(episode_dir_path, name))])
                        episode_collection_dict['episode_idx_list'] = list(range(num_episodes))
                    # load each episode as a separate dataset and concatenate later
                    for episode_idx in episode_collection_dict['episode_idx_list']:
                        episode = ContactDatasetEpisodic(episode_collection_dict['dataset_dir_path'], 
                                                                            episode_idx=episode_idx,
                                                                            model_type=self.hparams.model_type,
                                                                            im_resize=self.im_resize,
                                                                            max_depth_clip=self.hparams.max_depth_clip,
                                                                            downsampled_images=settings_dict['downsampled_images'],
                                                                            l515=settings_dict['l515'], 
                                                                            is_real_dataset=settings_dict['real_dataset'], 
                                                                            compensate_obj_gravity_dict=self.hparams.compensate_obj_gravity_dict,
                                                                            optical_flow_dict=self.hparams.optical_flow_dict,
                                                                            proprio_filter_dict=settings_dict['proprio_filter_dict'],
                                                                            cropping_dict=self.hparams.cropping_dict,
                                                                            context_frame_dict=self.hparams.context_frame_dict,
                                                                            global_proprio_input_dict=self.hparams.global_proprio_input_dict,
                                                                            local_proprio_input_dict=self.hparams.local_proprio_input_dict,
                                                                            blur_contact_map_dict=self.hparams.blur_contact_map_dict,
                                                                            proprio_history_plotting_dict = self.hparams.proprio_history_plotting_dict,
                                                                            is_anno_local=settings_dict['is_annotated_local'],
                                                                            is_anno_global=settings_dict['is_annotated_global'],
                                                                            viz_dataset=settings_dict['log_video'], viz_global_contact=settings_dict['viz_global_contact'],
                                                                            )
                        # get the name of the episode
                        dataset_episode_names_list.append(episode.get_episode_name(episode_idx))
                        dataset_episodes_list.append(episode) 

                        # get the path basename
                        dataset_path_basename_list.append(os.path.basename(episode_collection_dict['dataset_dir_path']))
                if len(dataset_episodes_list) > 1:
                    dataset = torch.utils.data.ConcatDataset(dataset_episodes_list)
                else:
                    dataset = dataset_episodes_list[0]
                
                # also need to append the name, so that we can identify the dataset later
                # num workers
                # batch size
                dataset_dict = {'dataset': dataset,
                                'dataset_path_basename_list': dataset_path_basename_list, 
                                'episode_names_list': dataset_episode_names_list,
                                }
                dataset_dict.update(settings_dict)
                self.test_dataset_dict_list.append(dataset_dict)

    def _setup_fit_datasets(self):
        self.num_non_curated_val_datasets = 1

        self.val_dataset_list = []
        self.val_dataset_dir_path_list = []
        val_concatdataset_list = []
        self.val_dataset_episode_idx_range_list = []
        fit_tot_num_episodes = self.hparams.fit_dataset_dict['train_num_episodes'] + self.hparams.fit_dataset_dict['val_num_episodes']
        cum_num_episodes = 0
        fit_dataset_idx = 0
        remainder_dataset = False
        # first construct the val dataset list
        while cum_num_episodes < self.hparams.fit_dataset_dict['val_num_episodes']:
            dataset_dir_path = self.hparams.fit_dataset_dict['dataset_dir_path_list'][fit_dataset_idx]
            dataset = ContactDatasetEpisodic(dataset_dir_path, 
                                            model_type=self.hparams.model_type,
                                            im_resize=self.im_resize,
                                            downsampled_images=self.hparams.fit_dataset_dict['downsampled_images'],
                                            max_depth_clip=self.hparams.max_depth_clip,
                                            l515=self.hparams.fit_dataset_dict['l515'], 
                                            is_real_dataset=self.hparams.fit_dataset_dict['real_dataset'], 
                                            compensate_obj_gravity_dict=self.hparams.compensate_obj_gravity_dict,
                                            optical_flow_dict=self.hparams.optical_flow_dict,
                                            cropping_dict=self.hparams.cropping_dict,
                                            context_frame_dict=self.hparams.context_frame_dict,
                                            global_proprio_input_dict=self.hparams.global_proprio_input_dict,
                                            local_proprio_input_dict=self.hparams.local_proprio_input_dict,
                                            blur_contact_map_dict=self.hparams.blur_contact_map_dict,
                                            proprio_history_plotting_dict = self.hparams.proprio_history_plotting_dict,
                                            is_anno_local=self.hparams.fit_dataset_dict['is_annotated_local'],
                                            is_anno_global=self.hparams.fit_dataset_dict['is_annotated_global'],
                                            viz_dataset=False, viz_global_contact=False, 
                                            )
            self.val_dataset_list.append(dataset)
            self.val_dataset_dir_path_list.append(dataset_dir_path)

            dataset_num_episodes = dataset.get_num_episodes()
            cum_num_episodes += dataset_num_episodes
            
            if cum_num_episodes > self.hparams.fit_dataset_dict['val_num_episodes']:
                # if we've gone over the number of episodes we want, then we need to split the dataset
                # and add the remaining episodes to the next dataset
                dataset_num_episodes -= (cum_num_episodes - self.hparams.fit_dataset_dict['val_num_episodes'])
                cum_num_episodes = self.hparams.fit_dataset_dict['val_num_episodes']
                remainder_dataset = True
            else:
                fit_dataset_idx += 1

            valid_episode_idxs = list(range(dataset_num_episodes))
            self.val_dataset_episode_idx_range_list.append([0,dataset_num_episodes])
            
            if self.only_contact:
                valid_idxs = dataset.get_contact_indices_for_episode_list(valid_episode_idxs)
            else:
                valid_idxs = dataset.get_indices_for_episode_list(valid_episode_idxs)
            val_concatdataset_list.append(torch.utils.data.Subset(dataset, valid_idxs)) 

        if len(val_concatdataset_list) > 1:
            self.val_dataset = torch.utils.data.ConcatDataset(val_concatdataset_list)    
        else:
            self.val_dataset = val_concatdataset_list[0]

        # now construct the train dataset list
        self.train_dataset_list = []
        self.train_dataset_dir_path_list = []
        train_concatdataset_list = []
        self.train_dataset_episode_idx_range_list = []
        while cum_num_episodes < fit_tot_num_episodes:
            dataset_dir_path = self.hparams.fit_dataset_dict['dataset_dir_path_list'][fit_dataset_idx]
            self.train_dataset_list.append(dataset)
            self.train_dataset_dir_path_list.append(dataset_dir_path)
            if remainder_dataset:
                # if we have a remainder dataset, then we need to add the remaining episodes to the current dataset
                # use previously loaded dataset to avoid reloading
                prev_dataset_num_episodes = dataset_num_episodes
                dataset_num_episodes = (dataset.get_num_episodes() - dataset_num_episodes) # dataset_num_episodes is the number of episodes in the previous dataset
                cum_num_episodes += dataset_num_episodes
            else:
                dataset = ContactDatasetEpisodic(dataset_dir_path, 
                                                model_type=self.hparams.model_type,
                                                im_resize=self.im_resize,
                                                max_depth_clip=self.hparams.max_depth_clip,
                                                downsampled_images=self.hparams.fit_dataset_dict['downsampled_images'],
                                                l515=self.hparams.fit_dataset_dict['l515'], 
                                                is_real_dataset=self.hparams.fit_dataset_dict['real_dataset'], 
                                                compensate_obj_gravity_dict=self.hparams.compensate_obj_gravity_dict,
                                                optical_flow_dict=self.hparams.optical_flow_dict,
                                                cropping_dict=self.hparams.cropping_dict,
                                                context_frame_dict=self.hparams.context_frame_dict,
                                                global_proprio_input_dict=self.hparams.global_proprio_input_dict,
                                                local_proprio_input_dict=self.hparams.local_proprio_input_dict,
                                                blur_contact_map_dict=self.hparams.blur_contact_map_dict,
                                                proprio_history_plotting_dict = self.hparams.proprio_history_plotting_dict,
                                                is_anno_local=self.hparams.fit_dataset_dict['is_annotated_local'],
                                                is_anno_global=self.hparams.fit_dataset_dict['is_annotated_global'],
                                                viz_dataset=False, viz_global_contact=False,
                                                )
                dataset_num_episodes = dataset.get_num_episodes()
                cum_num_episodes += dataset_num_episodes
            if cum_num_episodes > fit_tot_num_episodes:
                # if we've gone over the number of episodes we want, then we need to split the dataset
                # and add the remaining episodes to the next dataset
                dataset_num_episodes -= (cum_num_episodes - fit_tot_num_episodes)
                cum_num_episodes = fit_tot_num_episodes
            else:
                fit_dataset_idx += 1

            if remainder_dataset:
                # start slice from prev_dataset_num_episodes to get the remaining episodes
                train_episode_idxs = list(range(prev_dataset_num_episodes, dataset_num_episodes + prev_dataset_num_episodes))
                self.train_dataset_episode_idx_range_list.append([prev_dataset_num_episodes, dataset_num_episodes + prev_dataset_num_episodes])
                remainder_dataset = False
            else:
                train_episode_idxs = list(range(dataset_num_episodes))
                self.train_dataset_episode_idx_range_list.append([0, dataset_num_episodes])
            if self.only_contact:
                train_idxs = dataset.get_contact_indices_for_episode_list(train_episode_idxs)
            else:
                train_idxs = dataset.get_indices_for_episode_list(train_episode_idxs)
            train_concatdataset_list.append(torch.utils.data.Subset(dataset, train_idxs))
        if len(train_concatdataset_list) > 1:
            self.train_dataset = torch.utils.data.ConcatDataset(train_concatdataset_list)
        else:
            self.train_dataset = train_concatdataset_list[0]
        
        # # process the curated datasets
        self.curated_validation_dataset_dict_list = []
        curated_dataset_model_type = self.hparams.model_type
        if self.hparams.model_type == 'conditional_local': # dont want only contact for the curated datasets
            curated_dataset_model_type = 'local'
        for curated_validation_dataset_dict in self.hparams.curated_val_dataset_list:
            # add the dataset to the list if enable is true
            if curated_validation_dataset_dict['enable']:
                dataset_episodes_list = []
                dataset_episode_names_list = []
                dataset_path_basename_list = []
                settings_dict = curated_validation_dataset_dict['settings_dict']
                for episode_collection_dict in curated_validation_dataset_dict['dataset_episode_collection_dict_list']:
                    # load each episode as a separate dataset and concatenate later
                    for episode_idx in episode_collection_dict['episode_idx_list']:
                        episode = ContactDatasetEpisodic(episode_collection_dict['dataset_dir_path'], 
                                                                            episode_idx=episode_idx,
                                                                            model_type=curated_dataset_model_type,
                                                                            im_resize=self.im_resize,
                                                                            max_depth_clip=self.hparams.max_depth_clip,
                                                                            l515=settings_dict['l515'], 
                                                                            is_real_dataset=settings_dict['real_dataset'], 
                                                                            downsampled_images=settings_dict['downsampled_images'],
                                                                            compensate_obj_gravity_dict=self.hparams.compensate_obj_gravity_dict,
                                                                            optical_flow_dict=self.hparams.optical_flow_dict,
                                                                            cropping_dict=self.hparams.cropping_dict,
                                                                            context_frame_dict=self.hparams.context_frame_dict,
                                                                            global_proprio_input_dict=self.hparams.global_proprio_input_dict,
                                                                            local_proprio_input_dict=self.hparams.local_proprio_input_dict,
                                                                            blur_contact_map_dict=self.hparams.blur_contact_map_dict,
                                                                            proprio_history_plotting_dict = self.hparams.proprio_history_plotting_dict,
                                                                            is_anno_local=settings_dict['is_annotated_local'],
                                                                            is_anno_global=settings_dict['is_annotated_global'],
                                                                            viz_dataset=settings_dict['log_video'], viz_global_contact=settings_dict['viz_global_contact'],
                                                                            )
                        # get the name of the episode
                        dataset_episode_names_list.append(episode.get_episode_name(episode_idx))
                        dataset_episodes_list.append(episode) 

                        # get the path basename
                        dataset_path_basename_list.append(os.path.basename(episode_collection_dict['dataset_dir_path']))
                if len(dataset_episodes_list) > 1:
                    dataset = torch.utils.data.ConcatDataset(dataset_episodes_list)
                else:
                    dataset = dataset_episodes_list[0]
                
                # also need to append the name, so that we can identify the dataset later
                # and the epoch interval
                # num workers
                # batch size
                dataset_dict = {'dataset': dataset,
                                'dataset_path_basename_list': dataset_path_basename_list, 
                                'episode_names_list': dataset_episode_names_list,
                                'epoch_interval': curated_validation_dataset_dict['epoch_interval'],
                                }
                dataset_dict.update(settings_dict)
                self.curated_validation_dataset_dict_list.append(dataset_dict)
            
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.hparams.fit_dataset_dict['train_batch_size'], shuffle=True, num_workers=self.hparams.fit_dataset_dict['train_num_workers'], pin_memory=True)

    def val_dataloader(self):
        val_dataloader_list = []
        self.curated_val_dataloader_subset_idx_list = []
        self.curated_val_dataloader_length_list = []
        val_dataloader_list.append(DataLoader(self.val_dataset, batch_size=self.hparams.fit_dataset_dict['val_batch_size'], shuffle=False, num_workers=self.hparams.fit_dataset_dict['val_num_workers'], pin_memory=True))
        if self.trainer.current_epoch > 0: # first epoch is boring, skip
            for dataset_idx, dataset_dict in enumerate(self.curated_validation_dataset_dict_list):
                if self.trainer.current_epoch % dataset_dict['epoch_interval'] == 0:
                    val_dataloader_list.append(DataLoader(dataset_dict['dataset'], batch_size=dataset_dict['batch_size'], shuffle=False, num_workers=dataset_dict['num_workers'], pin_memory=True))
                    self.curated_val_dataloader_subset_idx_list.append(dataset_idx)
        self.curated_val_dataloader_length_list.extend([len(dataloader) for dataloader in val_dataloader_list])
        return val_dataloader_list

    def test_dataloader(self):
        test_dataloader_list = []
        self.test_dataloader_length_list = []
        for dataset_idx, dataset_dict in enumerate(self.test_dataset_dict_list):
            test_dataloader_list.append(DataLoader(dataset_dict['dataset'], batch_size=dataset_dict['batch_size'], shuffle=False, num_workers=dataset_dict['num_workers'], pin_memory=True))
        self.test_dataloader_length_list.extend([len(dataloader) for dataloader in test_dataloader_list])
        return test_dataloader_list
    
    def get_viz_input(self, batch, is_annotated_global=False, is_annotated_local=False, real_dataset=False):
        viz_input_dict = {}
        viz_input_dict['depth_image'] = batch[self.batch_keys_dict['depth_image_key']].squeeze(1) # B x 1 x H x W -> B x H x W
        viz_input_dict['color_image_path'] = batch[self.batch_keys_dict['color_image_path_key']] # B dim list of single element tuple 
        viz_input_dict['wrench_history'] = batch[self.batch_keys_dict['wrench_history_plotting_key']] # B x T x 6
        viz_input_dict['wrench_history_times'] = batch[self.batch_keys_dict['wrench_history_plotting_times_key']] # B x T 
        viz_input_dict['current_time'] = batch[self.batch_keys_dict['current_time_key']] # B x 1
        viz_input_dict['episode_idx'] = batch[self.batch_keys_dict['episode_idx_key']] # B
        viz_input_dict['grasped_object_mass'] = batch[self.batch_keys_dict['grasped_object_mass_key']] # B
        # Simulation or annotated locally
        if not real_dataset or is_annotated_local:
            viz_input_dict['num_contacts'] = batch[self.batch_keys_dict['num_contacts_key']]
            viz_input_dict['contact_pxls'] = batch[self.batch_keys_dict['contact_pxls_key']]
            # TODO only add this if blur is enabled in local model case because otherwise we can just use the target from batch
            viz_input_dict['contact_map'] = batch[self.batch_keys_dict['target_contact_map_key']] 
        # If real data, then add global label history to viz input
        if not real_dataset or is_annotated_global:
            # If return global labels -- batch['plotting_global_labels']
            if self.batch_keys_dict['global_label_plotting_key'] in batch.keys():
                # Visualize global labels
                viz_input_dict['global_label_history'] = batch[self.batch_keys_dict['global_label_plotting_key']]
        
        if self.hparams.model_type in ['local', 'conditional_local', 'joint']:
            if self.hparams.optical_flow_dict['enable']:
                flow_key = self.batch_keys_dict['flow_image_key']
                viz_input_dict['flow_image'] = batch[flow_key] # make sure this is B x 3 x H x W
            if self.hparams.context_frame_dict['enable']:
                viz_input_dict['context_depth_frame'] = batch[self.batch_keys_dict['context_frame_key']].squeeze(1) # get rid of time dimension
                viz_input_dict['context_color_path'] = batch[self.batch_keys_dict['context_color_path_key']]
                viz_input_dict['context_EE_pose_pxl_coordinates'] = batch[self.batch_keys_dict['context_pose_pxl_coordinate_key']]
            
        return viz_input_dict

    def get_episode_metadata(self, batch, dataloader_idx):
        episode_metadata = {}
        episode_metadata['episode_idx'] = batch['episode']
        episode_metadata['is_last_in_episode'] = batch['is_last_in_episode']
        return episode_metadata

    def on_before_batch_transfer(self, batch: dict, dataloader_idx: int):
        return_dict = {}
        episode_metadata_dict = self.get_episode_metadata(batch, dataloader_idx)
        return_dict['episode_metadata'] = episode_metadata_dict
        return_dict['timestamp'] = batch['im_times'].squeeze(1) # B x 1
        if 'global_contact' in batch.keys():
            return_dict['global_contact'] = batch['global_contact']

        real_dataset = False
        video_logging = False
        is_annotated_local = False
        is_annotated_global = False
        # TODO maintain a dataloader_list where I can query the annotation!!!
        if self.trainer.state.fn.value == 'fit':
            # because I split up each dataset phase into multiple dataloaders for each episode, need to check which dataset phase Im in
            if dataloader_idx > self.num_non_curated_val_datasets - 1:
                dataset_dict = self.curated_validation_dataset_dict_list[self.curated_val_dataloader_subset_idx_list[dataloader_idx - self.num_non_curated_val_datasets]]
                video_logging = dataset_dict['log_video']
                real_dataset = dataset_dict['real_dataset']
                is_annotated_global = dataset_dict['is_annotated_global']
                is_annotated_local = dataset_dict['is_annotated_local']
        elif self.trainer.state.fn.value == 'test':
            dataset_dict = self.test_dataset_dict_list[dataloader_idx]
            real_dataset = dataset_dict['real_dataset']
            video_logging = dataset_dict['log_video']

            is_annotated_local = dataset_dict['is_annotated_local']
            is_annotated_global = dataset_dict['is_annotated_global']
        else:
            raise NotImplementedError
        
        if self.hparams.model_type == 'global':
            model_input_dict = get_global_input(batch, self.hparams, self.batch_keys_dict)
            return_dict['model_input_dict'] = model_input_dict
            # should get target if not real dataset or if real dataset and is_anno_global
            if not real_dataset or is_annotated_global: 
                target_dict = get_global_target(batch, self.hparams, self.batch_keys_dict)
                return_dict['target_dict'] = target_dict
            if video_logging:
                video_input_dict = self.get_viz_input(batch, real_dataset=real_dataset, is_annotated_global=is_annotated_global, is_annotated_local=is_annotated_local)
                return_dict['viz_input_dict'] = video_input_dict
            return return_dict
        elif self.hparams.model_type in ['local', 'conditional_local']:
            model_input_dict =  get_local_input(batch, self.hparams, self.batch_keys_dict)
            return_dict['model_input_dict'] = model_input_dict
            # Alredy add annotation part for real dataset
            if not real_dataset or is_annotated_local:
                target_dict = get_local_target(batch, self.hparams, self.batch_keys_dict)
                return_dict['target_dict'] = target_dict
            if video_logging:
                video_input_dict = self.get_viz_input(batch, real_dataset=real_dataset, is_annotated_global=is_annotated_global, is_annotated_local=is_annotated_local)
                return_dict['viz_input_dict'] = video_input_dict
            return return_dict
        elif self.hparams.model_type == 'joint':
            return get_joint_input(batch, self.hparams, real_dataset)
        else:
            raise ValueError('model type must be local, conditional_local, global, or joint!')
    
    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        # only transfer the model input dict and target to the device
        batch['model_input_dict'] = {k: v.to(device) for k, v in batch['model_input_dict'].items()}
        if 'target_dict' in batch:
            batch['target_dict'] = {k: v.to(device) for k, v in batch['target_dict'].items()}
        # keep the rest of the batch on the cpu
        return batch
    
