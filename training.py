import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.cli import LightningArgumentParser, LightningCLI
import wandb

from src.dataset.contact_datamodule import ContactDataModule
from src.model.local_contact_model import LocalContactModel
from src.dataset.contact_dataclasses import CuratedValidationDatasetDict
import torch

import os

from jsonargparse import ArgumentParser, ActionConfigFile
class LocalContactTrainingCLI(LightningCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser):
        # second one overwrites the defaults in the first
        parser.default_config_files = ['./config/training.yaml', './config/scratch_training.yaml']
        parser.link_arguments('seed_everything', 'model.global_seed')
        parser.link_arguments('model.model_type', 'data.model_type')
        parser.link_arguments('model.im_resize_tuple', 'data.im_resize_tuple')

        parser.link_arguments('model.proprioception_input_dict.external_wrench', 'data.local_proprio_input_dict.external_wrench')
        parser.link_arguments('model.proprioception_input_dict.EE_pose', 'data.local_proprio_input_dict.EE_pose')
        parser.link_arguments('model.proprioception_input_dict.desired_joint_torques', 'data.local_proprio_input_dict.desired_joint_torques')
        parser.link_arguments('model.proprioception_input_dict.joint_velocities', 'data.local_proprio_input_dict.joint_velocities')
        parser.link_arguments('model.proprioception_input_dict.measured_joint_torques', 'data.local_proprio_input_dict.measured_joint_torques')
        parser.link_arguments('model.proprioception_input_dict.rotation_representation', 'data.local_proprio_input_dict.rotation_representation')
        parser.link_arguments('model.proprioception_input_dict.in_cam_frame_dict.enable', 'data.local_proprio_input_dict.in_cam_frame_dict.enable')
        parser.link_arguments('model.proprioception_input_dict.in_cam_frame_dict.C_R_W', 'data.local_proprio_input_dict.in_cam_frame_dict.C_R_W')
        parser.link_arguments('model.proprioception_input_dict.in_cam_frame_dict.C_t_W', 'data.local_proprio_input_dict.in_cam_frame_dict.C_t_W')
        parser.link_arguments('model.proprioception_input_dict.history_dict.enable', 'data.local_proprio_input_dict.history_dict.enable')
        parser.link_arguments('model.proprioception_input_dict.history_dict.time_window', 'data.local_proprio_input_dict.history_dict.time_window')
        parser.link_arguments('model.proprioception_input_dict.history_dict.sample_freq', 'data.local_proprio_input_dict.history_dict.sample_freq')
                 
        parser.link_arguments('model.compensate_obj_gravity_dict.enable', 'data.compensate_obj_gravity_dict.enable')
        parser.link_arguments('model.compensate_obj_gravity_dict.EE_pos_x_objCoM', 'data.compensate_obj_gravity_dict.EE_pos_x_objCoM')
        parser.link_arguments('model.compensate_obj_gravity_dict.EE_pos_y_objCoM', 'data.compensate_obj_gravity_dict.EE_pos_y_objCoM')
        parser.link_arguments('model.compensate_obj_gravity_dict.EE_pos_z_objCoM', 'data.compensate_obj_gravity_dict.EE_pos_z_objCoM')
        parser.link_arguments('model.compensate_obj_gravity_dict.start_idx', 'data.compensate_obj_gravity_dict.start_idx')
        parser.link_arguments('model.compensate_obj_gravity_dict.num_idxs', 'data.compensate_obj_gravity_dict.num_idxs')

        parser.link_arguments('model.normalize_dict.enable', 'data.local_normalize_dict.enable')
        parser.link_arguments('model.normalize_dict.O_position_EE_min', 'data.local_normalize_dict.O_position_EE_min')
        parser.link_arguments('model.normalize_dict.O_position_EE_max', 'data.local_normalize_dict.O_position_EE_max')
        parser.link_arguments('model.normalize_dict.O_F_ext_EE_min', 'data.local_normalize_dict.O_F_ext_EE_min')
        parser.link_arguments('model.normalize_dict.O_F_ext_EE_max', 'data.local_normalize_dict.O_F_ext_EE_max')
        parser.link_arguments('model.normalize_dict.desired_torques_max', 'data.local_normalize_dict.desired_torques_max')
        parser.link_arguments('model.normalize_dict.desired_torques_min', 'data.local_normalize_dict.desired_torques_min')
        parser.link_arguments('model.normalize_dict.joint_velocities_min', 'data.local_normalize_dict.joint_velocities_min')
        parser.link_arguments('model.normalize_dict.joint_velocities_max', 'data.local_normalize_dict.joint_velocities_max')
        
        parser.link_arguments('model.optical_flow_dict.enable', 'data.optical_flow_dict.enable')
        parser.link_arguments('model.optical_flow_dict.use_image', 'data.optical_flow_dict.use_image')
        parser.link_arguments('model.optical_flow_dict.normalize', 'data.optical_flow_dict.normalize')
        parser.link_arguments('model.optical_flow_dict.max_flow_norm', 'data.optical_flow_dict.max_flow_norm')
        parser.link_arguments('model.optical_flow_dict.fill_holes', 'data.optical_flow_dict.fill_holes')
        
        parser.link_arguments('model.context_frame_dict.enable', 'data.context_frame_dict.enable')
        
        parser.link_arguments('model.cropping_dict.enable', 'data.cropping_dict.enable')
        parser.link_arguments('model.cropping_dict.bb_height', 'data.cropping_dict.bb_height')
        parser.link_arguments('model.cropping_dict.bb_width', 'data.cropping_dict.bb_width')
        parser.link_arguments('model.cropping_dict.down_scale', 'data.cropping_dict.down_scale')

        parser.link_arguments('model.blur_contact_map_dict.enable', 'data.blur_contact_map_dict.enable')
        parser.link_arguments('model.blur_contact_map_dict.kernel_size', 'data.blur_contact_map_dict.kernel_size')
        parser.link_arguments('model.blur_contact_map_dict.sigma', 'data.blur_contact_map_dict.sigma')

        parser.link_arguments('model.add_noise_dict.enable', 'data.add_noise_dict.enable')
        parser.link_arguments('model.add_noise_dict.noise_type', 'data.add_noise_dict.noise_type')
        parser.link_arguments('model.add_noise_dict.FT_dict', 'data.add_noise_dict.FT_dict')
        parser.link_arguments('model.add_noise_dict.desired_joint_torque_dict', 'data.add_noise_dict.desired_joint_torque_dict')
        parser.link_arguments('model.add_noise_dict.joint_velo_dict', 'data.add_noise_dict.joint_velo_dict')

    def before_instantiate_classes(self):
        pass

def cli_main(args: list = None):
    cli = LocalContactTrainingCLI(LocalContactModel, ContactDataModule, args=args, save_config_overwrite=True)
if __name__ == '__main__':  
    cli_main()
