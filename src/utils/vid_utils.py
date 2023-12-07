#%%
# import argparse
# import logging
# from pickletools import uint8
from pathlib import Path

import torch
import torchvision

import numpy as np

from moviepy.editor import *

import wandb 

import os

from typing import Tuple

import sys
sys.path.append(os.path.join(os.getcwd(), '../', '../'))
from src.plotting.plotmaker import MethodEvaluation, DatasetEvaluation, PlotMaker
from src.utils.viz_utils import *
from src.dataset.contact_dataset_episodic import depth_png_to_map

import glob
import natsort
import shutil 
import matplotlib.pyplot as plt
# make the plt figs large
plt.rcParams['figure.figsize'] = [24, 12]
#%%
class VidUtils():
    def __init__(self, run_id: str, real_dataset: bool = False, context_frame: bool = False, file_root_dir: str = 'videos', is_annotated_local: int=False, model_type: str='local', fps:int=10, resolution:Tuple[int, int]=(720, 1280)):
        self.real_dataset = real_dataset
        self.model_type = model_type
        self.fps = fps
        self.file_root_dir = file_root_dir
        self.is_anno_local = is_annotated_local
        self.vid_imgs = []
        self.epoch = 0
        self.run_id =  run_id
        self.resolution = resolution
        self.context_frame = context_frame
        if not os.path.exists(os.path.join(self.file_root_dir, self.run_id)):
            os.makedirs(os.path.join(self.file_root_dir, self.run_id))
        if real_dataset:
            self.caption = 'from left to right:\n' 
            self.caption += 'pred_depth_overlay_im\n'
            self.caption += 'pred_color_overlay_im'
            self.caption += 'forces_plot\n'
            self.caption += 'force_mag_plot\n'
            self.caption += 'torques_plot\n'
            self.caption += 'torque_mag_plot\n'
        else:
            self.caption = 'from left to right:\n' 
            self.caption += 'target_pred_depth_overlay_im\n'
            self.caption += 'target_pred_color_overlay_im\n'
            self.caption += 'forces_plot\n'
            self.caption += 'force_mag_plot\n'
            self.caption += 'torques_plot\n'
            self.caption += 'torque_mag_plot\n'

    def pad_to_resolution_height(self, im, resolution_h):
        # image dim is C x H x W
        # pad image to resolution_h
        im_h, im_w = im.shape[-2:]
        if im_h < resolution_h:
            pad_h = resolution_h - im_h
            pad_top = int(pad_h / 2)
            pad_bottom = pad_h - pad_top
            im = np.pad(im, ((0,0), (pad_top, pad_bottom), (0, 0)), 'constant')
        return im
    
    def add_frame(self, heatmap_dict, wrench_plots_dict):
        grid_im_tensors = []
        if self.real_dataset:
            # compose the video
            # change H x W x C np arrays to C x H x W tensors
            if not self.is_anno_local:
                grid_im_tensors.append(torch.moveaxis(torch.tensor(heatmap_dict['pred_depth_overlay_im']), -1, 0)) 
                grid_im_tensors.append(torch.moveaxis(torch.tensor(heatmap_dict['pred_color_overlay_im']), -1, 0))
            else:
                grid_im_tensors.append(torch.moveaxis(torch.tensor(heatmap_dict['circled_target_pred_depth_overlay_im']), -1, 0))
                grid_im_tensors.append(torch.moveaxis(torch.tensor(heatmap_dict['circled_target_pred_color_overlay_im']), -1, 0))

            if self.model_type == 'joint':
                grid_im_tensors.append(torch.moveaxis(torch.tensor(heatmap_dict['cond_pred_depth_overlay_im']), -1, 0)) 
                grid_im_tensors.append(torch.moveaxis(torch.tensor(heatmap_dict['cond_pred_color_overlay_im']), -1, 0))
            if self.context_frame:
                grid_im_tensors.append(torch.moveaxis(torch.tensor(heatmap_dict['context_depth_im']), -1, 0))
                grid_im_tensors.append(torch.moveaxis(torch.tensor(heatmap_dict['context_color_im']), -1, 0))
            grid_im_tensors.append(torch.moveaxis(torch.tensor(wrench_plots_dict['forces_plot']), -1, 0))
            # grid_im_tensors.append(torch.moveaxis(torch.tensor(wrench_plots_dict['force_mag_plot']), -1, 0))
            grid_im_tensors.append(torch.moveaxis(torch.tensor(wrench_plots_dict['torques_plot']), -1, 0))
            # grid_im_tensors.append(torch.moveaxis(torch.tensor(wrench_plots_dict['torque_mag_plot']), -1, 0))

            ## grid the images
            grid_im = np.array(torchvision.utils.make_grid(grid_im_tensors, nrow=2, padding=2, normalize=False, pad_value=255))
            if self.resolution is not None:
                grid_im = self.pad_to_resolution_height(grid_im, self.resolution[0])
            self.vid_imgs.append(grid_im)
        else:
            # change H x W x C np arrays to C x H x W tensors
            grid_im_tensors.append(torch.moveaxis(torch.tensor(heatmap_dict['circled_target_pred_depth_overlay_im']), -1, 0)) 
            grid_im_tensors.append(torch.moveaxis(torch.tensor(heatmap_dict['circled_target_pred_color_overlay_im']), -1, 0)) 
            if self.model_type == 'joint':
                grid_im_tensors.append(torch.moveaxis(torch.tensor(heatmap_dict['circled_target_cond_pred_depth_overlay_im']), -1, 0)) 
                grid_im_tensors.append(torch.moveaxis(torch.tensor(heatmap_dict['circled_target_cond_pred_color_overlay_im']), -1, 0)) 
            if self.context_frame:
                grid_im_tensors.append(torch.moveaxis(torch.tensor(heatmap_dict['context_depth_im']), -1, 0))
                grid_im_tensors.append(torch.moveaxis(torch.tensor(heatmap_dict['context_color_im']), -1, 0))
            grid_im_tensors.append(torch.moveaxis(torch.tensor(wrench_plots_dict['forces_plot']), -1, 0))
            # grid_im_tensors.append(torch.moveaxis(torch.tensor(wrench_plots_dict['force_mag_plot']), -1, 0))
            grid_im_tensors.append(torch.moveaxis(torch.tensor(wrench_plots_dict['torques_plot']), -1, 0))
            # grid_im_tensors.append(torch.moveaxis(torch.tensor(wrench_plots_dict['torque_mag_plot']), -1, 0))
            ## grid the images
            # NROW IS ACTUALLY NUMBER OF COLUMNS!!!
            grid_im = np.array(torchvision.utils.make_grid(grid_im_tensors, nrow=2, padding=2, normalize=False, pad_value=255))
            if self.resolution is not None:
                grid_im = self.pad_to_resolution_height(grid_im, self.resolution[0])
            self.vid_imgs.append(grid_im)

    def write_video(self, epoch = None, dataset_name = None, filename = ''):
        if len(self.vid_imgs) > 0:
            run_dir = os.path.join(self.file_root_dir, self.run_id)
            if epoch is not None:
                run_dir = os.path.join(run_dir, 'epoch_' + str(epoch))
            if dataset_name is not None:
                run_dir = os.path.join(run_dir, dataset_name)
            if not os.path.exists(run_dir):
                os.makedirs(run_dir)
            file_path = os.path.join(run_dir, filename + '.webm') # TODO change to mp4
            self.vid_imgs = np.moveaxis(np.stack(self.vid_imgs), 1, -1)
            clip = ImageSequenceClip(list(self.vid_imgs), fps = self.fps)
            # clip.write_videofile(file_path, codec='vp9', preset='medium', ffmpeg_params=["-lossless", str(1)])
            # using vp9 codec with crf to 5 and 8 threads
            # guide to vp9 and ffmpeg here: https://sites.google.com/a/webmproject.org/wiki/ffmpeg/vp9-encoding-guide
            clip.write_videofile(file_path, codec='libvpx-vp9', preset='medium', ffmpeg_params=["-quality", "good", "-crf", str(10), "-b:v", "0", "-threads", "8"])
        
            self.reset()
            return file_path
    
    def log_video(self, logging_step, prefix=None, postfix=None):
        wandb_video = self.return_wandb_video
        if wandb_video is None:
            pass
        else:
            if prefix is None:
                prefix = ''
            else:
                prefix += '_'
            if postfix is None:
                postfix = ''
            else:
                postfix = '_' + postfix
            wandb.log({prefix + "video" + postfix: wandb_video}, step = logging_step)
        self.reset()
    
    def return_wandb_video(self, file_name = ''):
        if len(self.vid_imgs) > 0:
            file_path = self.write_video(file_name)
            return wandb.Video(file_path, caption=self.caption, format="webm")
        else:
            self.reset()
            return None
        
    def add_ft_calibration_frame(self, color_frame, wrench_plots_dict):
        grid_im_tensors = []
        grid_im_tensors.append(torch.moveaxis(torch.tensor(color_frame), -1, 0))
        # pad with empty element
        grid_im_tensors.append(torch.full((3, 180, 320), 255, dtype=torch.uint8))
        grid_im_tensors.append(torch.moveaxis(torch.tensor(wrench_plots_dict['forces_plot']), -1, 0))
        grid_im_tensors.append(torch.moveaxis(torch.tensor(wrench_plots_dict['torques_plot']), -1, 0))
        grid_im_tensors.append(torch.moveaxis(torch.tensor(wrench_plots_dict['forces_difference_plot']), -1, 0))
        grid_im_tensors.append(torch.moveaxis(torch.tensor(wrench_plots_dict['torques_difference_plot']), -1, 0))

        ## grid the images
        self.vid_imgs.append(np.array(torchvision.utils.make_grid(grid_im_tensors, nrow=2, padding=2, normalize=False, pad_value=255)))
    
    def reset(self):
        self.vid_imgs = []

