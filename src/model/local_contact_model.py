import sys

import pytorch_lightning as pl
import torch
import torchmetrics
from torch import nn, optim
from torch.nn import functional as F
from torchmetrics.classification import (BinaryAccuracy, BinaryAUROC, BinaryPrecisionRecallCurve, BinaryAveragePrecision, BinaryROC,
                                         BinaryF1Score,
                                         BinaryPrecision, BinaryRecall, Dice)
from torchmetrics.aggregation import MeanMetric

sys.path.append('../')
import ast
import types
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

import wandb
from src.model.coordconv import AddCoords
from src.model.unet_fusion import UNetFusion
from src.utils.nms_metric import NMS_match
from src.utils.table_utils import TableUtils
from src.utils.vid_utils import VidUtils
from src.utils.viz_utils import VizUtils
# from src.dataset.contact_datamodule import NormalizeDict, CompensateObjGravityDict
# import the normalizedict and compensateobjgravitydict dataclasses from src.dataset.contact_datamodule
from src.dataset.contact_dataclasses import NormalizeDict, CompensateObjGravityDict, OpticalFlowDict, CroppingDict, ContextFrameDict, BlurContactMapDict, InCamFrameDict, ProprioceptionInputDict, NMSDict, AddNoiseDict, EvaluationMetricsSettingsDict
from src.dataset.contact_dataset_episodic import ContactDatasetEpisodic
import datetime
class LocalContactModel(pl.LightningModule):
    def __init__(self, 
                lr: float=1e-3, weight_decay: float=0.0, global_seed: int=0,
                model_type: str='local',
                blur_contact_map_dict: BlurContactMapDict = BlurContactMapDict(),
                im_resize_tuple: Tuple[int, int] = (240, 320),
                pose_hidden_list: List[int] = [64, 64],
                wrench_hidden_list: List[int] = [64, 64],
                proprioception_input_dict: ProprioceptionInputDict = ProprioceptionInputDict(),
                fuse_inputs: bool = False,
                compensate_obj_gravity_dict: CompensateObjGravityDict=CompensateObjGravityDict(),
                normalize_dict: NormalizeDict = NormalizeDict(), 
                bilinear: bool = False,
                optical_flow_dict: OpticalFlowDict = OpticalFlowDict(),
                context_frame_dict: ContextFrameDict = ContextFrameDict(),
                cropping_dict: CroppingDict = CroppingDict(),
                nms_dict: NMSDict = NMSDict(),
                add_noise_dict: AddNoiseDict = AddNoiseDict(),
                max_depth_clip: float = 2.0,
                viz_utils_dict: dict=None, vid_utils_dict: dict=None,
                eval_metrics_settings_dict: EvaluationMetricsSettingsDict = EvaluationMetricsSettingsDict(),
                **kwargs):
        super().__init__()
        
        # only run this if compensate_obj_gravity_dict is a dataclass
        if isinstance(proprioception_input_dict, ProprioceptionInputDict):
            proprioception_input_dict = asdict(proprioception_input_dict)
        if isinstance(compensate_obj_gravity_dict, CompensateObjGravityDict):
            compensate_obj_gravity_dict = asdict(compensate_obj_gravity_dict)
        if isinstance(blur_contact_map_dict, BlurContactMapDict):
            blur_contact_map_dict = asdict(blur_contact_map_dict)
        if isinstance(optical_flow_dict, OpticalFlowDict):
            optical_flow_dict = asdict(optical_flow_dict)
        if isinstance(context_frame_dict, ContextFrameDict):
            context_frame_dict = asdict(context_frame_dict)
        if isinstance(cropping_dict, CroppingDict):
            cropping_dict = asdict(cropping_dict)
        if isinstance(normalize_dict, NormalizeDict):
            normalize_dict = asdict(normalize_dict)
        if isinstance(nms_dict, NMSDict):
            nms_dict = asdict(nms_dict)
        if isinstance(add_noise_dict, AddNoiseDict):
            add_noise_dict = asdict(add_noise_dict)
        if isinstance(eval_metrics_settings_dict, EvaluationMetricsSettingsDict):
            eval_metrics_settings_dict = asdict(eval_metrics_settings_dict)
        if cropping_dict['enable']:
            eval_metrics_settings_dict['evaluate_on_EE_cropped_dict']['enable'] = False
        if proprioception_input_dict['history_dict']['enable']:
            self.hparams.time_dim = int(proprioception_input_dict['history_dict']['time_window'] * proprioception_input_dict['history_dict']['sample_freq'])
        else:
            self.hparams.time_dim = 1
        pose_dim = 0
        if proprioception_input_dict['EE_pose']:
            pose_dim += 7
        pose_dim = pose_dim*self.hparams.time_dim
        pose_enc_dim = pose_dim
        wrench_dim = 0
        fuse_ft = False
        if proprioception_input_dict['external_wrench']:
            fuse_ft = True
            wrench_dim += 6
        if proprioception_input_dict['desired_joint_torques']:
            fuse_ft = True
            wrench_dim += 7
        wrench_dim = wrench_dim*self.hparams.time_dim
        wrench_enc_dim = wrench_dim

        input_channels = 1
        if optical_flow_dict['enable']:
            if optical_flow_dict['use_image']:
                input_channels += 3
            else:
                input_channels += 2
        if cropping_dict['enable']:
            input_channels += 3 # For coordinate convolution
        output_channels = 1
        self.im_size = im_resize_tuple
        
        self.model = UNetFusion(context_frame_dict, cropping_dict, im_resize_tuple, input_channels, pose_dim, pose_enc_dim, wrench_dim, wrench_enc_dim, 
        pose_hidden_list=pose_hidden_list, wrench_hidden_list=wrench_hidden_list,
        fuse_pose=proprioception_input_dict['EE_pose'], fuse_ft=fuse_ft, fuse_inputs=fuse_inputs, output_channels=output_channels, bilinear=bilinear)
        
        self.custom_logger_index = 1
    
        # keep this at the bottom so that everything above is saved into hparams!
        self.save_hyperparameters()

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), self.hparams.lr, weight_decay=self.hparams.weight_decay)
        return optimizer
    
    def on_fit_start(self) -> None:
        self.train_prefix = 'train_'
        self.train_NMS = NMS_match(**self.hparams.nms_dict)
        self.train_NMS.prefix = self.train_prefix
        self.train_NMS.to(self.device)

        self.val_prefix = 'val_'
        # self.val_metrics = torchmetrics.MetricCollection([
        #     Dice(),
        #     # NMS_L2_norm(half_window_size=self.hparams.NMS_half_window_size),
        #     # BinaryAUROC(),
        #     # BinaryAveragePrecision(),
        # ], compute_groups=True, prefix=self.val_prefix)
        # self.val_metrics.to(self.device)
        # # save these metrics to the PL state dict
        # self.val_metrics.persistent = True

        # NMS metrics initialization
        self.val_NMS_metrics = NMS_match(**self.hparams.nms_dict)
        self.val_NMS_metrics.prefix = self.val_prefix
        self.val_NMS_metrics.to(self.device)

        # self.val_baseline_metrics = torchmetrics.MetricCollection([
        #     Dice(),
        #     # BinaryAUROC(),
        #     # BinaryAveragePrecision(),
        # ], compute_groups=True, prefix=self.val_prefix + 'baseline_')
        # self.val_baseline_metrics.to(self.device)

        self._initialize_viz_utils()
        self._initialize_vid_utils()

        # log the train val dataset paths and episode idxs
        train_dataset_dir_path_list = self.trainer.datamodule.train_dataset_dir_path_list
        val_dataset_dir_path_list = self.trainer.datamodule.val_dataset_dir_path_list
        self.logger.log_hyperparams(params={'train_dataset_dir_path_list': train_dataset_dir_path_list, 'val_dataset_dir_path_list': val_dataset_dir_path_list})
        train_dataset_episode_idx_range_list = self.trainer.datamodule.train_dataset_episode_idx_range_list  
        val_dataset_episode_idx_range_list = self.trainer.datamodule.val_dataset_episode_idx_range_list
        self.logger.log_hyperparams(params={'train_dataset_episode_idx_range_list': train_dataset_episode_idx_range_list, 'val_dataset_episode_idx_range_list': val_dataset_episode_idx_range_list})

        # curated validation dataset related stuff
        self.curated_validation_dataset_metrics_dict_list = []
        for curated_validation_dataset_dict in self.trainer.datamodule.curated_validation_dataset_dict_list:
            # create a list of dicts for each dataset where the dict keys are the episode names
            episode_metrics_dict = {}
            for episode_name in curated_validation_dataset_dict['episode_names_list']:
                metric_dict = {}
                metric_dict['metric_collection'] = torchmetrics.MetricCollection([
                    Dice(),
                    # NMS_L2_norm(half_window_size=self.hparams.NMS_half_window_size),
                    # BinaryAUROC(),
                    # BinaryAveragePrecision(),
                ], compute_groups=True, prefix=self.val_prefix + episode_name + '_')
                metric_dict['metric_collection'].to(self.device)
                # save these metrics to the PL state dict
                metric_dict['metric_collection'].persistent = True
                # NMS metrics initialization
                metric_dict['NMS_metrics'] = NMS_match(**self.hparams.nms_dict)
                metric_dict['NMS_metrics'].prefix = self.val_prefix + episode_name + '_'
                metric_dict['NMS_metrics'].to(self.device)
                metric_dict['NMS_metrics'].persistent = True

                episode_metrics_dict[episode_name] = metric_dict
            self.curated_validation_dataset_metrics_dict_list.append(episode_metrics_dict)
        
    def _initialize_viz_utils(self):
        # initialize the viz utils
        self.hparams.viz_utils_dict['max_flow_norm'] = self.hparams.optical_flow_dict['max_flow_norm']
        self.hparams.viz_utils_dict['max_depth_clip'] = self.hparams.max_depth_clip
        self.hparams.viz_utils_dict['im_resize'] = self.hparams.im_resize_tuple
        if self.hparams.viz_utils_dict['normalize_local_prob']:
            # set the max local prob to the max prob of the blur kernel
            if self.hparams.blur_contact_map_dict['enable']:
                max_local_prob = np.max(cv2.getGaussianKernel(self.hparams.blur_contact_map_dict['kernel_size'], 0))**2
            else:
                max_local_prob = 1.0
            self.hparams.viz_utils_dict['max_local_prob'] = max_local_prob
        
        # consider making this the kernel size instead....
        # if self.hparams.viz_utils_dict['circle_dict']['enable']:
            # self.hparams.viz_utils_dict['circle_dict']['radius'] = self.hparams.nms_dict['window_radius']

        self.viz_utils = VizUtils(**self.hparams.viz_utils_dict)
    
    def _initialize_vid_utils(self):
        # initialize the vid utils
        # if dummy run then experiment id will be a method...
        self.hparams.vid_utils_dict['context_frame'] = self.hparams.context_frame_dict['enable']
        self.hparams.vid_utils_dict['model_type'] = self.hparams.model_type
        # if not isinstance(self.trainer.logger.experiment.id, types.MethodType):
        if hasattr(self.trainer.logger, 'experiment'):
            self.hparams.vid_utils_dict['run_id'] = self.trainer.logger.experiment.id
        elif hasattr(self.trainer.logger, 'run_name'):
            self.hparams.vid_utils_dict['run_id'] = self.trainer.logger.run_name
        else:
            raise ValueError('Could not find run id in logger')
            
            # self.hparams.vid_utils_dict['run_id'] = 'dummy_run'
        self.vid_utils = VidUtils(**self.hparams.vid_utils_dict)

    def training_step(self, batch, batch_idx):
        self.train_batch_idx = batch_idx
        
        input_dict = batch['model_input_dict']
        target_dict = batch['target_dict']
        y = target_dict['y'] # this is also cropped if cropping is enabled
        y_hat_logit = self(input_dict)  # the unet will pad out the prediction to the original size in case of cropping
        y_hat = torch.sigmoid(y_hat_logit)

        # for cropping must necessarily evaluate train loss on cropped target
        train_loss = F.binary_cross_entropy_with_logits(y_hat_logit, y)
        # however for the non training losses, we can evaluate on the original target
        # in the case of cropping, add another train loss for the original target that does not get backpropagated

        # these metrics are for evaluation only
        with torch.no_grad():
            if self.hparams.cropping_dict['enable']:
                self.log('train_loss_cropped', train_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
                # swap the y_hat for the padded version
                EE_pxls = input_dict['EE_pose_pxl'] # B x 2
                y_hat_logit = self.pad_cropped_logit(y_hat_logit, EE_pxls) # back to B x 1 x H_OG x W_OG
                y_hat = torch.sigmoid(y_hat_logit)
            
            unmodified_y = target_dict['unmodified_y']
            
            # Calculate NMS
            NMS_pixel_coord_list = self.prediction_heatmap_to_NMS_pixel_coord_list(y_hat_logit, **self.hparams.nms_dict)
            self.train_NMS.update(NMS_pixel_coord_list, y_hat, unmodified_y.int().cpu())

            train_loss_unmodified = F.binary_cross_entropy_with_logits(y_hat_logit, unmodified_y)
            self.log('train_loss', train_loss_unmodified, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            # self.train_metrics(y_hat_logit, unmodified_y.int())
            # self.log_dict(self.train_metrics, on_step=True, on_epoch=True, prog_bar=False, logger=True)
            
        return train_loss

    def on_train_epoch_end(self):
        # if using torch metrics, they are reset automatically
        NMS_metrics_dict = self.train_NMS.compute()
        # add '_epoch' postfix to each key
        self.log_dict({key + '_epoch': value for key, value in NMS_metrics_dict.items()}, prog_bar=False)
        self.train_NMS.reset()
    
    def on_validation_epoch_start(self):
        self.eval_table_created = False
        self.within_eval_dataset_episode_idx = 0

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        # COMPLETE HACK PROBABLY SHOULD NOT BE DOING THIS
        self.trainer._results.dataloader_idx = dataloader_idx

        self.val_batch_idx = batch_idx
        
        model_input_dict = batch['model_input_dict']

        target_dict = None
        if 'target_dict' in batch:
            target_dict = batch['target_dict']
        eval_dict = self._shared_eval_step(model_input_dict, target_dict = target_dict)
        
        if dataloader_idx < self.trainer.datamodule.num_non_curated_val_datasets: # on train val sim dataset
            if 'cropped_loss' in eval_dict:
                self.log('val_loss_cropped', eval_dict['cropped_loss'], on_step=False, on_epoch=True, prog_bar=True, logger=True, add_dataloader_idx=False)

            self.val_NMS_metrics.update(eval_dict['NMS_pixel_coord_list'], eval_dict['y_hat'], eval_dict['unmodified_y'].int().cpu())
            
            self.log('val_loss', eval_dict['loss'], on_step=False, on_epoch=True, prog_bar=True, logger=True, add_dataloader_idx=False)
            # self.val_metrics(eval_dict['y_hat_logit'], eval_dict['unmodified_y'].int())
            # self.log_dict(self.val_metrics, on_step=False, on_epoch=True, prog_bar=False, logger=True, add_dataloader_idx=False)
            
        else: # curated datasets!
            # get the dataset dict from the datamodule indexed by the dataloader_idx
            eval_dataloader_idx = self.trainer.datamodule.curated_val_dataloader_subset_idx_list[dataloader_idx - self.trainer.datamodule.num_non_curated_val_datasets]
            dataset_dict = self.trainer.datamodule.curated_validation_dataset_dict_list[eval_dataloader_idx]

            self._eval_dataset(batch, dataset_dict, eval_dict, eval_dataloader_idx)
    
    def on_validation_epoch_end(self):
        self.log_dict({'epoch': self.current_epoch})
        # if using torch metrics, they are reset automatically
        NMS_metrics_dict = self.val_NMS_metrics.compute()
        # add '_epoch' postfix to each key
        self.log_dict({key + '_epoch': value for key, value in NMS_metrics_dict.items()}, prog_bar=True)
        self.val_NMS_metrics.reset()

    def _eval_dataset(self, batch, dataset_dict, eval_dict, eval_dataloader_idx):
        # in this function, we want to loop through an evaluation dataset consisting of multiple episodes
        # somehow track the episode index so that we know when to save the video as well as move on to the next episodes metric
        dataset_name = dataset_dict['dataset_name']
        real_dataset = dataset_dict['real_dataset']
        self.vid_utils.real_dataset = real_dataset # default to false
        self.viz_utils.real_dataset = real_dataset
        episode_metatadata_dict = batch['episode_metadata']
        
        # number of episodes in the dataset
        num_episodes = len(dataset_dict['episode_names_list'])
        episode_name = dataset_dict['episode_names_list'][self.within_eval_dataset_episode_idx]
        dataset_path_basename = dataset_dict['dataset_path_basename_list'][self.within_eval_dataset_episode_idx]

        # get trainer stage (if fit or test)
        # needed to access the right metrics
        stage = self.trainer.state.stage
        if stage == 'test':
            self.trainer.loggers[self.custom_logger_index].current_episode_dataset_name = dataset_path_basename
            self.trainer.loggers[self.custom_logger_index].current_episode = episode_name

        if stage == 'fit' or stage == 'validate':
            if not self.eval_table_created:
                columns = ['epoch', 'dataset_path_basename', 'episode_name', 'episode_index',]
                if dataset_dict['log_video']:
                    columns += ['video']
                # if dataset_dict['evaluate_metrics']:
                    # columns += ['metrics']
                self.eval_table = TableUtils(columns=columns, name=dataset_name + '_table')
            self.eval_table_created = True

        viz_input_dict = None
        if dataset_dict['log_video']:
            # if 'viz_input_dict' in batch:
            viz_input_dict = batch['viz_input_dict']
            viz_input_dict['EE_pose_pxl_coordinates'] = eval_dict['EE_pose_pxl_coordinates']
        
        eval_output_dict = self._eval_episodic_dataset_step(episode_name, eval_dataloader_idx, eval_dict, dataset_dict, viz_input_dict=viz_input_dict)

        # if this is the last batch in the episode, save the video and add it to the table
        if episode_metatadata_dict['is_last_in_episode']:
            if dataset_dict['log_video']:
                if stage == 'fit' or stage == 'validate':
                    # save the video
                    video_filepath = self.vid_utils.write_video(epoch = self.current_epoch, dataset_name = dataset_name, filename = episode_name)
                    # return the wandb video
                    wandb_video = wandb.Video(video_filepath, format="webm")

                    # add the video to the table
                    self.eval_table.add_row([self.current_epoch, dataset_path_basename, episode_name, episode_metatadata_dict['episode_idx'], wandb_video])
                elif stage == 'test':
                    self.viz_utils.idx = 0
            # reset the metrics if test stage
            if stage == 'test': 
                self.trainer.loggers[self.custom_logger_index].end_of_episode = True # logger will auto reset after saving
                if dataset_dict['evaluate_metrics']:
                    if not self.hparams.nms_dict['per_sample_metrics']:
                        nms_metric_dict = self.test_NMS_metrics.compute()
                        self.log_dict(nms_metric_dict, prog_bar=False, on_step=True, on_epoch=False, logger=True, add_dataloader_idx=False)
                        self.log(self.test_prefix + '_loss', self.test_bce_loss/self.episode_sample_count, on_step=True, on_epoch=False, prog_bar=True, logger=True, add_dataloader_idx=False)
                        self.log(self.test_prefix + '_loss_cumsum', self.test_bce_loss, on_step=True, on_epoch=False, prog_bar=True, logger=True, add_dataloader_idx=False)
                        if self.hparams.cropping_dict['enable'] or self.hparams.eval_metrics_settings_dict['evaluate_on_EE_cropped_dict']['enable']:
                            self.log(self.test_prefix + '_loss_cropped', self.test_bce_loss_cropped/self.episode_sample_count, on_step=True, on_epoch=False, prog_bar=True, logger=True, add_dataloader_idx=False)
                            self.log(self.test_prefix + '_loss_cropped_cumsum', self.test_bce_loss_cropped, on_step=True, on_epoch=False, prog_bar=True, logger=True, add_dataloader_idx=False)
                            self.test_bce_loss_cropped = 0
                        self.test_bce_loss = 0
                        self.episode_sample_count = 0
                        # clear the logger of the current episode's metrics after its dumped csv
                        
                    self.test_NMS_metrics.reset()
                    # test_metrics_cropped = self.test_metrics_cropped.compute()
                    # self.log_dict(test_metrics_cropped, on_step=True, prog_bar=False, logger=True, add_dataloader_idx=False)
                    # self.test_metrics_cropped.reset()
            
            # if this is last episode in the dataset, log the table
            if self.within_eval_dataset_episode_idx == num_episodes - 1:
                if dataset_dict['log_video']: # or dataset_dict['evaluate_metrics']:
                    if stage == 'fit' or stage == 'validate':
                        # log the table
                        self.trainer.logger.log_table(key=self.eval_table.name, columns=self.eval_table.columns, data=self.eval_table.return_data())
                        self.eval_table_created = False
                
                self.within_eval_dataset_episode_idx = 0
            else:
                # increment the episode index
                self.within_eval_dataset_episode_idx += 1

    def _eval_episodic_dataset_step(self, episode_name, curated_dataloader_idx, eval_dict, dataset_dict, viz_input_dict=None):
        stage = self.trainer.state.stage

        eval_output_dict = {}
        if dataset_dict['evaluate_metrics']:
            # pass # not implemented yet
            if stage == 'fit' or stage == 'validate':
                raise NotImplementedError   

            elif stage == 'test':
                self.test_NMS_metrics.prefix = self.test_prefix + '_'
                if self.hparams.nms_dict['per_sample_metrics']:
                    self.log(self.test_prefix + '_timestamp', eval_dict['timestamp'], on_step=True, prog_bar=False, add_dataloader_idx=False)
                    self.log(self.test_prefix + '_global_contact', eval_dict['global_contact'], on_step=True, prog_bar=False, add_dataloader_idx=False)
                
                    # self.log(self.test_prefix + episode_name + '_loss', eval_dict['loss'], on_step=True, prog_bar=True, logger=True, add_dataloader_idx=False)
                    self.log(self.test_prefix + '_loss', eval_dict['loss'], on_step=True, on_epoch=False, prog_bar=True, logger=True, add_dataloader_idx=False)
                    
                    # self.test_NMS_metrics.prefix = self.test_prefix + episode_name + '_'
                    
                    # put all nms_pixel_coord_list elements on self.device
                    # eval_dict['NMS_pixel_coord_list'] = [nms_pixel_coord.to(self.device) for nms_pixel_coord in eval_dict['NMS_pixel_coord_list']]
                    # nms_metric_dict = self.test_NMS_metrics(eval_dict['NMS_pixel_coord_list'], eval_dict['y_hat'], eval_dict['unmodified_y'].int().cpu())
                    self.test_NMS_metrics.update(eval_dict['NMS_pixel_coord_list'], eval_dict['y_hat'], eval_dict['unmodified_y'].int().cpu())
                    nms_metric_dict = self.test_NMS_metrics.compute()
                    self.log_dict(nms_metric_dict, on_step=True, on_epoch=False, prog_bar=False, logger=True, add_dataloader_idx=False)
                
                    if 'cropped_loss' in eval_dict:
                        self.log(self.test_prefix + '_loss_cropped', eval_dict['cropped_loss'], on_step=True, on_epoch=False, prog_bar=True, logger=True, add_dataloader_idx=False)
                        # self.log(self.test_prefix + episode_name + '_loss_cropped', eval_dict['cropped_loss'], on_step=True, prog_bar=True, logger=True, add_dataloader_idx=False)
                    
                        # self.test_metrics_cropped.update(eval_dict['y_hat_logit_cropped'], eval_dict['y_cropped'].int())
                        # self.log_dict(test_metrics_cropped, on_step=True, on_epoch=False, prog_bar=False, logger=True, add_dataloader_idx=False)
                else:
                    self.episode_sample_count += eval_dict['y_hat'].shape[0]
                    self.test_bce_loss += eval_dict['loss']
                    self.test_NMS_metrics.update(eval_dict['NMS_pixel_coord_list'], eval_dict['y_hat'], eval_dict['unmodified_y'].int().cpu())
                    if 'cropped_loss' in eval_dict:
                        self.test_bce_loss_cropped += eval_dict['cropped_loss']

        if dataset_dict['log_video']:
            # MOVE TO CPU
            y_hat = torch.sigmoid(eval_dict['y_hat_logit']).cpu()
            # hardcode way to specify our batch size is 0
            idx_in_batch = 0
            # timestamp = viz_input_dict['im_times'][idx_in_batch] # B dim 
            viz_output_dict = self._viz_step(y_hat, viz_input_dict, real_dataset=dataset_dict['real_dataset'], idx_in_batch=idx_in_batch, NMS_pixel_coord_list=eval_dict['NMS_pixel_coord_list'])
            # add the visualization dict frames to the video logger
            if stage == 'fit' or stage == 'validate': # for test we will write frames individually with viz_utils
                self.vid_utils.add_frame(viz_output_dict['heatmap_dict'], viz_output_dict['wrench_plots_dict'])
        return eval_output_dict
   
    def _shared_eval_step(self, model_input_dict, target_dict = None):
        eval_dict = {}
        y_hat_logit = self(model_input_dict)


        eval_dict['EE_pose_pxl_coordinates'] = None
        if target_dict is not None: 
            if self.hparams.eval_metrics_settings_dict['evaluate_on_EE_cropped_dict']['enable']:
                # crop the y_hat_logit to the EE pose
                EE_pxls = model_input_dict['EE_pose_pxl'] # B x 2
                eval_dict['EE_pose_pxl_coordinates'] = EE_pxls
                # go through the batch and crop the y_hat_logit to the EE pose
                # create a y_hat_logit_cropped tensor that has the same shape as y_hat_logit except for the H and W which should be based on self.hparams.eval_metrics_settings_dict['evaluate_on_EE_cropped_dict']['bb_height']['bb_width']
                y_hat_logit_cropped = torch.zeros((y_hat_logit.shape[0], self.hparams.eval_metrics_settings_dict['evaluate_on_EE_cropped_dict']['bb_height'], self.hparams.eval_metrics_settings_dict['evaluate_on_EE_cropped_dict']['bb_width'])).cpu()
                for idx_in_batch in range(y_hat_logit.shape[0]):
                    crop_output = ContactDatasetEpisodic.cropping(y_hat_logit.shape[-2:], self.hparams.eval_metrics_settings_dict['evaluate_on_EE_cropped_dict'], y_hat_logit.cpu().numpy(), EE_pxls[idx_in_batch].cpu().numpy())
                    y_hat_logit_cropped[idx_in_batch] = torch.tensor(crop_output[0]) #bb image is always first
                y_hat_logit_cropped = y_hat_logit_cropped.to(self.device)
                eval_dict['y_hat_logit_cropped'] = y_hat_logit_cropped

                # now crop the target as well
                y_cropped = torch.zeros((y_hat_logit.shape[0], self.hparams.eval_metrics_settings_dict['evaluate_on_EE_cropped_dict']['bb_height'], self.hparams.eval_metrics_settings_dict['evaluate_on_EE_cropped_dict']['bb_width'])).cpu()
                for idx_in_batch in range(y_hat_logit.shape[0]):
                    crop_output = ContactDatasetEpisodic.cropping(y_hat_logit.shape[-2:], self.hparams.eval_metrics_settings_dict['evaluate_on_EE_cropped_dict'], target_dict['y'].cpu().numpy(), EE_pxls[idx_in_batch].cpu().numpy())
                    y_cropped[idx_in_batch] = torch.tensor(crop_output[0])
                y_cropped = y_cropped.to(self.device)
                target_dict['y_cropped'] = y_cropped

                loss = F.binary_cross_entropy_with_logits(y_hat_logit_cropped, y_cropped)
                eval_dict['cropped_loss'] = loss
            y = target_dict['y']
            loss = F.binary_cross_entropy_with_logits(y_hat_logit, y)
            unmodified_y = target_dict['unmodified_y']
            eval_dict['unmodified_y'] = unmodified_y
            if self.hparams.cropping_dict['enable']:
                eval_dict['y_hat_logit_cropped'] = y_hat_logit
                eval_dict['cropped_loss'] = loss
                eval_dict['y_cropped'] = y
        
        if self.hparams.cropping_dict['enable'] or self.hparams.eval_metrics_settings_dict['evaluate_on_EE_cropped_dict']['enable']:
            EE_pxls = model_input_dict['EE_pose_pxl'] # B x 2
            eval_dict['EE_pose_pxl_coordinates'] = EE_pxls
            if self.hparams.eval_metrics_settings_dict['evaluate_on_EE_cropped_dict']['enable']:
                y_hat_logit = y_hat_logit_cropped
            # swap the y_hat for the padded version
            y_hat_logit = self.pad_cropped_logit(y_hat_logit, EE_pxls) # back to B x 1 x H_OG x W_OG
            if target_dict is not None:            
                loss = F.binary_cross_entropy_with_logits(y_hat_logit, unmodified_y)

        if target_dict is not None: 
            eval_dict['loss'] = loss # this will always be full image size
            
        NMS_pixel_coord_list = self.prediction_heatmap_to_NMS_pixel_coord_list(y_hat_logit, **self.hparams.nms_dict)
        eval_dict['NMS_pixel_coord_list'] = NMS_pixel_coord_list

        eval_dict['y_hat_logit'] = y_hat_logit # this will always be full image size
        eval_dict['y_hat'] = torch.sigmoid(y_hat_logit)

        return eval_dict

    def _viz_step(self, y_hat, viz_input_dict, real_dataset=False, idx_in_batch=0, NMS_pixel_coord_list=None):
        self.vid_utils.real_dataset = real_dataset
        self.viz_utils.real_dataset = real_dataset
        
        # y_hat and y are expected to be probabilities, not logits
        # should be original H x W regardless of cropping
        context_bb_x1x2y1y2_list = None
        bb_x1x2y1y2_list = None
        y_hat = y_hat[idx_in_batch, ...].squeeze(0).cpu()
        if self.hparams.cropping_dict['enable']:
            # get the bounding box corners from the EE_pose_pxls and the bounding box dimensions
            EE_pose_pxl_coordinate = viz_input_dict['EE_pose_pxl_coordinates'][idx_in_batch]
            bb_x1x2y1y2_list = self.get_bounding_box_x1x2y1y2(EE_pose_pxl_coordinate=EE_pose_pxl_coordinate) 
            if self.hparams.context_frame_dict['enable']:
                # get the bounding box corners from the EE_pose_pxls and the bounding box dimensions
                context_EE_pose_pxl_coordinate = viz_input_dict['context_EE_pose_pxl_coordinates'][idx_in_batch]
                context_bb_x1x2y1y2_list = self.get_bounding_box_x1x2y1y2(EE_pose_pxl_coordinate=context_EE_pose_pxl_coordinate)

        # get the image tensor
        # this is the original uncropped image
        depth_image = viz_input_dict['depth_image'][idx_in_batch]#.squeeze(0) # get rid of the single time/channel dim
        # get the color image path
        color_im_path = viz_input_dict['color_image_path'][idx_in_batch][0] # index into the tuple (time dim)

        if 'contact_map' in viz_input_dict.keys():
            # get the target tensor
            target_loc_map_tensor = viz_input_dict['contact_map'][idx_in_batch]
        else:
            target_loc_map_tensor = None

        if 'flow_image' in viz_input_dict.keys():
            # get the flow image
            # correctly process this if flow (2 channel) or image (3 channel)
            flow_image = viz_input_dict['flow_image'][idx_in_batch] # 2 or 3 x H x W
            if flow_image.shape[0] == 2:
                flow_image = flow_image.permute(1, 2, 0).numpy()
                flow_image = self.viz_utils.flow_to_image(flow_image, return_BGR=False)
            elif flow_image.shape[0] == 3:
                flow_image = (255.*flow_image).permute(1, 2, 0).numpy()
        else:
            flow_image = None

        if 'contact_pxls' in viz_input_dict.keys():
            # get the contact pixels
            contact_pxls_flt = viz_input_dict['contact_pxls'][idx_in_batch]
        else:
            contact_pxls_flt = None

        if 'context_depth_frame' in viz_input_dict.keys():
            context_depth_frame = viz_input_dict['context_depth_frame'][idx_in_batch]
            context_color_path = viz_input_dict['context_color_path'][idx_in_batch] # somehow the redundant time dimension automatically was removed???
        else:
            context_depth_frame = None
            context_color_path = None

        if NMS_pixel_coord_list is not None:
            NMS_pixel_coord_list = NMS_pixel_coord_list[idx_in_batch]
            # convert coordinates tensor to int and a list
            NMS_pixel_coord_list = NMS_pixel_coord_list.int().tolist()

        save_dir = None
        if self.trainer.state.stage == 'test':
            save_dir = self.trainer.loggers[self.custom_logger_index].save_dir
        # generate the visualization dict 
        heatmap_dict = self.viz_utils.draw_heatmaps(depth_image, 
                                                    pred_tensor = y_hat, target_tensor=target_loc_map_tensor,
                                                    color_path=color_im_path, flow_image=flow_image, contact_pxls_flt=contact_pxls_flt, 
                                                    bb_x1x2y1y2_list=bb_x1x2y1y2_list,
                                                    context_depth_frame=context_depth_frame, context_color_path=context_color_path, context_bb_x1x2y1y2_list=context_bb_x1x2y1y2_list,
                                                    NMS_pred_pxl_coord_list=NMS_pixel_coord_list, cross_length=self.hparams.viz_utils_dict['cross_length'],
                                                    save_dir=save_dir
                                                    )
        # generate the wrench plots
        wrench_history = viz_input_dict['wrench_history'][idx_in_batch] # T x 6
        wrench_history_times = viz_input_dict['wrench_history_times'][idx_in_batch] #T
        
        global_labels = None
        if 'global_label_history' in viz_input_dict.keys():
            global_labels = viz_input_dict['global_label_history'][idx_in_batch] # T

        grasped_object_mass = None
        if self.hparams.compensate_obj_gravity_dict['enable']:
            grasped_object_mass = viz_input_dict['grasped_object_mass'][idx_in_batch].item() # 1

        wrench_plots_dict = self.viz_utils.plot_wrench_history(wrench_history, wrench_history_times, 
                                                               figsize_hw = self.trainer.datamodule.im_resize, 
                                                               grasped_obj_mass=grasped_object_mass,
                                                               global_labels=global_labels, save_dir=save_dir) 
        if self.trainer.state.stage == 'test':
            self.viz_utils.idx +=1
        
        return {'heatmap_dict': heatmap_dict, 'wrench_plots_dict': wrench_plots_dict}    
    
    def on_test_start(self) -> None:
        self.test_prefix = 'test'

        self.test_NMS_metrics = NMS_match(**self.hparams.nms_dict)
        self.test_NMS_metrics.prefix = self.test_prefix
        self.test_NMS_metrics.to(self.device)
        self.test_NMS_metrics.persistent = False
        
        self._initialize_viz_utils()
        self._initialize_vid_utils()

        self.eval_table_created = False
        self.within_eval_dataset_episode_idx = 0

        # setup the logger params
        # hardcoding the customlogger as the second logger
        self.trainer.loggers[self.custom_logger_index].log_hyperparams(self.hparams)
        # self.trainer.loggers[self.custom_logger_index].model_name = 
        # self.trainer.loggers[self.custom_logger_index].model_seed = self.hparams
    
        if not self.hparams.nms_dict['per_sample_metrics']:
            self.test_bce_loss = torch.tensor(0.0).to(self.device)
            if self.hparams.cropping_dict['enable'] or self.hparams.eval_metrics_settings_dict['evaluate_on_EE_cropped_dict']['enable']:
                self.test_bce_loss_cropped = torch.tensor(0.0).to(self.device)
            self.episode_sample_count = 0
        

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        self.test_batch_idx = batch_idx
        
        model_input_dict = batch['model_input_dict']
        target_dict = None
        if 'target_dict' in batch:
            target_dict = batch['target_dict']

        eval_dict = self._shared_eval_step(model_input_dict, target_dict = target_dict)
        eval_dict['timestamp'] = batch['timestamp'][0]
        if 'global_contact' in batch.keys():
            eval_dict['global_contact'] = batch['global_contact'][0]
        dataset_dict = self.trainer.datamodule.test_dataset_dict_list[dataloader_idx]
        self._eval_dataset(batch, dataset_dict, eval_dict, dataloader_idx)

    def on_test_end(self) -> None:
        return super().on_test_end()
    
    def forward(self, model_input_dict):
        # B X 1 X H X W to B X H X W
        return self.model(model_input_dict['image'], 
                     pose=model_input_dict.get('pose', None), 
                     wrench=model_input_dict.get('wrench', None),
                     bb_top_left_coordinate=model_input_dict.get('pose_pxl_for_coord', None),
                     context_frame=model_input_dict.get('first_epi_image',None),
                     context_bb_top_left_coordinate=model_input_dict.get('first_epi_pose_pxl_for_coord', None)).squeeze(1) # this will remove the channel dimension of the output if it exists
    
    @staticmethod
    def prediction_heatmap_to_NMS_pixel_coord_list(prediction_heatmap, threshold, top_n_samples, window_radius, **kwargs):
        # prediction_heatmap: B X H X W
        # run NMS
        y_pred_batch = torch.sigmoid(prediction_heatmap)
        y_pred_batch = y_pred_batch.squeeze().clone()
        if y_pred_batch.shape.__len__() == 2:
            y_pred_batch = y_pred_batch.unsqueeze(0)   # 1, H, W
        
        # loop through the batch
        NMS_pixel_coordinates_list = []
        for sample_idx in range(y_pred_batch.shape[0]):
            y_pred = y_pred_batch[sample_idx]
            # if there is no ground truth, gt_features = tensor([], size=(0, 2))
            #### NMS ####
            # get the indices of elements greater than a threshold
            # get indices where prob is greater than threshold
            v_idxs, u_idxs = torch.where(y_pred > threshold)
            # if there are no pixels with scores higher than the threshold, continue to the next image
            if v_idxs.shape[0] == 0:
                # create tensor of size 0, 2
                sorted_vu_coords = torch.empty(0, 2)
            else:
                y_pred_thresholded = y_pred[v_idxs, u_idxs]

                # concatenate the values and indices so during sorting we can keep track of the indices
                y_pred_thresholded = y_pred_thresholded.unsqueeze(1)
                v_idxs = v_idxs.unsqueeze(1)
                u_idxs = u_idxs.unsqueeze(1)
                y_pred_thresholded = torch.cat((y_pred_thresholded, v_idxs, u_idxs), dim=1)
                # sort the tensor by the first column, while keeping track of the indices
                sorted_vu_coords = y_pred_thresholded[y_pred_thresholded[:, 0].sort(descending=True)[1]][:, 1:] 

                for pixel_idx in range(top_n_samples):
                    # Not enough pixels left
                    if pixel_idx+1 >= sorted_vu_coords.shape[0]:
                        break
                    # and at the last iteration, delete all pixels below it in the sorting list
                    elif pixel_idx+1 == top_n_samples:
                        sorted_vu_coords = sorted_vu_coords[:pixel_idx+1]
                        break
                    # Calculate the distance between the current pixel and the rest of the pixels
                    du = sorted_vu_coords[pixel_idx, 0] - sorted_vu_coords[pixel_idx+1:, 0]
                    dv = sorted_vu_coords[pixel_idx, 1] - sorted_vu_coords[pixel_idx+1:, 1]
                    dist = torch.sqrt(du**2 + dv**2)
                    # delete the rows of pixels that are within the window radius, keeping the current-most pixels
                    sorted_vu_coords = torch.cat((sorted_vu_coords[:pixel_idx+1], sorted_vu_coords[pixel_idx+1:][dist > window_radius]), dim=0)

                # sorted_vu_coords is tensor of size N x 2 of v, u pixel coordinates
            NMS_pixel_coordinates_list.append(sorted_vu_coords.cpu())
        return NMS_pixel_coordinates_list
    
    def pad_cropped_logit(self, cropped_prediction, EE_poses_pxl_coordinates):
        # cropped prediction should be B x bb_H x bb_W
        assert cropped_prediction.shape[-2:] == (self.hparams.cropping_dict['bb_height'], self.hparams.cropping_dict['bb_width'])
        bb_height, bb_width = self.hparams.cropping_dict['bb_height'], self.hparams.cropping_dict['bb_width']
        padded_img = torch.zeros((cropped_prediction.shape[0], self.hparams.im_resize_tuple[0], self.hparams.im_resize_tuple[1]))
        for idx in range(cropped_prediction.shape[0]):
            x1, x2, y1, y2 = self.get_bounding_box_x1x2y1y2(EE_poses_pxl_coordinates[idx])
            # handle the case where the bounding box is completely out of range
            if x2 < 0 or x1 > self.hparams.im_resize_tuple[1] or y2 < 0 or y1 > self.hparams.im_resize_tuple[0]:
                padded_img[idx] = -1e9
            else:
                # handle the case where the bounding box is partially out of range
                x1_bb = max(x1, 0) - x1
                y1_bb = max(y1, 0) - y1
                # can never be greater than bb_width or bb_height
                x2_bb = bb_width - max(0, x2-self.hparams.im_resize_tuple[1])
                y2_bb = bb_height - max(0, y2-self.hparams.im_resize_tuple[0])
                
                pad_left = max(x1, 0)
                pad_right = max(self.hparams.im_resize_tuple[1] - x2, 0)
                pad_top = max(y1, 0)
                pad_bottom = max(self.hparams.im_resize_tuple[0] - y2, 0)

                padded_img[idx] = F.pad(input=cropped_prediction[idx][y1_bb:y2_bb, x1_bb:x2_bb], pad=(pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=-1e9)

        return padded_img.to(self.device)
    
    def get_bounding_box_x1x2y1y2(self, EE_pose_pxl_coordinate):
        bb_width = self.hparams.cropping_dict['bb_width']
        bb_height = self.hparams.cropping_dict['bb_height']
        down_shift = self.hparams.cropping_dict['down_scale']

        x1 = int(EE_pose_pxl_coordinate[0]-bb_width/2)
        x2 = int(EE_pose_pxl_coordinate[0]+bb_width/2)
        y1 = int(EE_pose_pxl_coordinate[1] + down_shift -(bb_height/2))
        y2 = int(EE_pose_pxl_coordinate[1] + down_shift +(bb_height/2))
        return x1, x2, y1, y2

    # # https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#on-save-checkpoint
    # def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        # return super().on_save_checkpoint(checkpoint)
    
    # # https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#on-load-checkpoint
    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        # strip checkpoint's state_dict of all keys that don't start with "model."

        if "state_dict" in checkpoint:
            # remove all keys that don't start with "model."
            for key in list(checkpoint["state_dict"].keys()):
                if not key.startswith("model."):
                    del checkpoint["state_dict"][key]
        return super().on_load_checkpoint(checkpoint)