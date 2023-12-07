import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.cli import ArgsType, LightningArgumentParser, LightningCLI
import wandb
from typing import Any, Dict, List, Optional, Union
import sys
from argparse import Namespace

from src.dataset.contact_datamodule import ContactDataModule
from src.model.local_contact_model import LocalContactModel

import torch

import os
from pathlib import Path

from src.utils.custom_csv_logger import CustomLogger
from dataclasses import asdict

from jsonargparse import ArgumentParser, ActionConfigFile
class LocalContactTestingCLI(LightningCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser):
        parser.default_config_files = ['./config/testing.yaml']
        # add what keys to overwrite in the config file
        parser.add_argument('--data_test_keys_to_overwrite_checkpoint', type=List[str], default=['test_dataset_dict', 'proprio_history_plotting_dict'])
        parser.add_argument('--model_test_keys_to_overwrite_checkpoint', type=List[str], default=['nms_dict', 'viz_utils_dict', 'vid_utils_dict'])
        parser.add_argument('--checkpoint_reference', type=str)
        # parser.add_argument('test.data.test_dataset_dict_list', type=List[TestDatasetDict])
        # parser.add_argument('test.data.proprio_history_plotting_dict', type=ProprioceptionHistoryDict)
        
        # link the checkpoint_reference to the data.checkpoint_reference
        # parser.link_arguments('checkpoint_reference', ['test.data.checkpoint_reference'])
        # pass

    def before_instantiate_classes(self):
        pass

    def instantiate_classes(self) -> None:
        self.config_init = self.parser.instantiate_classes(self.config)
        ckpt_path = self._download_checkpoint_from_wandb_and_get_path(self.config.test.checkpoint_reference)
        self._load_model_from_checkpoint_path(ckpt_path)
        # self.datamodule = self._get(self.config_init, "data")
        self._load_datamodule_from_checkpoint_path(ckpt_path)

        self._add_configure_optimizers_method_to_model(self.subcommand)
        self.trainer = self.instantiate_trainer()

        # load relevant checkpoint params to the trainers custom logger (like model epoch and seed)
        self._get_custom_logger_params_from_checkpoint_path(ckpt_path)

    def _get_custom_logger_params_from_checkpoint_path(self, ckpt_path):
        # must do this because the epoch and other run params are not loaded by model
        ckpt = torch.load(ckpt_path)
        epoch = ckpt['epoch']
        seed = ckpt['hyper_parameters'].get('global_seed', 0)
        model_name = self._get_model_name_from_checkpoint(ckpt)
        # strip the posixpath to get the run_name of the model
        run_name = Path(ckpt_path).parent.name

        for i, logger in enumerate(self.trainer.loggers):
            if isinstance(logger, CustomLogger):
                logger.model_epoch = epoch
                logger.model_seed = seed
                logger.model_name = model_name
                logger.run_name = run_name
                self.model.custom_logger_index = i

    def _get_model_name_from_checkpoint(self, checkpoint):
        # compose the model name from both model hparams and datamodule hparams
        datamodule_params = checkpoint['datamodule_hyper_parameters']
        model_params = checkpoint['hyper_parameters']
        model_name = model_params['model_type']
        if model_params['cropping_dict']['enable']:
            model_name += '_crop_'
            model_name += 'h' + str(model_params['cropping_dict']['bb_height'])
            model_name += 'w' + str(model_params['cropping_dict']['bb_width'])
            model_name += 'd' + str(model_params['cropping_dict']['down_scale'])
        if model_params['context_frame_dict']['enable']:
            model_name += '_ctxt'
        if model_params['optical_flow_dict']['enable']:
            model_name += '_flow'
            if model_params['optical_flow_dict']['use_image']:
                model_name += 'img'
        any_proprio = False
        for key, value in model_params['proprioception_input_dict'].items():
            if isinstance(value, bool) and value:
                any_proprio = True
                model_name += '_' + key
                if key == 'external_wrench':
                    if model_params['compensate_obj_gravity_dict']['enable']:
                        model_name += 'compgrav'
                        model_name += 'EEyCoM' + str(model_params['compensate_obj_gravity_dict']['EE_pos_y_objCoM'])
            if any_proprio:
                # model_name += 'poselayers' + str(model_params['pose_hidden_list'])
                # model_name += 'wrenchlayers' + str(model_params['wrench_hidden_list'])
                # if model_params['proprioception_input_dict']['in_cam_frame_dict']['enable']:
                    # model_name += 'incam'
                if model_params['proprioception_input_dict']['history_dict']['enable']:
                    model_name += 'hist'
                    model_name += 'dt' + str(model_params['proprioception_input_dict']['history_dict']['time_window'])
                    model_name += 'hz' + str(model_params['proprioception_input_dict']['history_dict']['sample_freq'])
                if model_params['add_noise_dict']['enable']:
                    model_name += 'noise' + model_params['add_noise_dict']['noise_type']
        if model_params['blur_contact_map_dict']['enable']:
            model_name += '_blur'
        model_name += '_numepi' + str(datamodule_params['fit_dataset_dict']['train_num_episodes'])
        model_name += '_lr' + str(model_params['lr'])
        model_name += '_wd' + str(model_params['weight_decay'])
        return model_name
    
    def _download_checkpoint_from_wandb_and_get_path(self, checkpoint_reference):
        # https://lightning.ai/docs/pytorch/stable/extensions/generated/lightning.pytorch.loggers.WandbLogger.html#lightning.pytorch.loggers.WandbLogger
        # reference can be retrieved in artifacts panel
        # "VERSION" can be a version (ex: "v2") or an alias ("latest or "best")
        # checkpoint_reference = "contact-estimation/local_contact/model-62907_2:latest"

        # download checkpoint locally (if not already cached)
        run = wandb.init(project="test_local_contact")
        artifact = run.use_artifact(checkpoint_reference, type="model")
        artifact_dir = artifact.download()

        # load checkpoint
        ckpt_path = Path(artifact_dir) / "model.ckpt"    
        return ckpt_path

    def _load_model_from_checkpoint_path(self, ckpt_path):
        model_overwrite_dict = self._get_data_overwrite_dict(self.config.test.model_test_keys_to_overwrite_checkpoint, self.config_init.test.model.hparams)
        self.model = LocalContactModel.load_from_checkpoint(ckpt_path, **model_overwrite_dict)
        # put model on cuda
        self.model.cuda()
        return ckpt_path
    
    def _load_datamodule_from_checkpoint_path(self, checkpoint_path):
        # data_overwrite_dict = dict((k, vars(self.config.test.data)[k]) for k in self.config.test.data_test_keys_to_overwrite_checkpoint if k in vars(self.config.test.data))
        # kwargs overwrites the hyperparameters in the checkpoint
        data_overwrite_dict = self._get_data_overwrite_dict(self.config.test.data_test_keys_to_overwrite_checkpoint, self.config_init.test.data.hparams)
        self.datamodule = ContactDataModule.load_from_checkpoint(checkpoint_path, **data_overwrite_dict)
    
    def _get_data_overwrite_dict(self, keys_to_overwrite, hparams):
        overwrite_dict = {}
        for key in keys_to_overwrite:
            # use the class path and init args in each dict to instantiate the class
                # use the hparams in the initialized datamodule in self.config_init
                    overwrite_dict[key] = hparams[key]
        return overwrite_dict
    
    def _overwrite_config_with_checkpoint(self, config, checkpoint, keys_to_overwrite):
        for key in keys_to_overwrite:
            self.config.test.data[key] = checkpoint[key]
        return config
    
    def _load_model_from_checkpoint(self, checkpoint):
        test_model_hparams = vars(self.config.test.model)
        model_keys_to_overwrite = self.config.test.model_test_keys_to_overwrite_checkpoint
        test_model_hparams = dict((k, test_model_hparams[k]) for k in model_keys_to_overwrite if k in test_model_hparams)
        checkpoint_model_hparams = checkpoint['hyper_parameters']
        model_keys = list(vars(self.config.test.model).keys())
        checkpoint_model_hparams = dict((k, checkpoint_model_hparams[k]) for k in model_keys if k in checkpoint_model_hparams)
        checkpoint_model_hparams.update(test_model_hparams)
        self.model = LocalContactModel(**checkpoint_model_hparams)

    def _load_datamodule_from_checkpoint(self, checkpoint):
        # merge the datamodule hyperparameters with the config_init test hyperparameters
        test_data_hparams = vars(self.config.test.data)
        # only select the keys we want to update
        data_keys_to_overwrite = self.config.test.data_test_keys_to_overwrite_checkpoint
        test_data_hparams = dict((k, test_data_hparams[k]) for k in data_keys_to_overwrite if k in test_data_hparams)
        # select only a subset of the hyperparameters from the checkpoint to update the datamodule hyperparameters
        # checkpoint_data_hparams = checkpoint['datamodule_hyper_parameters']
        # the kwargs in the load_from_checkpoint method overwrites any keys in the checkpoint
        # get the kwargs we want to overwrite from self.config.test.data from the self.config.test.data_test_keys_to_overwrite_checkpoint
        
        # data_overwrite_dict = dict((k, vars(self.config.test.data)[k]) for k in self.config.test.data_test_keys_to_overwrite_checkpoint if k in vars(self.config.test.data))
        data_overwrite_dict = self._get_data_overwrite_dict()
        
        # need to pass checkpoint_path not checkpoint...
        # self.datamodule = ContactDataModule.load_from_checkpoint(checkpoint['datamodule_hyper_parameters'], **data_overwrite_dict)

        # datamodule_keys = list(vars(self.config.test.data).keys())
        # checkpoint_data_hparams = dict((k, checkpoint_data_hparams[k]) for k in datamodule_keys if k in checkpoint_data_hparams)

        # datamodule_hyperparameters.update(checkpoint_data_hparams)
        # checkpoint_data_hparams.update(test_data_hparams)
        # self.datamodule = self._get(self.config_init, "data")
        # self.datamodule = ContactDataModule(**checkpoint_data_hparams)

    def _load_datamodule_from_state_dict(self, state_dict):
        self.datamodule = ContactDataModule.load_state_dict(state_dict)

    def _old_instantiate_classes(self) -> None:
        # self.model = self._get(self.config_init, "model")
        # use the hyperparameters from the checkpoint instead to instantiate the model
        # consider saving the model hyperparameters as yaml config in the checkpoint folder
        # load the checkpoint from the ckpt_path
        checkpoint = torch.load(self.config.test.ckpt_path)
        
        test_model_hparams = vars(self.config.test.model)
        model_keys_to_overwrite = self.config.test.model_test_keys_to_overwrite_checkpoint
        test_model_hparams = dict((k, test_model_hparams[k]) for k in model_keys_to_overwrite if k in test_model_hparams)
        checkpoint_model_hparams = checkpoint['hyper_parameters']
        model_keys = list(vars(self.config.test.model).keys())
        checkpoint_model_hparams = dict((k, checkpoint_model_hparams[k]) for k in model_keys if k in checkpoint_model_hparams)
        checkpoint_model_hparams.update(test_model_hparams)
        self.model = LocalContactModel(**checkpoint_model_hparams)
        
        # merge the datamodule hyperparameters with the config_init test hyperparameters
        test_data_hparams = vars(self.config.test.data)
        # only select the keys we want to update
        data_keys_to_overwrite = self.config.test.data_test_keys_to_overwrite_checkpoint
        test_data_hparams = dict((k, test_data_hparams[k]) for k in data_keys_to_overwrite if k in test_data_hparams)
        # select only a subset of the hyperparameters from the checkpoint to update the datamodule hyperparameters
        checkpoint_data_hparams = checkpoint['datamodule_hyper_parameters']
        # datamodule_keys = list(vars(self.config.test.data).keys())
        # checkpoint_data_hparams = dict((k, checkpoint_data_hparams[k]) for k in datamodule_keys if k in checkpoint_data_hparams)
        # datamodule_hyperparameters.update(checkpoint_data_hparams)
        checkpoint_data_hparams.update(test_data_hparams)
        # self.datamodule = self._get(self.config_init, "data")
        self.datamodule = ContactDataModule(**checkpoint_data_hparams)

        self._add_configure_optimizers_method_to_model(self.subcommand)
        self.trainer = self.instantiate_trainer()
    
def cli_main(args: list = None):
    
    cli = LocalContactTestingCLI(LocalContactModel, ContactDataModule, 
                                 args=args, save_config_overwrite=True)

def try_load_best_model():
    # https://lightning.ai/docs/pytorch/stable/extensions/generated/lightning.pytorch.loggers.WandbLogger.html#lightning.pytorch.loggers.WandbLogger
    checkpoint_reference = "contact-estimation/local_contact/model-62907_2:latest"

    # download checkpoint locally (if not already cached)
    run = wandb.init(project="local_contact")
    artifact = run.use_artifact(checkpoint_reference, type="model")
    artifact_dir = artifact.download()

    # load checkpoint
    model = LocalContactModel.load_from_checkpoint(Path(artifact_dir) / "model.ckpt")
    model.eval()

    return model
    
if __name__ == '__main__':  
    cli_main()
