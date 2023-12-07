# Copyright The Lightning AI team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
CSV logger
----------

CSV logger for basic experiment logging that does not require opening ports

"""
import logging
import os
from argparse import Namespace
from typing import Any, Dict, Optional, Union

from pytorch_lightning.loggers import Logger 
from pytorch_lightning.loggers.logger import rank_zero_experiment
from pytorch_lightning.utilities import rank_zero_only
import time
import datetime

import yaml

# https://pytorch-lightning.readthedocs.io/en/1.1.8/_modules/pytorch_lightning/loggers/csv_logs.html#ExperimentWriter
# for example of using csv writing

# pandas to take dictionary and write rows
# https://stackoverflow.com/questions/42632470/how-to-add-dictionaries-to-a-dataframe-as-a-row
import pandas as pd

class CustomLogger(Logger):
    r"""
    Log to local file system in yaml and CSV format.

    Logs are saved to ``os.path.join(save_dir, name, version)``.

    Example:
        >>> from lightning.pytorch import Trainer
        >>> from lightning.pytorch.loggers import CSVLogger
        >>> logger = CSVLogger("logs", name="my_exp_name")
        >>> trainer = Trainer(logger=logger)

    Args:
        save_dir: Save directory
        name: Experiment name. Defaults to ``'lightning_logs'``.
        version: Experiment version. If version is not specified the logger inspects the save
            directory for existing versions, then automatically assigns the next available version.
        prefix: A string to put at the beginning of metric keys.
        flush_logs_every_n_steps: How often to flush logs to disk (defaults to every 100 steps).
    """

    LOGGER_JOIN_CHAR = "-"

    def __init__(
        self,
        save_root_dir: str= 'logs',
        prefix: str = "",
        flush_logs_every_n_steps: int = 100,
        model_name: str = 'model',
        model_seed: int = 0,
        model_epoch: int = 0,
        log_once_per_episode: bool = False,
        save_logs: bool = False,
    ):
        super().__init__()
        self._save_root_dir = save_root_dir
        self._prefix = prefix
        self._flush_logs_every_n_steps = flush_logs_every_n_steps
        self._logged_rows = []
        self._rank = 0
        self.save_logs = save_logs

        self.current_episode = None
        self.current_episode_dataset_name = None
        self.end_of_episode = False

        self.model_name = model_name
        self.model_seed = model_seed
        self.model_epoch = model_epoch

        self.run_name = ''

        self.log_once_per_episode = log_once_per_episode

        
    @property
    def name(self) -> Union[str,None]:
        return self.model_name
    
    @property
    def version(self) -> Union[int, str, None]:
        return self.model_epoch
    
    @property
    def save_dir(self) -> Union[str,None]:
        if self.current_episode is not None:
            return os.path.join(self._save_root_dir, 
                                        self.model_name, 'seed_' + str(self.model_seed), 'epoch_' + str(self.model_epoch), 
                                        self.current_episode_dataset_name, self.current_episode)
        else:
            return self._save_root_dir

    @rank_zero_only
    def log_hyperparams(self, params: Union[Dict[str, Any], Namespace], *args: Any, **kwargs: Any) -> None:
        # log hyperparams to the model directory under logs
        # self.save_dir = os.path.join(self._save_root_dir, self.model_name, str(self.model_seed), str(self.model_epoch))
        # update params dict with the self.run_name
        params['run_name'] = self.run_name
        # write to yaml
        self.model_level_save_dir = os.path.join(self._save_root_dir, self.model_name, 'seed_' + str(self.model_seed), 'epoch_' + str(self.model_epoch))
        if not os.path.exists(self.model_level_save_dir):
            os.makedirs(self.model_level_save_dir)
        with open(os.path.join(self.model_level_save_dir, 'hparams.yaml'), 'w') as f:
            yaml.dump(params, f)
        # return super().log_hyperparams(params,)

    # @rank_zero_only
    # def log_hyperparams(self, params: Union[Dict[str, Any], Namespace]) -> None:
    #     params = _convert_params(params)
    #     self.experiment.log_hparams(params)

    @rank_zero_only
    def log_metrics(self, metrics: Dict[str, float], step: Union[int,None] = None) -> None:
        # gets all metrics logged in the step as a big dictionary
        if self.log_once_per_episode:
            if self.end_of_episode:
                self._logged_rows.append(metrics)
        else:
            self._logged_rows.append(metrics)
        # return super().log_metrics(metrics, step)
    
    @rank_zero_only
    def save(self) -> None:
        # return super().save()
        if self.save_logs:
            if self.end_of_episode:
                # Optional. Any code necessary to save logger data goes here
                df = pd.DataFrame(self._logged_rows)
                
                # write path
                save_dir = os.path.join(self._save_root_dir, 
                                        self.model_name, 'seed_' + str(self.model_seed), 'epoch_' + str(self.model_epoch), 
                                        self.current_episode_dataset_name, self.current_episode)
                
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)


                # write to csv
                # df.to_csv(os.path.join(save_dir, self.current_episode + ".csv"))

                # write pandas to pickle
                df.to_pickle(os.path.join(save_dir, "metrics_per_sample.pkl"))
                # also write to csv
                df.to_csv(os.path.join(save_dir, "metrics_per_sample.csv"))

                # print status message
                print("Saved metrics for episode{} to {}".format(self.current_episode, save_dir))
                # reset
                self._logged_rows.clear()

                # reset end of episode
                self.end_of_episode = False
        
    @rank_zero_only
    def finalize(self, status: str) -> None:
        return super().finalize(status)

    # @property
    # @rank_zero_experiment
    # def experiment(self) -> _FabricExperimentWriter:
    #     r"""

    #     Actual _ExperimentWriter object. To use _ExperimentWriter features in your
    #     :class:`~lightning.pytorch.core.module.LightningModule` do the following.

    #     Example::

    #         self.logger.experiment.some_experiment_writer_function()

    #     """
    #     if self._experiment is not None:
    #         return self._experiment

    #     self._fs.makedirs(self.root_dir, exist_ok=True)
    #     self._experiment = ExperimentWriter(log_dir=self.log_dir)
    #     return self._experiment
