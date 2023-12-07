from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple, Union
import os

@dataclass
class OpticalFlowDict:
    enable: bool = False
    use_image: bool = False
    normalize: bool = True
    max_flow_norm: float = 10.0
    fill_holes: bool = True

    @classmethod
    def from_dict(cls, d):
        return cls(**d)
    
@dataclass
class CroppingDict:
    enable: bool = False
    bb_height: int = 90
    bb_width: int = 110
    down_scale: int = 22

    def __post_init__(self):
        assert self.bb_height % 2 == 0 and self.bb_width % 2 == 0, "bb_height and bb_width must be even numbers"

    @classmethod
    def from_dict(cls, d):
        return cls(**d)

@dataclass
class ContextFrameDict:
    enable: bool = False

    @classmethod
    def from_dict(cls, d):
        return cls(**d)

@dataclass
class BlurContactMapDict:
    enable: bool = False
    kernel_size: int = 5
    sigma: float = 0

    @classmethod
    def from_dict(cls, d):
        return cls(**d)
@dataclass
class ProprioFilterDict:
    enable: bool = False
    filter_name: str = 'filtered_o_3_f_5'

    @classmethod
    def from_dict(cls, d):
        return cls(**d)
    
@dataclass
class FitDatasetDict:
    dataset_root_dir_path_local: str = ''
    dataset_root_dir_path_remote: str = ''
    dataset_name_list: List[str] = field(default_factory=lambda: [])
    dataset_dir_path_list: List[str] = field(default_factory=lambda: [])
    real_dataset: bool = False
    train_num_workers: int = 8
    val_num_workers: int = 8
    train_num_episodes: int = 5
    val_num_episodes: int = 5
    train_batch_size: int = 8
    val_batch_size: int = 8
    l515: bool = True
    is_annotated_local: bool = False
    is_annotated_global: bool = False
    downsampled_images: bool = False

    def __post_init__(self):
        hostname = os.uname()[1]
        running_locally = hostname == 'MAGI-SYSTEM'
        if self.dataset_dir_path_list == []:
            assert self.dataset_name_list != [], "dataset_name_list must be provided"
            assert self.dataset_root_dir_path_local != '' or self.dataset_root_dir_path_remote != '', "dataset_root_dir_path_local or dataset_root_dir_path_remote must be provided"
            if running_locally:
                dataset_root_path = self.dataset_root_dir_path_local
            else:
                dataset_root_path = self.dataset_root_dir_path_remote
            dataset_dir_path_list = []
            for dataset_dir_path in self.dataset_name_list:
                assert os.path.isdir(os.path.join(dataset_root_path, dataset_dir_path)), f"dataset_dir_path {os.path.join(dataset_root_path, dataset_dir_path)} does not exist"
                dataset_dir_path_list.append(os.path.join(dataset_root_path, dataset_dir_path))
            self.dataset_dir_path_list = dataset_dir_path_list
        else:
            assert self.dataset_dir_path_list != [], "dataset_dir_path_list must be provided"
        if self.real_dataset:
            self.is_annotated_local = True
            self.is_annotated_global = True

    @classmethod
    def from_dict(cls, d):
        return cls(**d)

# a dataset with list of episodes
@dataclass
class DatasetEpisodeCollectionDict:
    dataset_root_dir_path_local: str = ''
    dataset_root_dir_path_remote: str = ''
    dataset_dir_name: str = ''
    dataset_dir_path: str = ''
    episode_idx_list: Union[List[int], str, int] = -1

    def __post_init__(self):
        hostname = os.uname()[1]
        running_locally = hostname == 'MAGI-SYSTEM'
        if running_locally:
            dataset_root_path = self.dataset_root_dir_path_local
        else:
            dataset_root_path = self.dataset_root_dir_path_remote
        assert os.path.isdir(os.path.join(dataset_root_path, self.dataset_dir_name)), f"dataset_dir_path {os.path.join(dataset_root_path, self.dataset_dir_name)} does not exist"
        dataset_dir_path = os.path.join(dataset_root_path, self.dataset_dir_name)
        self.dataset_dir_path = dataset_dir_path

        # if episode_idx_list is a str, we can interpret it as a tuple that specifies a range
        if isinstance(self.episode_idx_list, str):
            assert self.episode_idx_list[0] == '(' and self.episode_idx_list[-1] == ')', "episode_idx_list must be a list or a string that specifies a range"
            self.episode_idx_list = self.episode_idx_list[1:-1].split(',')
            assert len(self.episode_idx_list) == 2, "episode_idx_list must be a list or a string that specifies a range"
            self.episode_idx_list = list(range(int(self.episode_idx_list[0]), int(self.episode_idx_list[1])+1))

    @classmethod
    def from_dict(cls, d):
        return cls(**d)
    

# this defines one dataloader
@dataclass
class EvaluationDatasetSettingsDict:
    dataset_name: str = ''
    num_workers: int = 8
    batch_size: int = 1
    evaluate_metrics: bool = False
    log_per_episode_metrics: bool = False
    log_video: bool = False
    viz_global_contact: bool = False
    real_dataset: bool = False
    is_annotated_local: bool = False
    is_annotated_global: bool = False
    l515: bool = True
    proprio_filter_dict: ProprioFilterDict = ProprioFilterDict()
    downsampled_images: bool = False

    def __post_init__(self):
        if self.proprio_filter_dict.enable: 
            assert self.real_dataset, "filter_ft only is for real datasets"

        if self.evaluate_metrics:
            if self.real_dataset:
                assert self.is_annotated_local, "is_annotated_local must be True on real datasets for evaluate_metrics"
                assert self.is_annotated_global, "is_annotated_global must be True on real datasets for evaluate_metrics"

    @classmethod
    def from_dict(cls, d):
        return cls(**d)
    
@dataclass
class CuratedValidationDatasetDict:
    enable: bool = False
    epoch_interval: int = 5

    settings_dict: EvaluationDatasetSettingsDict = EvaluationDatasetSettingsDict()
    dataset_episode_collection_dict_list: List[DatasetEpisodeCollectionDict] = field(default_factory=lambda: [])

    @classmethod
    def from_dict(cls, d):
        return cls(**d)
@dataclass
class TestDatasetDict:
    settings_dict: EvaluationDatasetSettingsDict = EvaluationDatasetSettingsDict()
    dataset_episode_collection_dict_list: List[DatasetEpisodeCollectionDict] = field(default_factory=lambda: [])

    @classmethod
    def from_dict(cls, d):
        return cls(**d)

@dataclass
class DatasetCollection:
    dataset_dict_list: List[CuratedValidationDatasetDict] = field(default_factory=lambda: [])

    @classmethod
    def from_dict(cls, d):
        return cls(**d)
@dataclass
class EvaluationDatasetDict:
    settings_dict: EvaluationDatasetSettingsDict = EvaluationDatasetSettingsDict()
    episode_subset_dataset_dict_list: List[DatasetEpisodeCollectionDict] = field(default_factory=lambda: [])

    @classmethod
    def from_dict(cls, d):
        return cls(**d)
    
@dataclass 
class EvaluationMetricsSettingsDict:
    evaluate_on_EE_cropped_dict: CroppingDict = CroppingDict()

    @classmethod
    def from_dict(cls, d):
        return cls(**d)

@dataclass
class NMSDict:
    window_radius: float = 5.0
    top_n_samples: int = 15
    threshold : float = 0.01  # Keeping less points is better # Need to turn to tensor
    lambda_FP: float = 1.0
    lambda_FN: float = 1.0
    max_num_contact: int = 15
    TP_pixel_radius: int = 15
    gaussian_kernel_size: int = 5
    per_sample_metrics: bool = False
    log_aggregate_metrics: bool = False

    @classmethod
    def from_dict(cls, d):
        return cls(**d)
    
@dataclass
class InCamFrameDict:
    enable: bool = False
    # add depth camera extrinsics
    # depth rotation matrix
    # store in column major order
    C_R_W: List[List[float]] = field(default_factory=lambda: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    C_t_W: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])

    @classmethod
    def from_dict(cls, d):
        return cls(**d)

# @dataclass
# class ProprioHistoryPlottingDict:
#     enable: bool = False
#     time_window: float = 1.0 
#     sample_freq: int = 100

#     @classmethod
#     def from_dict(cls, d):
#         return cls(**d)

@dataclass
class ProprioceptionHistoryDict:
    enable: bool = False
    time_window: float = 1.0
    sample_freq: int = 100

    @classmethod
    def from_dict(cls, d):
        return cls(**d)
@dataclass
class ProprioceptionInputDict:
    external_wrench: bool = False
    EE_pose: bool = False
    EE_velocity: bool = False
    joint_positions: bool = False
    joint_velocities: bool = False
    desired_joint_torques: bool = False
    measured_joint_torques: bool = False
    external_joint_torques: bool = False
    rotation_representation: str = 'quaternion' #must be quaternion, euler, or rotation_matrix
    in_cam_frame_dict: InCamFrameDict = field(default_factory=lambda: InCamFrameDict())
    history_dict: ProprioceptionHistoryDict = field(default_factory=lambda: ProprioceptionHistoryDict())

    def __post_init__(self):
        assert self.rotation_representation in ['quaternion', 'euler', 'rotation_matrix'], "rotation_representation must be quaternion, euler, or rotation_matrix"
        if not (self.external_wrench or self.EE_pose or self.desired_joint_torques or self.joint_velocities or self.measured_joint_torques):
            self.history_dict.enable == False
            self.in_cam_frame_dict.enable == False
    @classmethod
    def from_dict(cls, d):
        return cls(**d)
@dataclass
# add an initialization here that takes in camera extrinsics and computes the in_cam_frame normalization!
class NormalizeDict:
    enable: bool = False
    O_position_EE_max: List[float] = field(default_factory=lambda: [0.8, 0.6, 0.5])
    O_position_EE_min: List[float] = field(default_factory=lambda: [0.0, -0.6, -0.2])
    O_F_ext_EE_max: List[float] = field(default_factory=lambda: [20, 20, 20, 5, 5, 5])
    O_F_ext_EE_min: List[float] = field(default_factory=lambda: [-20, -20, -20, -5, -5, -5])
    desired_torques_max: List[float] = field(default_factory=lambda: [30, 30, 30, 30, 20, 20, 10])
    desired_torques_min: List[float] = field(default_factory=lambda: [-30, -30, -30, -30, -20, -20, -10])
    joint_velocities_max: List[float] = field(default_factory=lambda: [1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5])
    joint_velocities_min: List[float] = field(default_factory=lambda: [-1.5, -1.5, -1.5, -1.5, -1.5, -1.5, -1.5])
    joint_positions_max: List[float] = field(default_factory=lambda: [2.9, 1.8, 2.9, -0.05, 2.9, 3.76, 2.9])
    joint_positions_min: List[float] = field(default_factory=lambda: [-2.9, -1.8, -2.9, -3.1, -2.9, -0.02, -2.9])
    external_torques_max: List[float] = field(default_factory=lambda: [30, 30, 30, 30, 20, 20, 10])
    external_torques_min: List[float] = field(default_factory=lambda: [-30, -30, -30, -30, -20, -20, -10])
    measured_torques_max: List[float] = field(default_factory=lambda: [87, 87, 87, 87, 12, 12, 12])
    measured_torques_min: List[float] = field(default_factory=lambda: [-87, -87, -87, -87, -12, -12, -12])

    @classmethod
    def from_dict(cls, d):
        return cls(**d)

@dataclass
# Add noise to FT signal to avoid overfitting on global contact prediction
class AddNoiseDict:
    enable: bool = False
    noise_type: str = "gaussian_per_sample"
    FT_dict: Dict[str, Any] = field(default_factory=lambda: {"mean_gaussian": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                                "std_gaussian": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                                                                "freq_sin_mean": [25, 25, 25, 25, 25, 25],
                                                                "freq_sin_std": [5, 5, 5, 5, 5, 5],
                                                                "amp_sin_mean": [1, 1, 1, .3, .3, .3],
                                                                "amp_sin_std": [0.5, 0.5, 0.5, .1, .1, .1],
                                                                "phase_sin_mean": [0, 0, 0, 0, 0, 0],
                                                                "phase_sin_std": [1.5, 1.5, 1.5, 1.5, 1.5, 1.5]})
    desired_joint_torque_dict: Dict[str, Any] = field(default_factory=lambda: {"mean_gaussian": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                                                "std_gaussian": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                                                                                "freq_sin_mean": [25, 25, 25, 25, 25, 25, 25],
                                                                                "freq_sin_std": [5, 5, 5, 5, 5, 5],
                                                                                "amp_sin_mean": [.2, .2, .2, .2, .2, .2, .2],
                                                                                "amp_sin_std": [.1, .1, .1, .1, .1, .1, .1],
                                                                                "phase_sin_mean": [0, 0, 0, 0, 0, 0, 0],
                                                                                "phase_sin_std": [1.5, 1.5, 1.5, 1.5, 1.5, 1.5]})
    joint_velo_dict: Dict[str, Any] = field(default_factory=lambda: {"mean_gaussian": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                                    "std_gaussian": [0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06],
                                                                    "freq_sin_mean": [25, 25, 25, 25, 25, 25, 25],
                                                                    "freq_sin_std": [5, 5, 5, 5, 5, 5],
                                                                    "amp_sin_mean": [.1, .1, .1, .1, .1, .1, .1],
                                                                    "amp_sin_std": [.05, .05, .05, .05, .05, .05, .05],
                                                                    "phase_sin_mean": [0, 0, 0, 0, 0, 0, 0],
                                                                    "phase_sin_std": [1.5, 1.5, 1.5, 1.5, 1.5, 1.5]}) 
    external_joint_torque_dict: Dict[str, Any] = field(default_factory=lambda: {"mean_gaussian": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                                                "std_gaussian": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                                                                                "freq_sin_mean": [25, 25, 25, 25, 25, 25],
                                                                                "freq_sin_std": [5, 5, 5, 5, 5, 5],
                                                                                "amp_sin_mean": [.2, .2, .2, .2, .2, .2],
                                                                                "amp_sin_std": [.1, .1, .1, .1, .1, .1],
                                                                                "phase_sin_mean": [0, 0, 0, 0, 0, 0],
                                                                                "phase_sin_std": [1.5, 1.5, 1.5, 1.5, 1.5, 1.5]})
    measured_joint_torque_dict: Dict[str, Any] = field(default_factory=lambda: {"mean_gaussian": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                                                "std_gaussian": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                                                                                "freq_sin_mean": [25, 25, 25, 25, 25, 25],
                                                                                "freq_sin_std": [5, 5, 5, 5, 5, 5],
                                                                                "amp_sin_mean": [.2, .2, .2, .2, .2, .2],
                                                                                "amp_sin_std": [.1, .1, .1, .1, .1, .1],
                                                                                "phase_sin_mean": [0, 0, 0, 0, 0, 0],
                                                                                "phase_sin_std": [1.5, 1.5, 1.5, 1.5, 1.5, 1.5]})
    @classmethod
    def from_dict(cls, d):
        return cls(**d)

@dataclass
class CompensateObjGravityDict:
    enable: bool = False
    # EE frame is with y axis pointing down, z axis pointing forward
    EE_pos_x_objCoM: float = 0.0
    EE_pos_y_objCoM: float = 0.15
    EE_pos_z_objCoM: float = 0.0
    start_idx: int = 10
    num_idxs: int = 250

    @classmethod
    def from_dict(cls, d):
        return cls(**d)
    
def test_curated_validation_dataset_dict():
    # test initializing the dataclass from a dict
    curated_validation_dataset_dict = {
        'enable': True,
        'epoch_interval': 1,
        'settings_dict': {
                'dataset_name': 'real',
                'num_workers': 8,
                'batch_size': 1,
                'log_video': True,
                'real_dataset': True
        },
        'dataset_episode_collection_dict_list': [
            {
                'dataset_dir_path': '/mnt/hdd/datasetsHDD/contact_estimation/real/teleop_real_480x640_L515_2023-05-09-17-05-40',
                'episode_idx_list': [1,2,3]
            }
        ]
    }

    curated_validation_dataset_dict = CuratedValidationDatasetDict.from_dict(curated_validation_dataset_dict)
    print(curated_validation_dataset_dict)

def test_evaluation_dataset_settings_dict():
    # test proprioception input dict
    # proprioception_input_dict = ProprioceptionInputDict()
    # print(proprioception_input_dict)

    # fit_dataset_dict = FitDatasetDict()
    # print(fit_dataset_dict)

    # try to instantiate an EpisodeCollectionDatasetDict
    ep_dset_0 = DatasetEpisodeCollectionDict(dataset_dir_name='/mnt/hdd/datasetsHDD/contact_estimation/real/teleop_real_480x640_L515_2023-05-17-23-25-02', episode_idx_list=[0,1,2])
    ep_dset_1 = DatasetEpisodeCollectionDict(dataset_dir_name='/mnt/hdd/datasetsHDD/contact_estimation/real/teleop_real_480x640_L515_2023-05-17-23-25-02', episode_idx_list=[5])
    
    # episode_collection_dataset_dict = EpisodeCollectionDatasetDict(episode_subset_dataset_dict_list=[ep_dset_0, ep_dset_1])

    eval_dataset = EvaluationDatasetSettingsDict(episode_subset_dataset_dict_list=[ep_dset_0, ep_dset_1])

    print(eval_dataset)

if __name__ == '__main__':
    test_curated_validation_dataset_dict()