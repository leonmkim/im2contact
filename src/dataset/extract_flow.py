import torch
from torch.utils.data import DataLoader, random_split
import os
import numpy as np
import tqdm
from flow_dataset import FlowDataset
import pickle
import cv2 as cv
import sys
import shutil
sys.path.append('<pathto>/contact_estimation')
sys.path.append('<pathto>/contact_estimation/RAFT/core')

from RAFT.core.raft import RAFT
from RAFT.core.utils.utils import InputPadder, forward_interpolate 
from RAFT.core.utils import flow_viz
from RAFT.core.utils import frame_utils

import argparse
from tqdm import tqdm

class FlowExtractor():
    def __init__(self, dset_path, model_args, aligned_color_to_depth=False, batch_size=6, max_flow_mag=10.0) -> None:
        self.max_flow_mag = max_flow_mag
        self.dset = FlowDataset(dset_path, aligned_color_to_depth=aligned_color_to_depth)
        self.dataloader = DataLoader(self.dset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=8)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = torch.nn.DataParallel(RAFT(model_args))
        self.model.load_state_dict(torch.load(model_args.model))
        self.model.to(self.device)
        self.model.eval()
        self.image_shape = self.dset[0]['image_prev'].shape
        self.padder = InputPadder(self.image_shape)
    def extract_flow(self):
        # wrap in tqdm
        with torch.no_grad():
            for data in tqdm(self.dataloader):
                image_prev = data['image_prev'].float().to(self.device)
                image_current = data['image_current'].float().to(self.device)
                image_prev, image_current = self.padder.pad(image_prev, image_current)
                _, flow_up = self.model(image_prev, image_current, iters=10, test_mode=True)
                # unpad
                flow_up = self.padder.unpad(flow_up.detach())
                for i in range(flow_up.shape[0]):
                    flow = flow_up[i].permute(1, 2, 0).cpu().numpy()
                    # write flow to file
                    current_image_path = data['image_current_filepath'][i]
                    # make a flow directory if it doesn't exist
                    flow_dir = os.path.join(os.path.dirname(os.path.dirname(current_image_path)), 'flow')
                    if not os.path.exists(flow_dir):
                        os.makedirs(flow_dir)
                        # also get the intrinsics and extrinsics so we can copy them over using shutil
                        intrinsics_path = os.path.join(os.path.dirname(current_image_path), 'color_K.npy')
                        extrinsics_path = os.path.join(os.path.dirname(current_image_path), 'C_tf_W.npy')
                        flow_intrinsics_path = os.path.join(flow_dir, 'color_K.npy')
                        flow_extrinsics_path = os.path.join(flow_dir, 'C_tf_W.npy')
                        shutil.copy(intrinsics_path, flow_intrinsics_path)
                        shutil.copy(extrinsics_path, flow_extrinsics_path)

                    flow_path = os.path.join(flow_dir, os.path.basename(current_image_path).rstrip('.png') + '.pkl')
                    with open(flow_path, 'wb') as f:
                        pickle.dump(flow, f)
                    flow_image_dir = os.path.join(os.path.dirname(os.path.dirname(current_image_path)), 'flow_image')
                    if not os.path.exists(flow_image_dir):
                        os.makedirs(flow_image_dir)
                        # also get the intrinsics and extrinsics so we can copy them over using shutil
                        intrinsics_path = os.path.join(os.path.dirname(current_image_path), 'color_K.npy')
                        extrinsics_path = os.path.join(os.path.dirname(current_image_path), 'C_tf_W.npy')
                        flow_intrinsics_path = os.path.join(flow_image_dir, 'color_K.npy')
                        flow_extrinsics_path = os.path.join(flow_image_dir, 'C_tf_W.npy')
                        shutil.copy(intrinsics_path, flow_intrinsics_path)
                        shutil.copy(extrinsics_path, flow_extrinsics_path)

                    flow_im = flow_viz.flow_to_image(flow, convert_to_bgr=True, flow_norm=self.max_flow_mag).astype(np.uint8) # keep bgr to true for cv2 imwrite which expect bgr
                    flow_image_path = os.path.join(flow_image_dir, os.path.basename(current_image_path))
                    cv.imwrite(flow_image_path, flow_im)

if __name__ == "__main__":
    root_dset_dir = '/mnt/hdd/datasetsHDD/contact_estimation/simulated'
    dataset_name = 'teleop_10_obj_180p_2023-02-11-21-09-59'
    

    dset_path = os.path.join(root_dset_dir, dataset_name)
    # create the extractor
    model_args = argparse.Namespace()
    model_args.model = '../../RAFT/models/raft-sintel.pth'
    model_args.small = True
    model_args.mixed_precision = False
    model_args.alternate_corr = False

    extractor = FlowExtractor(dset_path, model_args)
    # process the dataset
    extractor.extract_flow()
