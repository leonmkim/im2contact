import torch
from torch.utils.data import DataLoader, random_split
import os
import numpy as np
import tqdm
import pickle
import sys
sys.path.append('../')
from src.dataset.contact_dataset_episodic import ContactDatasetEpisodic

class ContactExtractor():
    def __init__(self, dset_path, save_mask=True, save_avg_num_contacts=True, color_aligned_to_depth=False, is_real_dataset=False) -> None:
        if color_aligned_to_depth: # means this is l515 and in sim
            self.dset = ContactDatasetEpisodic(dset_path, l515=True, is_real_dataset=is_real_dataset)
        else:
            self.dset = ContactDatasetEpisodic(dset_path)
        self.save_mask = save_mask
        self.save_avg_num_contacts = save_avg_num_contacts
    def process_dataset(self):
        # get list of episodes
        episode_data_dir_path_list = self.dset.episode_data_dir_path_list
        # enumerate through list elements and get the idx
        for episode_idx, episode_data_dir_path in enumerate(tqdm.tqdm(episode_data_dir_path_list)):
            self.extract_contact_mask_and_avg_num_contacts_per_episode(episode_idx, episode_data_dir_path)
    def extract_contact_per_episode(self, episode_idx, episode_data_dir_path):
        total_contact_idxs = []
        # load the episode
        episode_idxs = self.dset.get_indices_for_episode_list([episode_idx])
        # create subset
        episode_dset = torch.utils.data.Subset(self.dset, episode_idxs)
        # create a dataloader
        episode_dataloader = DataLoader(episode_dset, batch_size=16, shuffle=False, drop_last=False, num_workers=8)
        # iterate over the episode
        for idx, data in enumerate(episode_dataloader):
            num_contacts = data['num_contacts']
            idxs = data['within_episode_idx']
            contact_idxs = idxs[num_contacts>0]
            total_contact_idxs.extend(contact_idxs.tolist())
        # save the contact indices
        episde_data_dir = os.path.join(episode_data_dir_path, 'contact_idxs.npy') 
        np.save(episde_data_dir, np.array(total_contact_idxs))
    
    def extract_contact_mask_and_avg_num_contacts_per_episode(self, episode_idx, episode_data_dir_path):
        contact_idxs_mask = []
        # load the episode
        episode_idxs = self.dset.get_indices_for_episode_list([episode_idx])
        # create subset
        episode_dset = torch.utils.data.Subset(self.dset, episode_idxs)
        num_episode_samples = len(episode_dset)
        total_num_contacts = 0
        # create a dataloader
        episode_dataloader = DataLoader(episode_dset, batch_size=16, shuffle=False, drop_last=False, num_workers=8)
        # iterate over the episode using tqdm
        for idx, data in enumerate(tqdm.tqdm(episode_dataloader)):
            num_contacts = data['num_contacts']
            total_num_contacts += num_contacts.sum().item()
            idxs = data['within_episode_idx']
            contact_mask = num_contacts>0
            contact_idxs_mask.extend(contact_mask.tolist())
        if self.save_mask:
            # save the contact indices
            episde_data_dir = os.path.join(episode_data_dir_path, 'contact_idxs_mask.npy') 
            np.save(episde_data_dir, np.array(contact_idxs_mask))
        if self.save_avg_num_contacts:
            # save the avg num contacts with pickle 
            avg_num_contacts = total_num_contacts/num_episode_samples
            episde_data_dir = os.path.join(episode_data_dir_path, 'avg_num_contacts.pickle') 
            with open(episde_data_dir, 'wb') as f:
                pickle.dump(avg_num_contacts, f)
    
if __name__ == "__main__":
    # root_dset_dir = os.path.expanduser('~/datasets/contact_estimation/simulated') #cluster
    root_dset_dir = os.path.expanduser('/mnt/hdd/datasetsHDD/contact_estimation/simulated') 
    dset_dir = 'teleop_10_obj_180p_2023-02-11-21-09-59'

    dset_path = os.path.join(root_dset_dir, dset_dir)
    # create the extractor
    save_mask = True
    save_avg_num_contacts = True
    extractor = ContactExtractor(dset_path, save_mask=save_mask, save_avg_num_contacts=save_avg_num_contacts)
    # process the dataset
    extractor.process_dataset()
 