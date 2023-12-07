# import argparse
# import logging
# from pickletools import uint8
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import numpy as np



class ConfusionMetrics():
    def __init__(self, device, pos_thresh = 0.5, num_pixels_per_sample=57600):
        self.pos_thresh = pos_thresh
        self.device = device
        self.num_pixels_per_sample = num_pixels_per_sample

        self.TP_global_contact_total = torch.tensor(0.).to(device) 
        self.FP_global_contact_total = torch.tensor(0.).to(device) 
        self.FN_global_contact_total = torch.tensor(0.).to(device) 
        self.TN_global_contact_total = torch.tensor(0.).to(device) 

        self.TP_local_contact_total = torch.tensor(0.).to(device) 
        self.FP_local_contact_total = torch.tensor(0.).to(device) 
        self.FN_local_contact_total = torch.tensor(0.).to(device) 
        self.TN_local_contact_total = torch.tensor(0.).to(device) 

        self.soft_TP_local_contact_total = torch.tensor(0.).to(device) 
        self.soft_FP_local_contact_total = torch.tensor(0.).to(device) 
        self.soft_FN_local_contact_total = torch.tensor(0.).to(device) 
        self.soft_TN_local_contact_total = torch.tensor(0.).to(device) 

        self.num_samples = 0

    def add_loc_confusion_matrix(self, loc_pred, loc_target = None, global_contact_prob=None): 
        # B x H x W
        confusion_dict = {}
        # remember this is batched so average over the batch size 
        batch_dim = loc_pred.shape[0]

        # need to apply sigmoid to prediction to convert logit to prob
        loc_pred_prob = torch.sigmoid(loc_pred)
        loc_pred_logic = loc_pred_prob >= self.pos_thresh
        confusion_dict['max_pred_prob'] = torch.amax(loc_pred_prob, dim=(-2,-1)) # returns #B dim where max is over image dims 
        if global_contact_prob is not None:
            confusion_dict['global_contact_pred'] = (global_contact_prob >= 0.5).float()
        else:
            confusion_dict['global_contact_pred'] = (torch.sum(loc_pred_logic, (-2,-1)) > 0).float()
        
        self.num_samples += batch_dim 
        
        if not loc_target is None: 
            loc_target_logic = loc_target >= self.pos_thresh
            TP_local_batch = torch.sum(torch.logical_and(loc_pred_logic, loc_target_logic), dim=(-2, -1)) 
            TN_local_batch = torch.sum(torch.logical_not(torch.logical_or(loc_pred_logic, loc_target_logic)), dim=(-2, -1)) 
            xor_tensor_local = torch.logical_xor(loc_pred_logic, loc_target_logic)
            FP_local_batch = torch.sum(torch.logical_and(loc_pred_logic, xor_tensor_local), dim=(-2, -1)) 
            FN_local_batch = torch.sum(torch.logical_and(loc_target, xor_tensor_local), dim=(-2, -1)) 
            
            confusion_dict['local_fscore_batch'] = (2. * TP_local_batch) / (2.* TP_local_batch + FP_local_batch + FN_local_batch)
            confusion_dict['local_iou_batch'] = (TP_local_batch) / (TP_local_batch + FP_local_batch + FN_local_batch)
            confusion_dict['local_mcc_batch'] = (TP_local_batch * TN_local_batch - FP_local_batch * FN_local_batch) / (torch.sqrt((TP_local_batch + FP_local_batch)*(TP_local_batch + FN_local_batch)*(TN_local_batch + FP_local_batch)*(TN_local_batch + FN_local_batch)))
            confusion_dict['local_recall_batch'] = (TP_local_batch) / (TP_local_batch + FN_local_batch)
            confusion_dict['local_precision_batch'] = (TP_local_batch) / (TP_local_batch + FP_local_batch)
            confusion_dict['local_specificity_batch'] = (TN_local_batch) / (TN_local_batch + FP_local_batch)
            confusion_dict['local_accuracy_batch'] = (TP_local_batch + TN_local_batch) / (TP_local_batch + TN_local_batch + FP_local_batch + FN_local_batch) 
            
            self.TP_local_contact_total +=  torch.sum(TP_local_batch)
            self.TN_local_contact_total +=  torch.sum(TN_local_batch)
            self.FP_local_contact_total +=  torch.sum(FP_local_batch)
            self.FN_local_contact_total +=  torch.sum(FN_local_batch)

            confusion_dict['TP_local_batch_mean'] = torch.sum(TP_local_batch) / batch_dim 
            confusion_dict['TN_local_batch_mean'] = torch.sum(TN_local_batch) / batch_dim
            confusion_dict['FP_local_batch_mean'] = torch.sum(FP_local_batch) / batch_dim
            confusion_dict['FN_local_batch_mean'] = torch.sum(FN_local_batch) / batch_dim

            # # soft confusion matrix
            # soft_TP_local_batch = torch.sum(loc_pred_prob*loc_target_logic, dim=(-2, -1))
            # soft_TN_local_batch = torch.sum(torch.logical_not(torch.logical_or(loc_pred_prob >= self.soft_pos_thresh, loc_target_logic)), dim=(-2, -1))
            # soft_xor_tensor_local = torch.logical_xor(loc_pred_prob >= self.soft_pos_thresh, loc_target_logic)
            # soft_FP_local_batch = torch.sum(torch.logical_and(loc_pred_prob >= self.soft_pos_thresh, soft_xor_tensor_local), dim=(-2, -1))
            # soft_FN_local_batch = torch.sum(torch.logical_and(loc_target_logic, soft_xor_tensor_local), dim=(-2, -1))

            # confusion_dict['soft_local_fscore_batch'] = (2. * soft_TP_local_batch) / (2.* soft_TP_local_batch + soft_FP_local_batch + soft_FN_local_batch)
            # confusion_dict['soft_local_iou_batch'] = (soft_TP_local_batch) / (soft_TP_local_batch + soft_FP_local_batch + soft_FN_local_batch)
            # confusion_dict['soft_local_mcc_batch'] = (soft_TP_local_batch * soft_TN_local_batch - soft_FP_local_batch * soft_FN_local_batch) / (torch.sqrt((soft_TP_local_batch + soft_FP_local_batch)*(soft_TP_local_batch + soft_FN_local_batch)*(soft_TN_local_batch + soft_FP_local_batch)*(soft_TN_local_batch + soft_FN_local_batch)))
            # confusion_dict['soft_local_recall_batch'] = (soft_TP_local_batch) / (soft_TP_local_batch + soft_FN_local_batch)
            # confusion_dict['soft_local_precision_batch'] = (soft_TP_local_batch) / (soft_TP_local_batch + soft_FP_local_batch)
            # confusion_dict['soft_local_specificity_batch'] = (soft_TN_local_batch) / (soft_TN_local_batch + soft_FP_local_batch)
            # confusion_dict['soft_local_accuracy_batch'] = (soft_TP_local_batch + soft_TN_local_batch) / (soft_TP_local_batch + soft_TN_local_batch + soft_FP_local_batch + soft_FN_local_batch)

            # self.soft_TP_local_contact_total +=  torch.sum(soft_TP_local_batch)
            # self.soft_TN_local_contact_total +=  torch.sum(soft_TN_local_batch)
            # self.soft_FP_local_contact_total +=  torch.sum(soft_FP_local_batch)
            # self.soft_FN_local_contact_total +=  torch.sum(soft_FN_local_batch)

            # confusion_dict['soft_TP_local_batch_mean'] = torch.sum(soft_TP_local_batch) / batch_dim
            # confusion_dict['soft_TN_local_batch_mean'] = torch.sum(soft_TN_local_batch) / batch_dim
            # confusion_dict['soft_FP_local_batch_mean'] = torch.sum(soft_FP_local_batch) / batch_dim
            # confusion_dict['soft_FN_local_batch_mean'] = torch.sum(soft_FN_local_batch) / batch_dim

            # global confusion matrix
            confusion_dict['global_contact_gt'] = (torch.sum(loc_target_logic, (-2,-1)) > 0).float()
            
            TP_global = torch.sum(torch.logical_and(confusion_dict['global_contact_pred'], confusion_dict['global_contact_gt'])) 
            TN_global = torch.sum(torch.logical_not(torch.logical_or(confusion_dict['global_contact_pred'], confusion_dict['global_contact_gt']))) 
            xor_tensor_global = torch.logical_xor(confusion_dict['global_contact_pred'], confusion_dict['global_contact_gt'])
            FP_global = torch.sum(torch.logical_and(confusion_dict['global_contact_pred'], xor_tensor_global)) 
            FN_global = torch.sum(torch.logical_and(confusion_dict['global_contact_gt'], xor_tensor_global)) 

            self.TP_global_contact_total +=  TP_global
            self.TN_global_contact_total +=  TN_global
            self.FP_global_contact_total +=  FP_global
            self.FN_global_contact_total +=  FN_global

            confusion_dict['TP_global_batch_mean'] = TP_global / batch_dim 
            confusion_dict['TN_global_batch_mean'] = TN_global / batch_dim
            confusion_dict['FP_global_batch_mean'] = FP_global / batch_dim
            confusion_dict['FN_global_batch_mean'] = FN_global / batch_dim

        return confusion_dict
    
    def get_confusion_matrix_mean(self, if_global=False):
        if if_global:
            TP = self.TP_global_contact_total / self.num_samples
            TN = self.TN_global_contact_total / self.num_samples
            FP = self.FP_global_contact_total / self.num_samples
            FN = self.FN_global_contact_total / self.num_samples
        else:
            TP = self.TP_local_contact_total / self.num_samples
            TN = self.TN_local_contact_total / self.num_samples
            FP = self.FP_local_contact_total / self.num_samples
            FN = self.FN_local_contact_total / self.num_samples

        confusion_dict = {}
        confusion_dict['TP'] = TP
        confusion_dict['TN'] = TN
        confusion_dict['FP'] = FP
        confusion_dict['FN'] = FN

        return confusion_dict
    def add_global_contact_confusion_matrix(self, pred_prob, target_prob=None):
        confusion_dict = {}
        # remember this is batched so average over the batch size 
        batch_dim = target_prob.shape[0]

        # need to apply sigmoid to prediction to convert logit to prob
        pred_logic = pred_prob >= self.pos_thresh
        target_logic = target_prob >= self.pos_thresh
        TP = torch.sum(torch.logical_and(pred_logic, target_logic)) 
        TN = torch.sum(torch.logical_not(torch.logical_or(pred_logic, target_logic))) 
        xor_tensor = torch.logical_xor(pred_logic, target_logic)
        FP = torch.sum(torch.logical_and(pred_logic, xor_tensor)) 
        FN = torch.sum(torch.logical_and(target_logic, xor_tensor)) 

        self.TP_global_contact_total +=  TP
        self.TN_global_contact_total +=  TN
        self.FP_global_contact_total +=  FP
        self.FN_global_contact_total +=  FN

        # confusion_dict['TP_avg'] = TP / batch_dim 
        # confusion_dict['TN_avg'] = TN / batch_dim
        # confusion_dict['FP_avg'] = FP / batch_dim
        # confusion_dict['FN_avg'] = FN / batch_dim

        self.num_samples += batch_dim

        return confusion_dict

    def compute_fscore(self, if_global=False):
        if if_global:
            TP_total = self.TP_global_contact_total
            FP_total = self.FP_global_contact_total
            FN_total = self.FN_global_contact_total
        else:
            TP_total = self.TP_local_contact_total
            FP_total = self.FP_local_contact_total
            FN_total = self.FN_local_contact_total

        fscore_denom = 2.* TP_total + FP_total + FN_total
        if fscore_denom == 0:
            fscore_avg = torch.tensor(float('nan'))
        else:
            fscore_avg = (2. * TP_total) / fscore_denom
        return fscore_avg

    def compute_iou(self, if_global=False):
        if if_global:
            TP_total = self.TP_global_contact_total
            FP_total = self.FP_global_contact_total
            FN_total = self.FN_global_contact_total
        else:
            TP_total = self.TP_local_contact_total
            FP_total = self.FP_local_contact_total
            FN_total = self.FN_local_contact_total
        iou_denom = TP_total + FP_total + FN_total
        if iou_denom == 0:
            iou_avg = torch.tensor(float('nan'))
        else:
            iou_avg = TP_total / iou_denom
        return iou_avg
    
    def compute_MCC(self, if_global=False):
        if if_global:
            TP_total = self.TP_global_contact_total
            FP_total = self.FP_global_contact_total
            TN_total = self.TN_global_contact_total
            FN_total = self.FN_global_contact_total
        else:
            TP_total = self.TP_local_contact_total
            FP_total = self.FP_local_contact_total
            TN_total = self.TN_local_contact_total
            FN_total = self.FN_local_contact_total
        MCC_denom = (torch.sqrt((TP_total + FP_total)*(TP_total + FN_total)*(TN_total + FP_total)*(TN_total + FN_total)))
        if MCC_denom == 0:
            MCC_avg = torch.tensor(float('nan'))
        else:
            MCC_avg = (TP_total * TN_total - FP_total * FN_total) / MCC_denom 
        return MCC_avg

    def compute_recall(self, if_global=False):
        if if_global:
            TP_total = self.TP_global_contact_total
            FN_total = self.FN_global_contact_total
        else:
            TP_total = self.TP_local_contact_total
            FN_total = self.FN_local_contact_total

        recall_denom = TP_total + FN_total
        if recall_denom == 0:
            recall_avg = torch.tensor(float('nan'))
        else:
            recall_avg = TP_total  / recall_denom 
        return recall_avg
    
    def compute_precision(self, if_global=False):
        if if_global:
            TP_total = self.TP_global_contact_total
            FP_total = self.FP_global_contact_total
        else:
            TP_total = self.TP_local_contact_total
            FP_total = self.FP_local_contact_total

        precision_denom = TP_total + FP_total
        if precision_denom == 0:
            precision_avg = torch.tensor(float('nan'))
        else:
            precision_avg = TP_total  / precision_denom 
        return precision_avg

    def compute_specificity(self, if_global=False):
        if if_global:
            TN_total = self.TN_global_contact_total
            FP_total = self.FP_global_contact_total
        else:
            TN_total = self.TN_local_contact_total
            FP_total = self.FP_local_contact_total

        specificity_denom = TN_total + FP_total
        if specificity_denom == 0:
            specificity_avg = torch.tensor(float('nan'))
        else:
            specificity_avg = TN_total  / specificity_denom 
        return specificity_avg
    
    def compute_accuracy(self, if_global=False):
        if if_global:
            TP_total = self.TP_global_contact_total
            TN_total = self.TN_global_contact_total
            denom = self.TP_global_contact_total + self.TN_global_contact_total + self.FP_global_contact_total + self.FN_global_contact_total
        else:
            TP_total = self.TP_local_contact_total
            TN_total = self.TN_local_contact_total
            denom = self.num_samples * self.num_pixels_per_sample

        accuracy_avg = (TP_total + TN_total)  / denom 
        return accuracy_avg
    
    def compute_all_metrics(self, prefix, if_global=False):
        if if_global:
            metric_dict = { 

                prefix + 'global_fscore': self.compute_fscore(if_global=True),
                prefix + 'global_MCC': self.compute_MCC(if_global=True),
                prefix + 'global_precision': self.compute_precision(if_global=True),
                prefix + 'global_specificity': self.compute_specificity(if_global=True),
                prefix + 'global_recall': self.compute_recall(if_global=True),
                prefix + 'global_accuracy': self.compute_accuracy(if_global=True),
            }
        else:
            metric_dict = { 
                prefix + 'local_fscore': self.compute_fscore(),
                prefix + 'local_MCC': self.compute_MCC(),
                prefix + 'local_precision': self.compute_precision(),
                prefix + 'local_specificity': self.compute_specificity(),
                prefix + 'local_recall': self.compute_recall(),
                prefix + 'local_iou': self.compute_iou(),
                prefix + 'local_accuracy': self.compute_accuracy(),
                prefix + 'global_fscore': self.compute_fscore(if_global=True),
                prefix + 'global_MCC': self.compute_MCC(if_global=True),
                prefix + 'global_precision': self.compute_precision(if_global=True),
                prefix + 'global_specificity': self.compute_specificity(if_global=True),
                prefix + 'global_recall': self.compute_recall(if_global=True),
                prefix + 'global_accuracy': self.compute_accuracy(if_global=True),
            }
        return metric_dict
    
    def reset(self):
        self.TP_global_total = torch.tensor(0.).to(self.device) 
        self.FP_global_total = torch.tensor(0.).to(self.device) 
        self.FN_global_total = torch.tensor(0.).to(self.device) 
        self.TN_global_total = torch.tensor(0.).to(self.device) 

        self.TP_local_contact_total = torch.tensor(0.).to(self.device) 
        self.FP_local_contact_total = torch.tensor(0.).to(self.device) 
        self.FN_local_contact_total = torch.tensor(0.).to(self.device) 
        self.TN_local_contact_total = torch.tensor(0.).to(self.device) 

        self.num_samples = 0
