import torch 
import torchmetrics
from torchmetrics.aggregation import MeanMetric 
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import numpy as np

import torch.nn.functional as F

import cv2
    
class NMS_match(torchmetrics.Metric):
    is_differentiable: bool = False
    higher_is_better: bool = False
    full_state_update: bool = False
    def __init__(self, top_n_samples, max_num_contact, TP_pixel_radius, lambda_FP, lambda_FN, gaussian_kernel_size=5, prefix='', dist_sync_on_step=False, log_aggregate_metrics=False, per_sample_metrics=False, **kwargs):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.per_sample_metrics = per_sample_metrics

        self.max_num_contact = max_num_contact # max number of predictions we could have made
        self.top_n_samples = top_n_samples
        self.TP_pixel_radius = TP_pixel_radius

        self.gaussian_kernel_size = gaussian_kernel_size
        
        self.lambda_FP = lambda_FP
        self.lambda_FN = lambda_FN

        self.prefix = prefix

        self.log_aggregate_metrics = log_aggregate_metrics

        self.add_state("total_num_samples", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("cumsum_average_TP_distance", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("cumsum_FP_summed_distances", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("cumsum_FN_summed_distances", default=torch.tensor(0.0), dist_reduce_fx="sum")

        self.add_state("TP_count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("FP_count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("FN_count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("TN_count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("P_count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("N_count", default=torch.tensor(0), dist_reduce_fx="sum")

        self.add_state("valid_FP_distance_sample_count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("valid_FN_distance_sample_count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("non_empty_TP_sample_count", default=torch.tensor(0), dist_reduce_fx="sum")

        self.add_state("TP_cumsum_probability", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("FP_cumsum_probability", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("FN_cumsum_probability", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("TN_cumsum_probability", default=torch.tensor(0.0), dist_reduce_fx="sum")

        self.add_state("cumsum_integrated_probability_over_image", default=torch.tensor(0.0), dist_reduce_fx="sum")
        
        self.sample_average_TP_distance = torch.tensor(0.0).to(self.device)
        self.sample_FP_summed_distances = torch.tensor(0.0).to(self.device)
        self.sample_FN_summed_distances = torch.tensor(0.0).to(self.device)

        self.sample_FP_count = torch.tensor(0).to(self.device)
        self.sample_FN_count = torch.tensor(0).to(self.device)
        self.sample_TP_count = torch.tensor(0).to(self.device)
        self.sample_TN_count = torch.tensor(0).to(self.device)
        self.sample_P_count = torch.tensor(0).to(self.device)
        self.sample_N_count = torch.tensor(0).to(self.device)

        self.sample_valid_FP_distance_sample_count = torch.tensor(0).to(self.device)
        self.sample_valid_FN_distance_sample_count = torch.tensor(0).to(self.device)
        self.sample_non_empty_TP_sample_count = torch.tensor(0).to(self.device)

        self.sample_TP_average_probability = torch.tensor(0.0).to(self.device)
        self.sample_FP_average_probability = torch.tensor(0.0).to(self.device)
        self.sample_FN_average_probability = torch.tensor(0.0).to(self.device)
        self.sample_TN_average_probability = torch.tensor(0.0).to(self.device)

        self.sample_integrated_probability_over_image = torch.tensor(0.0).to(self.device)

    def update(self, nms_pixel_list, y_hat, y_target):
        # accepts a list of len B where each element is a tensor of shape (num_pixels, 2)
        # B x H_OG x W_OG
        # y_target = y_target.squeeze()
        assert len(nms_pixel_list) == y_target.shape[0], "nms_pixel_list and y_target must have the same batch size"

        # reset the sample metrics so that they dont accumulate over samples
        self.sample_average_TP_distance = torch.tensor(0.0).to(self.device)
        self.sample_FP_summed_distances = torch.tensor(0.0).to(self.device)
        self.sample_FN_summed_distances = torch.tensor(0.0).to(self.device)
        
        self.sample_FP_count = torch.tensor(0).to(self.device)
        self.sample_FN_count = torch.tensor(0).to(self.device)
        self.sample_TP_count = torch.tensor(0).to(self.device)
        self.sample_TN_count = torch.tensor(0).to(self.device)
        self.sample_P_count = torch.tensor(0).to(self.device)
        self.sample_N_count = torch.tensor(0).to(self.device)

        self.sample_valid_FP_distance_sample_count = torch.tensor(0).to(self.device)
        self.sample_valid_FN_distance_sample_count = torch.tensor(0).to(self.device)
        self.sample_non_empty_TP_sample_count = torch.tensor(0).to(self.device)
        
        self.sample_TP_average_probability = torch.tensor(0.0).to(self.device)
        self.sample_FP_average_probability = torch.tensor(0.0).to(self.device)
        self.sample_FN_average_probability = torch.tensor(0.0).to(self.device)
        self.sample_integrated_probability_over_image = torch.tensor(0.0).to(self.device)

        # loop through the batch
        for i in range(y_target.shape[0]):
            y_hat_i = y_hat[i]
            y_target_i = y_target[i]
            assert y_hat_i.shape == y_target_i.shape, "y_hat and y_target must have the same shape"

            # compute the integrated probability over the image
            self.sample_integrated_probability_over_image += torch.sum(y_hat_i)

            gt_features = torch.stack((torch.where(y_target_i)[0], torch.where(y_target_i)[1])).T   # One feature: [v, u]
            # if there is no ground truth, gt_features = tensor([], size=(0, 2))
            pred_features = nms_pixel_list[i]

            ### Counting ###
            if gt_features.shape[0] > self.max_num_contact:
                # warn that max num contact is too small and how many gt contacts were missed
                print(f"Warning: max_num_contact is too small at {self.max_num_contact} and {gt_features.shape[0] - self.max_num_contact} gt contacts were missed")

            self.sample_N_count += max(self.max_num_contact - gt_features.shape[0], 0)
            self.sample_P_count += gt_features.shape[0]

            # choose top n samples here instead of max num contact because the top_n_samples trades off how many negatives you could have predicted 
            self.sample_TN_count = torch.tensor(max(self.top_n_samples - pred_features.shape[0], 0))

            # if either prediction or target set is empty...
            if pred_features.shape[0] == 0 or gt_features.shape[0] == 0:
                self.sample_FP_count += max(0, pred_features.shape[0] - gt_features.shape[0])
                self.sample_FN_count += max(0, gt_features.shape[0] - pred_features.shape[0])
                
                if self.sample_FP_count > 0:
                    # generate a mask of gaussian kernel window size around TP pixels to get the sum of prediction probabilities
                    pred_mask = np.zeros((y_hat_i.shape[0], y_hat_i.shape[1]))
                    # pred features is a tensor of shape (num_pixels, 2)
                    for pred_feature in pred_features:
                        # convert to int to index into the mask
                        pred_mask[int(pred_feature[0]), int(pred_feature[1])] = 1
                    # use cv2 to apply a gaussian blur to the mask using self.gaussian_kernel_size
                    pred_mask = cv2.GaussianBlur(pred_mask, (self.gaussian_kernel_size, self.gaussian_kernel_size), 0)
                    # convert any non-zero entries to 1
                    pred_mask[pred_mask != 0] = 1
                    # sum prediction probabilities y_hat_i over the mask
                    self.sample_FP_average_probability += torch.sum(y_hat_i * torch.tensor(pred_mask).to(self.device))
                    # at the end, divide by the number of TPs to get the average probability
                elif self.sample_FN_count > 0:
                    # generate a mask of gaussian kernel window size around TP pixels to get the sum of prediction probabilities
                    gt_mask = np.zeros((y_hat_i.shape[0], y_hat_i.shape[1]))
                    # gt features is a tensor of shape (num_pixels, 2)
                    for gt_feature in gt_features:
                        # convert to int to index into the mask
                        gt_mask[int(gt_feature[0]), int(gt_feature[1])] = 1
                    # use cv2 to apply a gaussian blur to the mask using self.gaussian_kernel_size
                    gt_mask = cv2.GaussianBlur(gt_mask, (self.gaussian_kernel_size, self.gaussian_kernel_size), 0)
                    # convert any non-zero entries to 1
                    gt_mask[gt_mask != 0] = 1
                    # sum prediction probabilities y_hat_i over the mask
                    self.sample_FN_average_probability += torch.sum(y_hat_i * torch.tensor(gt_mask).to(self.device))

                # both could be empty
                if pred_features.shape[0] == 0 and gt_features.shape[0] == 0:
                    self.sample_valid_FP_distance_sample_count += 1
                    self.sample_valid_FN_distance_sample_count += 1
            # only run matching if gt or pred are non-empty
            else:
                self.sample_valid_FP_distance_sample_count += 1
                self.sample_valid_FN_distance_sample_count += 1

                #### Matching ####

                # if either is empty, then the cost matrix is empty (shape is (0,1))
                # axis 0 is pred, axis 1 is gt
                cost_matrix = cdist(pred_features, gt_features, 'euclidean')
                row_ind, col_ind = linear_sum_assignment(cost_matrix)
                # filter out the matches that are too far away
                row_ind, col_ind = row_ind[cost_matrix[row_ind, col_ind] < self.TP_pixel_radius], col_ind[cost_matrix[row_ind, col_ind] < self.TP_pixel_radius]
                # store the row_ind and col_ind that were unmatched
                # empty if all were matched
                unmatched_row_ind = np.setdiff1d(np.arange(pred_features.shape[0]), row_ind, assume_unique=True)
                unmatched_col_ind = np.setdiff1d(np.arange(gt_features.shape[0]), col_ind, assume_unique=True)

                sample_TP_count = row_ind.shape[0]
                self.sample_TP_count += sample_TP_count
                sample_FP_count = pred_features.shape[0] - sample_TP_count
                self.sample_FP_count += sample_FP_count
                sample_FN_count = gt_features.shape[0] - sample_TP_count
                self.sample_FN_count += sample_FN_count
               
                if sample_TP_count > 0:
                    self.sample_non_empty_TP_sample_count += 1

                # mean of empty matrix is nan
                # we can just set this to zero and still sum since we average over TPs later
                sample_average_TP_distance = cost_matrix[row_ind, col_ind].mean()
                if not np.isnan(sample_average_TP_distance): # have to do this because any number + nan = nan
                    self.sample_average_TP_distance += sample_average_TP_distance
                    
                    # generate a mask of gaussian kernel window size around TP pixels to get the sum of prediction probabilities
                    pred_mask = np.zeros((y_hat_i.shape[0], y_hat_i.shape[1]))
                    # pred_mask[pred_features[row_ind, 0], pred_features[row_ind, 1]] = 1
                    for j in range(row_ind.shape[0]):
                        pred_mask[int(pred_features[row_ind[j], 0]), int(pred_features[row_ind[j], 1])] = 1
                    # use cv2 to apply a gaussian blur to the mask using self.gaussian_kernel_size
                    pred_mask = cv2.GaussianBlur(pred_mask, (self.gaussian_kernel_size, self.gaussian_kernel_size), 0)
                    # convert any non-zero entries to 1
                    pred_mask[pred_mask != 0] = 1
                    # sum prediction probabilities y_hat_i over the mask
                    self.sample_TP_average_probability += torch.sum(y_hat_i * torch.tensor(pred_mask).to(self.device))
                    # at the end, divide by the number of TPs to get the average


                # set relevant metrics to nans if either FP or FN count is zero
                # Can run linear_sum_assignment on the unmatched cost matrix to get the FP and FN cost, assuming both GT and pred are non-empty
                if sample_FP_count > 0 or sample_FN_count > 0:
                    unmatched_cost_matrix = cost_matrix.copy()
                    unmatched_cost_matrix[row_ind, col_ind] = np.nan
                    # if either axis is dim 1, then remove the matched entries from the unmatched cost matrix
                    if sample_FP_count > 0:
                        # for cumsum_FP_summed_distances, we want to sum over the distances of the unmatched pred features
                        # take the min of the unmatched cost matrix over the pred features and sum
                        # min over axis 1 is matching each FP to the closest GT
                        self.sample_FP_summed_distances += np.nansum(np.nanmin(unmatched_cost_matrix, axis=1))

                        # generate a mask of gaussian kernel window size around TP pixels to get the sum of prediction probabilities
                        pred_mask = np.zeros((y_hat_i.shape[0], y_hat_i.shape[1]))
                        # pred features is a tensor of shape (num_pixels, 2)
                        for j in range(unmatched_row_ind.shape[0]):
                            # convert to int to index into the mask
                            pred_mask[int(pred_features[unmatched_row_ind[j], 0]), int(pred_features[unmatched_row_ind[j], 1])] = 1
                        # use cv2 to apply a gaussian blur to the mask using self.gaussian_kernel_size
                        pred_mask = cv2.GaussianBlur(pred_mask, (self.gaussian_kernel_size, self.gaussian_kernel_size), 0)
                        # convert any non-zero entries to 1
                        pred_mask[pred_mask != 0] = 1
                        # sum prediction probabilities y_hat_i over the mask
                        self.sample_FP_average_probability += torch.sum(y_hat_i * torch.tensor(pred_mask).to(self.device))
                        # at the end, divide by the number of TPs to get the average probability

                    if sample_FN_count > 0:
                        # for cumsum_FN_summed_distances, we want to sum over the distances of the unmatched gt features
                        # take the min of the unmatched cost matrix over the gt features and sum
                        # min over axis 0 is matching each FN to the closest pred
                        self.sample_FN_summed_distances += np.nansum(np.nanmin(unmatched_cost_matrix, axis=0))
                        # unmatched_row_ind, unmatched_col_ind = linear_sum_assignment(unmatched_cost_matrix)

                        # generate a mask of gaussian kernel window size around TP pixels to get the sum of prediction probabilities
                        gt_mask = np.zeros((y_hat_i.shape[0], y_hat_i.shape[1]))
                        # gt features is a tensor of shape (num_pixels, 2)
                        for j in range(unmatched_col_ind.shape[0]):
                            # convert to int to index into the mask
                            gt_mask[int(gt_features[unmatched_col_ind[j], 0]), int(gt_features[unmatched_col_ind[j], 1])] = 1
                        # use cv2 to apply a gaussian blur to the mask using self.gaussian_kernel_size
                        gt_mask = cv2.GaussianBlur(gt_mask, (self.gaussian_kernel_size, self.gaussian_kernel_size), 0)
                        # convert any non-zero entries to 1
                        gt_mask[gt_mask != 0] = 1
                        # sum prediction probabilities y_hat_i over the mask
                        self.sample_FN_average_probability += torch.sum(y_hat_i * torch.tensor(gt_mask).to(self.device))

        # accumulate number of samples
        self.TP_count += self.sample_TP_count
        self.FP_count += self.sample_FP_count
        self.TN_count += self.sample_TN_count
        self.FN_count += self.sample_FN_count
        self.N_count += self.sample_N_count
        self.P_count += self.sample_P_count

        self.cumsum_average_TP_distance += torch.nan_to_num(self.sample_average_TP_distance)
        self.cumsum_FP_summed_distances += torch.nan_to_num(self.sample_FP_summed_distances)
        self.cumsum_FN_summed_distances += torch.nan_to_num(self.sample_FN_summed_distances)
        
        self.valid_FP_distance_sample_count += self.sample_valid_FP_distance_sample_count
        self.valid_FN_distance_sample_count += self.sample_valid_FN_distance_sample_count
        self.non_empty_TP_sample_count += self.sample_non_empty_TP_sample_count
        
        self.TP_cumsum_probability += torch.nan_to_num(self.sample_TP_average_probability)
        self.FP_cumsum_probability += torch.nan_to_num(self.sample_FP_average_probability)
        self.FN_cumsum_probability += torch.nan_to_num(self.sample_FN_average_probability)
        # divide these at compute stage by TP, FP, FN counts to get average probabilities
        self.cumsum_integrated_probability_over_image += torch.nan_to_num(self.sample_integrated_probability_over_image)
        # divide this at compute stage by total number of samples to get average

        self.total_num_samples += len(nms_pixel_list)

        if self.per_sample_metrics:
            self.sample_integrated_probability_over_image /= len(nms_pixel_list)
            # if there were no TPs, then sample_average_TP_distance should be nan
            if self.sample_TP_count == 0:
                self.sample_average_TP_distance = torch.tensor(np.nan).to(self.device)
                self.sample_TP_average_probability = torch.tensor(np.nan).to(self.device)
            else:
                self.sample_TP_average_probability /= self.sample_TP_count
            if self.sample_FP_count > 0:
                self.sample_FP_average_probability /= self.sample_FP_count
                if self.sample_P_count == 0: # if there were no GTs, then sample_cumsum_FP_summed_distances should be nan
                    self.sample_FP_summed_distances = torch.tensor(np.nan).to(self.device)
            else:
                self.sample_FP_average_probability = torch.tensor(np.nan).to(self.device)
            if (self.sample_TP_count + self.sample_FP_count) == 0 and self.sample_P_count > 0: # if there were no preds, then sample_cumsum_FN_summed_distances should be nan:
                self.sample_FN_summed_distances = torch.tensor(np.nan).to(self.device)
            if self.sample_FN_count > 0:
                self.sample_FN_average_probability /= self.sample_FN_count
            else:
                self.sample_FN_average_probability = torch.tensor(np.nan).to(self.device)

    def compute(self):
        metrics = {}
        # this is already averaged over TPs for each sample, but its not averaged over samples yet
        average_TP_distance_over_TPs = torch.nan_to_num(self.cumsum_average_TP_distance/self.non_empty_TP_sample_count, nan=torch.inf)
        # FP_distance only valid when gt is non-empty
        if self.valid_FP_distance_sample_count == 0:
            average_FP_summed_distance = torch.tensor(np.nan).to(self.device)
        else:
            average_FP_summed_distance = self.cumsum_FP_summed_distances/self.valid_FP_distance_sample_count
        # FN_distance only valid when pred is non-empty
        if self.valid_FN_distance_sample_count == 0:
            average_FN_summed_distance = torch.tensor(np.nan).to(self.device)
        else:
            average_FN_summed_distance = self.cumsum_FN_summed_distances/self.valid_FN_distance_sample_count

        average_FP_count = self.FP_count/self.total_num_samples
        average_FN_count = self.FN_count/self.total_num_samples
        average_TP_count = self.TP_count/self.total_num_samples
        average_TN_count = self.TN_count/self.total_num_samples
        average_P_count = self.P_count/self.total_num_samples

        if self.TP_count > 0:
            TP_average_probability = self.TP_cumsum_probability/self.TP_count
        else:
            TP_average_probability = torch.tensor(np.nan).to(self.device)
        if self.FP_count > 0:
            FP_average_probability = self.FP_cumsum_probability/self.FP_count
        else:
            FP_average_probability = torch.tensor(np.nan).to(self.device)
        if self.FN_count > 0:
            FN_average_probability = self.FN_cumsum_probability/self.FN_count
        else:
            FN_average_probability = torch.tensor(np.nan).to(self.device)
        average_integrated_probability_over_image = self.cumsum_integrated_probability_over_image / self.total_num_samples

        TP_average_probability_regression_loss = TP_average_probability - 1.
        FP_average_probability_regression_loss = FP_average_probability
        FN_average_probability_regression_loss = FN_average_probability - 1.
        average_integrated_probability_regression_loss = average_integrated_probability_over_image - average_P_count

        accuracy = torch.nan_to_num((self.TP_count + self.TN_count)/(self.P_count + self.N_count))
        specificity = torch.nan_to_num(self.TN_count/self.N_count) # true negative rate
        sensitivity = torch.nan_to_num(self.TP_count/self.P_count) # true positive rate
        precision = torch.nan_to_num(self.TP_count/(self.TP_count + self.FP_count)) # positive predictive value
        F1 = torch.nan_to_num(2 * (precision * sensitivity) / (precision + sensitivity))

        # combine relevant metrics into one scalar monolothic metric
        monolithic = average_TP_distance_over_TPs + self.lambda_FP*average_FP_summed_distance + self.lambda_FN*average_FN_summed_distance

        eval_aggregate_metrics = {
            self.prefix + 'NMS_P_count': self.P_count,
            self.prefix + 'NMS_N_count': self.N_count,
            self.prefix + 'NMS_TP_count': self.TP_count,
            self.prefix + 'NMS_TN_count': self.TN_count,
            self.prefix + 'NMS_FP_count': self.FP_count,
            self.prefix + 'NMS_FN_count': self.FN_count,
            self.prefix + 'NMS_cumsum_average_TP_distance': self.cumsum_average_TP_distance,
            self.prefix + 'NMS_cumsum_FP_summed_distances': self.cumsum_FP_summed_distances,
            self.prefix + 'NMS_cumsum_FN_summed_distances': self.cumsum_FN_summed_distances,
            self.prefix + 'NMS_non_empty_TP_sample_count': self.non_empty_TP_sample_count,
            self.prefix + 'NMS_valid_FP_distance_sample_count': self.valid_FP_distance_sample_count,
            self.prefix + 'NMS_valid_FN_distance_sample_count': self.valid_FN_distance_sample_count,
            self.prefix + 'NMS_TP_cumsum_probability': self.TP_cumsum_probability,
            self.prefix + 'NMS_FP_cumsum_probability': self.FP_cumsum_probability,
            self.prefix + 'NMS_FN_cumsum_probability': self.FN_cumsum_probability,
            self.prefix + 'NMS_cumsum_integrated_probability_over_image': self.cumsum_integrated_probability_over_image,
            self.prefix + 'NMS_total_num_samples': self.total_num_samples,
        }
        metrics.update(eval_aggregate_metrics)
        
        aggregate_metrics = {
            self.prefix + 'NMS_TP_average_probability': TP_average_probability,
            self.prefix + 'NMS_FP_average_probability': FP_average_probability,
            self.prefix + 'NMS_FN_average_probability': FN_average_probability,
            self.prefix + 'NMS_average_integrated_probability_over_image': average_integrated_probability_over_image,
            self.prefix + 'NMS_TP_average_probability_regression_loss': TP_average_probability_regression_loss,
            self.prefix + 'NMS_FP_average_probability_regression_loss': FP_average_probability_regression_loss,
            self.prefix + 'NMS_FN_average_probability_regression_loss': FN_average_probability_regression_loss,
            self.prefix + 'NMS_average_integrated_probability_regression_loss': average_integrated_probability_regression_loss,
            self.prefix + 'NMS_average_TP_distance_over_TPs': average_TP_distance_over_TPs,
            self.prefix + 'NMS_average_FP_summed_distance': average_FP_summed_distance,
            self.prefix + 'NMS_average_FN_summed_distance': average_FN_summed_distance,
            self.prefix + 'NMS_average_FP_count': average_FP_count,
            self.prefix + 'NMS_average_FN_count': average_FN_count,
            self.prefix + 'NMS_average_TP_count': average_TP_count,
            self.prefix + 'NMS_average_TN_count': average_TN_count,
            self.prefix + 'NMS_accuracy': accuracy,
            self.prefix + 'NMS_specificity': specificity,
            self.prefix + 'NMS_sensitivity': sensitivity,
            self.prefix + 'NMS_precision': precision,
            self.prefix + 'NMS_F1': F1,
            self.prefix + 'NMS_monolithic': monolithic,
        }
        metrics.update(aggregate_metrics)
        
        # self.prefix + 'NMS_sample_monolithic': sample_monolithic Cant compute this as FP and FN are sometimes nan
        if self.per_sample_metrics:
            # if divisor is 0 then set to nan for calculating accuracy, specificity, etc...
            if (self.sample_P_count + self.sample_N_count) == 0:
                # set metric to nan
                sample_accuracy = torch.tensor(np.nan).to(self.device)
            else:
                sample_accuracy = (self.sample_TP_count + self.sample_TN_count)/(self.sample_P_count + self.sample_N_count)
            if self.sample_N_count == 0:
                sample_specificity = torch.tensor(np.nan).to(self.device)
            else:
                sample_specificity = self.sample_TN_count/self.sample_N_count # true negative rate
            if self.sample_P_count == 0:
                sample_sensitivity = torch.tensor(np.nan).to(self.device)
            else:
                sample_sensitivity = self.sample_TP_count/self.sample_P_count # true positive rate
            if (self.sample_TP_count + self.sample_FP_count) == 0:
                sample_precision = torch.tensor(np.nan).to(self.device)
            else:
                sample_precision = self.sample_TP_count/(self.sample_TP_count + self.sample_FP_count) # positive predictive value
            if torch.isnan(sample_precision + sample_sensitivity) or (sample_precision + sample_sensitivity) == 0:
                sample_F1 = torch.tensor(np.nan).to(self.device)
            else:
                sample_F1 = 2 * (sample_precision * sample_sensitivity) / (sample_precision + sample_sensitivity)
            # sample_monolithic = self.sample_average_TP_distance + self.lambda_FP*self.sample_FP_summed_distance + self.lambda_FN*self.sample_FN_summed_distance

            # compute the probability regression losses
            # absolute value of the difference between the average predicted probability of the TP pixels and 1
            average_TP_probability_regression_loss = self.sample_TP_average_probability - 1
            # absolute value of the difference between the average predicted probability of the FP pixels and 0
            average_FP_probability_regression_loss = self.sample_FP_average_probability - 0
            # absolute value of the difference between the average predicted probability of the FN pixels and 1
            average_FN_probability_regression_loss = self.sample_FN_average_probability - 1

            integrated_probability_regression_loss = self.sample_integrated_probability_over_image - self.sample_P_count 

            per_sample_metrics = {
                self.prefix + 'NMS_sample_average_TP_distance': self.sample_average_TP_distance,
                self.prefix + 'NMS_sample_FP_summed_distances': self.sample_FP_summed_distances,
                self.prefix + 'NMS_sample_FN_summed_distances': self.sample_FN_summed_distances,
                self.prefix + 'NMS_sample_FP_count': self.sample_FP_count,
                self.prefix + 'NMS_sample_FN_count': self.sample_FN_count,
                self.prefix + 'NMS_sample_TP_count': self.sample_TP_count,
                self.prefix + 'NMS_sample_TN_count': self.sample_TN_count,
                self.prefix + 'NMS_sample_P_count': self.sample_P_count,
                self.prefix + 'NMS_sample_N_count': self.sample_N_count,
                self.prefix + 'NMS_sample_valid_FP_distance_sample_count': self.sample_valid_FP_distance_sample_count,
                self.prefix + 'NMS_sample_valid_FN_distance_sample_count': self.sample_valid_FN_distance_sample_count,
                self.prefix + 'NMS_sample_non_empty_TP_sample_count': self.sample_non_empty_TP_sample_count,
                self.prefix + 'NMS_sample_accuracy': sample_accuracy,
                self.prefix + 'NMS_sample_specificity': sample_specificity,
                self.prefix + 'NMS_sample_sensitivity': sample_sensitivity,
                self.prefix + 'NMS_sample_precision': sample_precision,
                self.prefix + 'NMS_sample_F1': sample_F1,
                self.prefix + 'NMS_sample_TP_average_probability': self.sample_TP_average_probability,
                self.prefix + 'NMS_sample_FP_average_probability': self.sample_FP_average_probability,
                self.prefix + 'NMS_sample_FN_average_probability': self.sample_FN_average_probability,
                self.prefix + 'NMS_sample_integrated_probability_over_image': self.sample_integrated_probability_over_image,
                self.prefix + 'NMS_sample_average_TP_probability_regression_loss': average_TP_probability_regression_loss,
                self.prefix + 'NMS_sample_average_FP_probability_regression_loss': average_FP_probability_regression_loss,
                self.prefix + 'NMS_sample_average_FN_probability_regression_loss': average_FN_probability_regression_loss,
                self.prefix + 'NMS_sample_integrated_probability_regression_loss': integrated_probability_regression_loss,
            }
            metrics.update(per_sample_metrics)
            
        return metrics
    
    @staticmethod
    def compute_accuracy(TP_count, TN_count, P_count, N_count):
        # if arguments are torch tensors then return a torch tensor
        if (P_count + N_count) == 0:
            if isinstance(TP_count, torch.Tensor):
                return torch.tensor(np.nan).to(TP_count.device)
        else:
            return (TP_count + TN_count)/(P_count + N_count)
    @staticmethod
    def compute_specificity(TN_count, N_count):
        if N_count == 0:
            if isinstance(TN_count, torch.Tensor):
                return torch.tensor(np.nan).to(TN_count.device)
        else:
            return TN_count/N_count
    @staticmethod
    def compute_sensitivity(TP_count, P_count):
        if P_count == 0:
            if isinstance(TP_count, torch.Tensor):
                return torch.tensor(np.nan).to(TP_count.device)
        else:
            return TP_count/P_count
    @staticmethod
    def compute_precision(TP_count, FP_count):
        if (TP_count + FP_count) == 0:
            if isinstance(TP_count, torch.Tensor):
                return torch.tensor(np.nan).to(TP_count.device)
        else:
            return TP_count/(TP_count + FP_count)
    @staticmethod
    def compute_F1_score(TP_count, FP_count, FN_count):
        if (TP_count + FP_count + FN_count) == 0:
            if isinstance(TP_count, torch.Tensor):
                return torch.tensor(np.nan).to(TP_count.device)
        else:
            return 2*TP_count/(2*TP_count + FP_count + FN_count)
    @staticmethod
    def compute_average_TP_distance_over_TPs(cumsum_average_TP_distance, valid_TP_sample_count):
        if valid_TP_sample_count == 0:
            if isinstance(cumsum_average_TP_distance, torch.Tensor):
                return torch.tensor(np.nan).to(cumsum_average_TP_distance.device)
        else:
            return cumsum_average_TP_distance/valid_TP_sample_count
    @staticmethod
    def compute_average_FP_summed_distance(cumsum_FP_summed_distances, valid_FP_distance_sample_count):
        if valid_FP_distance_sample_count == 0:
            if isinstance(cumsum_FP_summed_distances, torch.Tensor):
                return torch.tensor(np.nan).to(cumsum_FP_summed_distances.device)
        else:
            return cumsum_FP_summed_distances/valid_FP_distance_sample_count
    @staticmethod
    def compute_average_FN_summed_distance(cumsum_FN_summed_distances, valid_FN_distance_sample_count):
        if valid_FN_distance_sample_count == 0:
            if isinstance(cumsum_FN_summed_distances, torch.Tensor):
                return torch.tensor(np.nan).to(cumsum_FN_summed_distances.device)
        else:
            return cumsum_FN_summed_distances/valid_FN_distance_sample_count
    @staticmethod
    def compute_TP_average_probability(cumsum_TP_average_probability, TP_count):
        if TP_count == 0:
            if isinstance(cumsum_TP_average_probability, torch.Tensor):
                return torch.tensor(np.nan).to(cumsum_TP_average_probability.device)
        else:
            return cumsum_TP_average_probability/TP_count
    @staticmethod
    def compute_FP_average_probability(cumsum_FP_average_probability, FP_count):
        if FP_count == 0:
            if isinstance(cumsum_FP_average_probability, torch.Tensor):
                return torch.tensor(np.nan).to(cumsum_FP_average_probability.device)
        else:
            return cumsum_FP_average_probability/FP_count
    @staticmethod
    def compute_FN_average_probability(cumsum_FN_average_probability, FN_count):
        if FN_count == 0:
            if isinstance(cumsum_FN_average_probability, torch.Tensor):
                return torch.tensor(np.nan).to(cumsum_FN_average_probability.device)
        else:
            return cumsum_FN_average_probability/FN_count
    @staticmethod
    def compute_average_integrated_probability_over_image(cumsum_integrated_probability_over_image, total_num_samples):
        if total_num_samples == 0:
            if isinstance(cumsum_integrated_probability_over_image, torch.Tensor):
                return torch.tensor(np.nan).to(cumsum_integrated_probability_over_image.device)
        else:
            return cumsum_integrated_probability_over_image/total_num_samples
    @staticmethod
    def compute_average_TP_probability_loss(TP_average_probability):
        return TP_average_probability - 1.
    @staticmethod
    def compute_average_FP_probability_loss(FP_average_probability):
        return FP_average_probability
    @staticmethod
    def compute_average_FN_probability_loss(FN_average_probability):
        return FN_average_probability - 1.
    @staticmethod
    def compute_average_integrated_probability_over_image_loss(average_integrated_probability_over_image, average_P_count):
        return average_integrated_probability_over_image - average_P_count

class NMS_L2_norm(torchmetrics.Metric):
    is_differentiable: bool = False
    higher_is_better: bool = False
    full_state_update: bool = False
    def __init__(self, dist_sync_on_step=False, half_window_size=5):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.half_window_size = half_window_size
        self.add_state("mean_NMS_L2_over_contacts", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_num_samples", default=torch.tensor(0), dist_reduce_fx="sum")
        # self.mean_metric = MeanMetric(nan_strategy=)

    def update(self, y_pred_logit, y_target):
        # B x 1 x H_OG x W_OG
        y_pred = torch.sigmoid(y_pred_logit)
        y_target = y_target.squeeze()
        nms_loss = 0
        tot_num_contacts = 0
        for idx in range(y_pred.shape[0]):
            gt = torch.where(y_target[idx])  # the location of gt in target
            num_contact = gt[0].shape[0]
            tot_num_contacts += num_contact
            if num_contact == 0:
                continue
            for num in range(num_contact):
                v_contact = int(gt[0][num])
                u_contact = int(gt[1][num])
                x1 = max(0, u_contact-self.half_window_size)
                x2 = min(y_pred.shape[-1], u_contact+self.half_window_size+1)
                y1 = max(0, v_contact-self.half_window_size)
                y2 = min(y_pred.shape[-2], v_contact+self.half_window_size+1)
                max_pred = torch.argmax(y_pred[idx,y1:y2,x1:x2]) # 11, 11
                h_window, w_window = y2-y1, x2-x1
                v_pred, u_pred = max_pred//w_window, max_pred%w_window
                v_contact_in_window, u_contact_in_window = v_contact-y1, u_contact-x1
                nms_loss += torch.sqrt((v_pred - v_contact_in_window)**2 + (u_pred - u_contact_in_window)**2)
        if tot_num_contacts == 0:
            # meaningless to compute loss so just pass
            pass
        else:
            # on step, the global state is reset
            # but on epoch, the global state is not reset
            self.mean_NMS_L2_over_contacts += nms_loss/tot_num_contacts
            self.total_num_samples += 1

    def compute(self):
        return self.mean_NMS_L2_over_contacts/self.total_num_samples
