# import argparse
# import logging
# from pickletools import uint8
import sys
import os
sys.path.append('../../')
# from RAFT.core.utils.flow_viz import flow_to_image
from src.utils.flow_viz import flow_to_image
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, random_split
import torchvision

import os
import numpy as np
import copy
import cv2
import pickle

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

def tensor_to_depth_map(depth_tensor):
    tensor_to_float32 = torchvision.transforms.ConvertImageDtype(torch.float32)
    return np.array(tensor_to_float32(depth_tensor))

def depth_map_to_im(depth_map, colormap, max_depth_clip=2.0, viz_max_depth_clip=1.3, viz_min_depth_clip=0.3, return_BGR = False):
    nan_depth_mask = depth_map == 0
    if viz_max_depth_clip is not None:
        unnorm_image_np = depth_map*max_depth_clip # back to meters
        image_np_clipped = np.clip(unnorm_image_np, viz_min_depth_clip, viz_max_depth_clip) - viz_min_depth_clip
        image_np = ((255. / (viz_max_depth_clip - viz_min_depth_clip)) * image_np_clipped)
    else: 
        image_np = (255. * image_np) 

    image_np_uint8 = (255.-image_np).astype(np.uint8) # invert the depth map so that colormap is also inverted to be more white than black
    image_np_color = cv2.applyColorMap(image_np_uint8, colormap)
    image_np_color[nan_depth_mask] = 255 # color the nan depths white
    if return_BGR:
        return image_np_color
    else:
        return cv2.cvtColor(image_np_color, cv2.COLOR_BGR2RGB)
    
def path_to_im(im_resize, im_path):
    color_im = cv2.imread(im_path)
    color_im_resize = cv2.resize(color_im, (im_resize[1], im_resize[0]), interpolation=cv2.INTER_CUBIC)
    # DONT NEED GRAY2RGB CONVERSION AS IMREAD AUTOMATICALLY CONVERTS, UNLESS FLAG IS SET!
    # if is_gray: 
    #     color_im_resize = cv2.cvtColor(color_im_resize, cv2.COLOR_GRAY2BGR) # H x W x 3
    image_np_color = np.array(color_im_resize)
    return image_np_color

def tensor_to_colormap_im(im_tensor, colormap):
    NotImplementedError
    # expects im_tensor of HxW dimension
    # returns np_array of HxWx3 RGB
    # tensor_to_pil = torchvision.transforms.functional.to_pil_image
    im_tensor_np = np.array(im_tensor)
    return cv2.cvtColor(cv2.applyColorMap(im_tensor_np, colormap), cv2.COLOR_BGR2RGB)

def overlay_im_list(im_list, alpha_list, return_BGR=False):
    # im_list is a list of np_array of HxWx3 BGR
    # alpha_list is a list of alpha values for each image
    # im_size is a tuple (H, W)
    # returns a single np_array of HxWx3 RGB
    assert len(im_list) == len(alpha_list)
    assert len(im_list) > 0
    base_im = np.zeros(im_list[0].shape)
    for i in range(len(im_list)):
        assert im_list[i].shape == base_im.shape
        base_im += (alpha_list[i]*im_list[i]).astype(np.uint8)
    base_im = np.clip(base_im, a_min=0., a_max=255.)
    if not return_BGR:
        base_im = cv2.cvtColor(base_im.astype(np.uint8), cv2.COLOR_BGR2RGB) 
    return base_im

def masked_overlay_im_list(base_im, base_alpha, im_list, alpha_list, mask_list, dont_flip_RGB=False):
    # TODO rename return_BGR to keep_RGB_order
    assert len(im_list) == len(alpha_list)
    assert len(im_list) > 0

    base_im = copy.deepcopy(base_im)

    # combine the masks by summation and then convert back to bool
    overlay_mask = np.zeros(mask_list[0].shape[:2], dtype=np.bool_)
    for i in range(len(mask_list)):
        if len(mask_list[i].shape) == 3:
            mask_list[i] = np.any(mask_list[i], axis=2)
        overlay_mask += mask_list[i]
    overlay_mask = overlay_mask > 0
    if len(overlay_mask.shape) == 3:
        overlay_mask = np.any(overlay_mask, axis=2)
    
    overlay_im = np.zeros(base_im.shape).astype(np.float32)
    for i in range(len(im_list)):
        assert im_list[i].shape == base_im.shape
        # overlay_im[overlay_mask] += (alpha_list[i]*im_list[i][overlay_mask])#.astype(np.uint8)
        overlay_im[mask_list[i]] += (alpha_list[i]*im_list[i][mask_list[i]].astype(np.float32))#.astype(np.uint8)
    overlay_im = np.clip(overlay_im, a_min=0., a_max=255.).astype(np.uint8)

    # overlay_mask = overlay_im > 0
    base_im[overlay_mask] = (base_alpha*base_im[overlay_mask].astype(np.float32) + (1-base_alpha)*overlay_im[overlay_mask].astype(np.float32))#.astype(np.uint8)
    base_im = np.clip(base_im, a_min=0., a_max=255.).astype(np.uint8)
    if not dont_flip_RGB:
        base_im = cv2.cvtColor(base_im, cv2.COLOR_BGR2RGB)
    return base_im    

def prob_heatmap_viz(prob_heatmap, cmap, near_zero_tol=1e-3, max_local_prob=1.0):
    # expects heatmap of HxW dimension
    # returns np_array of HxWx3 RGB
    prob_mask = np.array(prob_heatmap > near_zero_tol)
    prob_loc_map_np = np.clip((np.array(prob_heatmap)*(255./max_local_prob)), 0, 255)
    prob_loc_map_np_color = cv2.applyColorMap(prob_loc_map_np.astype(np.uint8), cmap) # colormap expects 0-255 uint8
    prob_im_masked = (prob_loc_map_np_color*prob_mask[..., None]).astype(np.uint8)
    # heatmap_im = cv2.cvtColor(pred_im_masked, cv2.COLOR_BGR2RGB) 
    return prob_im_masked, prob_mask # keep prob mask 2 dim

def prediction_pixel_coordinates_to_image(im_size, pred_pxl_coords_list, cross_length=5, cross_color=[255, 0, 0], output_BGR=False, thickness=1):
    pred_cross_image = np.zeros((im_size[0], im_size[1], 3), dtype=np.uint8)
    cross_color = np.array(cross_color, dtype=np.uint8)
    # Change from "+" to "x"
    for coordinate in pred_pxl_coords_list:
        # draw a circle at the coordinate
        pred_cross_image = cv2.circle(pred_cross_image, (int(coordinate[1]), int(coordinate[0])), radius=cross_length, color=(int(cross_color[0]), int(cross_color[1]), int(cross_color[2])), thickness=thickness)
        # pred_cross_image = cv2.line(pred_cross_image, (int(coordinate[1]-cross_length), int(coordinate[0]-cross_length)), (int(coordinate[1]+cross_length), int(coordinate[0]+cross_length)), color=(int(cross_color[0]), int(cross_color[1]), int(cross_color[2])), thickness=thickness)
        # pred_cross_image = cv2.line(pred_cross_image, (int(coordinate[1]-cross_length), int(coordinate[0]+cross_length)), (int(coordinate[1]+cross_length), int(coordinate[0]-cross_length)), color=(int(cross_color[0]), int(cross_color[1]), int(cross_color[2])), thickness=thickness)
    # get an HxW mask if any of the channels are above 0
    pred_cross_image_mask = np.array(pred_cross_image > 0)
    # take the or of the mask across all channels to get a HxW mask 
    pred_cross_image_mask = np.any(pred_cross_image_mask, axis=2)
    
    if output_BGR:
        return cv2.cvtColor(pred_cross_image, cv2.COLOR_RGB2BGR), pred_cross_image_mask
    else:
        return pred_cross_image, pred_cross_image_mask

def prob_transparency_map_viz(prob_heatmap, color, lower_color, upper_color, max_prob=1., min_prob=0.):
    prob_heatmap_color = (color*np.ones(prob_heatmap.shape + (3,), dtype=np.float32))
    # cut out all prob above max_prob and below min_prob
    prob_mask = np.array((prob_heatmap > min_prob) & (prob_heatmap < max_prob))
    prob_upper_mask = np.array(prob_heatmap > max_prob)
    prob_lower_mask = np.array(prob_heatmap < min_prob)
    # normalize the clipped prob heatmap to 0-1
    prob_heatmap = np.clip((np.clip(prob_heatmap, a_min=min_prob, a_max=max_prob) - min_prob)/(max_prob - min_prob), a_min=0.2, a_max=1.)
    prob_heatmap_color = (prob_heatmap[..., None]*prob_heatmap_color*prob_mask[..., None]).astype(np.uint8)
    prob_heatmap_color[prob_upper_mask] = upper_color
    prob_heatmap_color[prob_lower_mask] = lower_color

    return prob_heatmap_color

def global_prob_viz(global_prob, cmap, im_shape):
    prob_im = np.zeros(im_shape)
    im_width = im_shape[1]
    prob_mask = np.full(im_shape, False)
    prob_im[0:10, :int(im_width*global_prob), ...] = 255  
    prob_mask[0:10, :int(im_width*global_prob), ...] = True
    prob_im = cv2.applyColorMap(prob_im.astype(np.uint8), cmap) # colormap expects 0-255 uint8
    return prob_im, prob_mask

def logarithm_map(prob_heatmap, prob_scaling, min_prob, max_prob):
    prob_heatmap_scaled = prob_heatmap*prob_scaling
    prob_heatmap_log = np.log(prob_heatmap_scaled)
    prob_heatmap_normalized = (prob_heatmap_log - np.log(min_prob*prob_scaling))/(np.log(max_prob*prob_scaling) - np.log(min_prob*prob_scaling))
    prob_heatmap_normalized = np.clip(prob_heatmap_normalized, 0, 1)
    return prob_heatmap_normalized

def inverse_logarithm_map(prob_heatmap_normalized, prob_scaling, min_prob, max_prob):
    prob_heatmap_log = prob_heatmap_normalized*(np.log(max_prob*prob_scaling) - np.log(min_prob*prob_scaling)) + np.log(min_prob*prob_scaling)
    prob_heatmap_scaled = np.exp(prob_heatmap_log)
    prob_heatmap = prob_heatmap_scaled/prob_scaling
    return prob_heatmap

def prob_transparency_map_log_viz(prob_heatmap, prob_scaling, color, lower_color, upper_color, min_prob, max_prob):
    prob_heatmap_color = (color*np.ones(prob_heatmap.shape + (3,), dtype=np.float32))
    # cut out all prob above max_prob and below min_prob
    prob_mask = np.array((prob_heatmap > min_prob) & (prob_heatmap < max_prob))
    prob_upper_mask = np.array(prob_heatmap > max_prob)
    prob_lower_mask = np.array(prob_heatmap < min_prob)
    # prob_heatmap_scaled = prob_heatmap*prob_scaling
    # prob_heatmap_log = np.log(prob_heatmap_scaled)
    # prob_heatmap_normalized = (prob_heatmap_log - np.log(min_prob*prob_scaling))/(np.log(max_prob*prob_scaling) - np.log(min_prob*prob_scaling))
    # prob_heatmap_normalized = np.clip(prob_heatmap_normalized, 0, 1)
    prob_heatmap_normalized = VizUtils.logarithm_map(prob_heatmap, prob_scaling, min_prob, max_prob)
    prob_heatmap_color = (prob_heatmap_normalized[..., None]*prob_heatmap_color*prob_mask[..., None]).astype(np.uint8)
    prob_heatmap_color[prob_upper_mask] = upper_color
    prob_heatmap_color[prob_lower_mask] = lower_color

def prob_colormap_log_viz(prob_heatmap, prob_scaling, colormap, lower_color, upper_color, min_prob, max_prob):
    # prob_heatmap_color = (color*np.ones(prob_heatmap.shape + (3,), dtype=np.float32))
    # cut out all prob above max_prob and below min_prob
    prob_mask = np.array((prob_heatmap > min_prob) & (prob_heatmap < max_prob))
    prob_upper_mask = np.array(prob_heatmap > max_prob)
    prob_lower_mask = np.array(prob_heatmap < min_prob)
    # prob_heatmap_scaled = prob_heatmap*prob_scaling
    # prob_heatmap_log = np.log(prob_heatmap_scaled)
    # prob_heatmap_normalized = (prob_heatmap_log - np.log(min_prob*prob_scaling))/(np.log(max_prob*prob_scaling) - np.log(min_prob*prob_scaling))
    # prob_heatmap_normalized = np.clip(prob_heatmap_normalized, 0, 1)
    prob_heatmap_normalized = VizUtils.logarithm_map(prob_heatmap, prob_scaling, min_prob, max_prob)
    # prob_heatmap_color = cv2.applyColorMap((prob_heatmap_normalized*255).astype(np.uint8), colormap)
    # use matploblib colormap
    prob_heatmap_color = np.array(colormap(prob_heatmap_normalized))[:, :, :3] # removing the alpha channel
    prob_heatmap_color = (prob_heatmap_color*255).astype(np.uint8)
    prob_heatmap_color = (prob_heatmap_normalized[..., None]*prob_heatmap_color*prob_mask[..., None]).astype(np.uint8)
    prob_heatmap_color[prob_upper_mask] = upper_color
    prob_heatmap_color[prob_lower_mask] = lower_color

    return prob_heatmap_color
def bb_x1x2y1y2_to_mask(bb_x1x2y1y2, im_shape):
    bb_mask = np.zeros(im_shape, dtype=np.bool)
    bb_mask[bb_x1x2y1y2[2]:bb_x1x2y1y2[3], bb_x1x2y1y2[0]:bb_x1x2y1y2[1], ...] = True
    return bb_mask

def flow_viz(flow_map, flow_max_norm, min_flow_threshold, return_BGR=True):
    # unnormalize flow map
    # flow_map = flow_map*flow_max_norm
    # construct a flow_magnitude map from the flow
    flow_magnitude = np.linalg.norm(flow_map, axis=-1)
    flow_mask = flow_magnitude > min_flow_threshold
    flow_image = flow_to_image(flow_map, convert_to_bgr=return_BGR, flow_norm=flow_max_norm).astype(np.uint8) # keep bgr to true for cv2 imwrite which expect bgr
    return flow_image, flow_mask

def color_image_from_path(color_path, im_size=None, return_grayscale=False, return_BGR=True):
    color_im = cv2.imread(color_path)
    if im_size is not None:
        color_im = cv2.resize(color_im, (im_size[1], im_size[0]), interpolation=cv2.INTER_CUBIC)
    #TODO if flag is set, convert to grayscale
    if return_grayscale:
        color_im = cv2.cvtColor(color_im, cv2.COLOR_BGR2GRAY)
        # change the shape to H x W x 3
        color_im = np.stack([color_im, color_im, color_im], axis=2)
    # DONT NEED GRAY2RGB CONVERSION AS IMREAD AUTOMATICALLY CONVERTS, UNLESS FLAG IS SET!
    # if is_gray: 
    #     color_im_resize = cv2.cvtColor(color_im_resize, cv2.COLOR_GRAY2BGR) # H x W x 3
    if not return_BGR:
        color_im = cv2.cvtColor(color_im, cv2.COLOR_BGR2RGB)
    image_np_color = np.array(color_im)
    return image_np_color

def save_image(save_path, image_np):
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    # write the image to disk
    cv2.imwrite(save_path, image_np)

def save_pickleable(save_path, obj):
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    with open(save_path, 'wb') as f:
        pickle.dump(obj, f)

def desaturate_color_image(image_np_color, saturation_coefficient=1.0, brightness_coefficient=1.0):
    imghsv = cv2.cvtColor(image_np_color, cv2.COLOR_RGB2HSV).astype("float32")
    (h, s, v) = cv2.split(imghsv)
    s = s * saturation_coefficient
    s = np.clip(s,0,255)
    
    # Add brightness
    # value = brightness_coefficient
    # lim = 255 - value
    # v[v > lim] = 255
    # v[v <= lim] += value

    v = v * brightness_coefficient
    v = np.clip(v,0,255)

    imghsv = cv2.merge([h,s,v])
    image_np_color = cv2.cvtColor(imghsv.astype("uint8"), cv2.COLOR_HSV2RGB)
    return image_np_color

def generate_viz_rectangle(bb_x1x2y1y2_list, im_size):
    # TODO change this to a masked overlay
    if isinstance(bb_x1x2y1y2_list, np.ndarray):
        bb_x1x2y1y2_list = bb_x1x2y1y2_list.tolist()
    [x1, x2, y1, y2] = bb_x1x2y1y2_list
    # make sure to bound the rectangle to the image size
    x1 = max(0, x1)
    x2 = min(im_size[1], x2)
    y1 = max(0, y1)
    y2 = min(im_size[0], y2)

    rectangle_overlay_im = np.zeros((im_size[0], im_size[1], 3))
    if x2 <= 0 or x1 >= im_size[1] or y2 <= 0 or y1 >= im_size[0]:
        pass
    else:
        rectangle_overlay_im = cv2.rectangle(rectangle_overlay_im, (x1, y1), (x2, y2), (255, 255, 255), 1)
    rectangle_overlay_mask = rectangle_overlay_im > 0
    return rectangle_overlay_im, rectangle_overlay_mask

def process_continuous_start_end_times(wrench_time, global_labels):
    start_time = []
    end_time = []
    prev_label = 0
    for i in range(global_labels.shape[0]):
        # strat from 1
        if i == 0 and global_labels[0] == 1:
            start_time.append(wrench_time[0])
        # from 0 to 1, start global
        elif prev_label == 0 and global_labels[i] == 1:
            start_time.append(wrench_time[i])
        # from 1 to 0, end global
        elif prev_label == 1 and global_labels[i] == 0:
            end_time.append(wrench_time[i])
        # end with 1
        elif i == global_labels.shape[0] - 1 and global_labels[i] == 1:
            end_time.append(wrench_time[i])
        prev_label = global_labels[i]
    return start_time, end_time

def gt_contact_pxls_to_im(contact_pxls_flt, cross_length, cross_thickness, cross_color, im_size):
    if isinstance(contact_pxls_flt, torch.Tensor):
        contact_pxls_flt = contact_pxls_flt.cpu().numpy()
    circled_contacts_im = np.full((im_size[0], im_size[1], 3), fill_value=0).astype(np.uint8)
    for pxls in contact_pxls_flt:
        if not np.isnan(pxls).any():
            pxls = pxls.astype(int).tolist()
                # Instead of drawing a circle, draw a cross, the radius is half the length of the cross
                # circled_contacts_im = cv2.circle(circled_contacts_im, tuple(pxls), radius=int(self.circle_dict['radius']), color=self.circle_dict['color'], thickness=self.circle_dict['thickness'])
            half_cross_length = int(cross_length)
            circled_contacts_im = cv2.line(circled_contacts_im, (int(pxls[0]-half_cross_length), int(pxls[1])), (int(pxls[0]+half_cross_length), int(pxls[1])), color=cross_color, thickness=cross_thickness)
            circled_contacts_im = cv2.line(circled_contacts_im, (int(pxls[0]), int(pxls[1]-half_cross_length)), (int(pxls[0]), int(pxls[1]+half_cross_length)), color=cross_color, thickness=cross_thickness)
                # reverse the rgb order
    circled_contacts_im = cv2.cvtColor(circled_contacts_im, cv2.COLOR_BGR2RGB)
    circled_contacts_mask = np.array(circled_contacts_im > 0)
    circled_contacts_mask = np.any(circled_contacts_mask, axis=-1)
    return circled_contacts_im,circled_contacts_mask
def generate_full_depth_overlay_im(depth_im, bb_x1x2y1y2=None):
    im_size = depth_im.shape[:2]
    if bb_x1x2y1y2 is not None:
        bb_im, bb_mask = generate_viz_rectangle(bb_x1x2y1y2, im_size)
        depth_im = masked_overlay_im_list(depth_im, 0.5, [bb_im], [1.0], [bb_mask], dont_flip_RGB=True)
    return depth_im

def generate_full_overlay_im(color_im, pred_contact_pixels, pred_prob_map, 
                             only_show_crop=False, just_for_reference=False,
                             saturation_coefficient=1.0, value_coefficient=1.0,
                             flow_alpha=0.2, pred_probmap_alpha=0.3,
                             gt_alpha=0.5, nms_alpha=0.5, 
                             flow_map=None, 
                             bb_x1x2y1y2=None, gt_contact_pixels=None, 
                             gt_cross_length=4, pred_cross_length=3, 
                             cross_thickness=1, pred_thickness=1, 
                             max_flow_norm=10., flow_threshold = 0.13, 
                             gt_color=(250, 0, 250), pred_cross_color=(0,255,0), 
                             pred_prob_cmap=cv2.COLORMAP_WINTER):
    
    im_size = color_im.shape[:2]

    if not just_for_reference:
        if saturation_coefficient != 1.0 or value_coefficient != 1.0:
            color_im = desaturate_color_image(color_im, saturation_coefficient, value_coefficient)

        pred_prob_map_im, pred_prob_map_mask = prob_heatmap_viz(pred_prob_map, pred_prob_cmap, near_zero_tol=1e-3, max_local_prob=0.05)
        NMS_im, NMS_im_mask = prediction_pixel_coordinates_to_image(im_size, pred_contact_pixels, pred_cross_length, pred_cross_color, output_BGR=True, thickness=pred_thickness)
    if bb_x1x2y1y2 is not None:
        bb_im, bb_mask = generate_viz_rectangle(bb_x1x2y1y2, im_size)
        color_im = masked_overlay_im_list(color_im, 0.5, [bb_im], [1.0], [bb_mask], dont_flip_RGB=True)

    if gt_contact_pixels is not None:
        gt_im, gt_im_mask = gt_contact_pxls_to_im(gt_contact_pixels, gt_cross_length, cross_thickness, gt_color, im_size)

    if not just_for_reference:        
        if flow_map is not None:
            flow_viz_im, flow_mask = flow_viz(flow_map, max_flow_norm, flow_threshold, True)
            if bb_x1x2y1y2 is not None:
                bb_mask = bb_x1x2y1y2_to_mask(bb_x1x2y1y2, im_size)
                # take the logical or of the bb mask and the flow mask
                flow_mask = np.logical_and(flow_mask, bb_mask)

            # overlay the flow onto the color image using the masked overlay method
            color_flow_im = masked_overlay_im_list(color_im, (1-flow_alpha), [flow_viz_im], [1.0], [flow_mask], dont_flip_RGB=True)
            color_flow_prob_im = masked_overlay_im_list(color_flow_im, (1-pred_probmap_alpha), [pred_prob_map_im], [1.0], [pred_prob_map_mask], dont_flip_RGB=True)
        else:
            color_flow_prob_im = masked_overlay_im_list(color_im, (1-pred_probmap_alpha), [pred_prob_map_im], [1.0], [pred_prob_map_mask], dont_flip_RGB=True)
        if gt_contact_pixels is not None:
            full_im = masked_overlay_im_list(color_flow_prob_im, (1-gt_alpha), [gt_im], [1], [gt_im_mask], dont_flip_RGB=True)
            full_im = masked_overlay_im_list(full_im, (1-nms_alpha), [NMS_im], [1], [NMS_im_mask], dont_flip_RGB=True)
        else:
            full_im = masked_overlay_im_list(color_flow_prob_im, (1-nms_alpha), [NMS_im], [1], [NMS_im_mask], dont_flip_RGB=True)

        if only_show_crop:
            # index into the image to only show the crop
            full_im = full_im[bb_x1x2y1y2[2]:bb_x1x2y1y2[3], bb_x1x2y1y2[0]:bb_x1x2y1y2[1], :]
    else:
        full_im = masked_overlay_im_list(color_im, (1-gt_alpha), [gt_im], [1], [gt_im_mask], dont_flip_RGB=True)

    return full_im

class VizUtils():
    def __init__(self, 
                max_depth_clip: float = 2.0,   #### Change colormap from winter to summer
                pred_cmap = cv2.COLORMAP_SUMMER, target_cmap = cv2.COLORMAP_HOT, image_cmap = cv2.COLORMAP_BONE,
                pred_base_color = (0, 0, 255), target_base_color = (255, 0, 0), 
                viz_max_depth_clip = 2.0, viz_min_depth_clip = 0.4, 
                im_resize = (240, 320), real_dataset=False, near_zero_tol=1e-3, circle_dict=None, 
                max_local_prob=1.0, normalize_local_prob=True, forces_ylim=[-15, 15], torques_ylim=[-5, 5],
                convert_color_to_gray=False, max_flow_norm=10.0, color_image_settings_dict=None, **kwargs
    ):
        self.idx = 0
        self.max_depth_clip = max_depth_clip
        self.viz_max_depth_clip = viz_max_depth_clip
        self.viz_min_depth_clip = viz_min_depth_clip
        self.tensor_to_float32 = torchvision.transforms.ConvertImageDtype(torch.float32)
        self.tensor_to_pil = torchvision.transforms.functional.to_pil_image
        self.pred_cmap = pred_cmap
        self.target_cmap = target_cmap
        self.image_cmap = image_cmap
        self.pred_base_color = pred_base_color
        self.target_base_color = target_base_color
        self.real_dataset = real_dataset
        self.near_zero_tol = near_zero_tol
        self.im_resize = im_resize
        self.normalize_local_prob = normalize_local_prob
        self.convert_color_to_gray = convert_color_to_gray
        self.flow_max_norm = max_flow_norm
        self.forces_ylim = forces_ylim
        self.torques_ylim = torques_ylim
        self.color_image_settings_dict = color_image_settings_dict
        if self.normalize_local_prob:
            self.max_local_prob = max_local_prob
        else: 
            self.max_local_prob = 1.0

        if circle_dict is None:
            self.circle_dict = {'enable': False}
        else:
            self.circle_dict = circle_dict
    
    # TODO add the actual inputs to the network to compare
    # TODO add plot titles/legends/labels
    def plot_wrench_history(self, wrench_history, wrench_times, figsize_hw = (180, 320), wrench_pred = None, grasped_obj_mass=None, global_labels=None, save_dir=None): #in width x height 
        # TODO for obj mass plot, actually take in the EE pose and EE_T_objCoM to compute the torques
        figure_height = figsize_hw[0]
        figure_width = figsize_hw[1]
        fontsize = 10
        pred_marker_size = 2
        pred_line_width = 1
        base_line_width = 2

        fig = Figure(figsize=(figure_width/100, figure_height/100), dpi=100)
        canvas = FigureCanvasAgg(fig)
        
        start_time, end_time = [], []
        if global_labels is not None:   
            start_time, end_time = process_continuous_start_end_times(wrench_times, global_labels)
        
        ax = fig.add_subplot()
        ax.plot(wrench_times, wrench_history[:, 0], 'r', wrench_times, wrench_history[:, 0], 'w|', ms=2, lw=base_line_width) # plot the line
        ax.plot(wrench_times, wrench_history[:, 1], 'g', wrench_times, wrench_history[:, 1], 'w|', ms=2, lw=base_line_width) # plot the line
        ax.plot(wrench_times, wrench_history[:, 2], 'b', wrench_times, wrench_history[:, 2], 'w|', ms=2, lw=base_line_width) # plot the line
        if wrench_pred is not None:
            ax.plot(wrench_times, wrench_pred[:, 0], 'r--', ms=pred_marker_size, lw = pred_line_width)
            ax.plot(wrench_times, wrench_pred[:, 1], 'g--', ms=pred_marker_size, lw = pred_line_width)
            ax.plot(wrench_times, wrench_pred[:, 2], 'b--', ms=pred_marker_size, lw = pred_line_width)
        if grasped_obj_mass is not None:
            obj_z_force = grasped_obj_mass * 9.81
            ax.plot(wrench_times, obj_z_force * np.ones_like(wrench_times), 'k--', ms=pred_marker_size, lw = pred_line_width)
        
        if len(start_time)!=0 or len(end_time)!=0:
            ax = self.draw_global_contact(ax, wrench_times, start_time, end_time)
        
        ax.grid(visible=True)
        ax.set_ylim(self.forces_ylim)
        ax.set_title('Forces (RGB = XYZ)', fontsize=fontsize)
        ax.set_xlabel('Time (s)', fontsize=fontsize)
        ax.set_ylabel('Force (N)', fontsize=fontsize)
        ax.tick_params(axis='both', which='major', labelsize=8)
        canvas.draw()
        forces_plot = np.array(canvas.buffer_rgba())[..., :3]
        
        ax.cla()
        ax.plot(wrench_times, wrench_history[:, 3], 'r', wrench_times, wrench_history[:, 3], 'w|', ms=2, lw=2) # plot the line
        ax.plot(wrench_times, wrench_history[:, 4], 'g', wrench_times, wrench_history[:, 4], 'w|', ms=2, lw=2) # plot the line
        ax.plot(wrench_times, wrench_history[:, 5], 'b', wrench_times, wrench_history[:, 5], 'w|', ms=2, lw=2) # plot the line
        if wrench_pred is not None:
            ax.plot(wrench_times, wrench_pred[:, 3], 'r--', ms=pred_marker_size, lw=pred_line_width)
            ax.plot(wrench_times, wrench_pred[:, 4], 'g--', ms=pred_marker_size, lw=pred_line_width)
            ax.plot(wrench_times, wrench_pred[:, 5], 'b--', ms=pred_marker_size, lw=pred_line_width)

        if len(start_time)!=0 or len(end_time)!=0:
            ax = self.draw_global_contact(ax, wrench_times, start_time, end_time)

        ax.grid(visible=True)
        ax.set_ylim(self.torques_ylim)
        ax.set_title('Torques (RGB = XYZ)', fontsize=fontsize)
        ax.set_xlabel('Time (s)', fontsize=fontsize)
        ax.set_ylabel('Torque (Nm)', fontsize=fontsize)
        ax.tick_params(axis='both', which='major', labelsize=8)
        canvas.draw()
        torques_plot = np.array(canvas.buffer_rgba())[..., :3]
        # torques_plot = Image.fromarray(rgba).convert('RGB')

        if save_dir is not None:
            self.save_image(os.path.join(save_dir, 'external_EE_force_plots', str(self.idx) + '.png'), forces_plot)
            self.save_image(os.path.join(save_dir, 'external_EE_torque_plots', str(self.idx) + '.png'), torques_plot)
        
        wrench_plots_dict = {
            'forces_plot': forces_plot,
            'torques_plot': torques_plot,
        }

        # plot the difference between the actual and predicted forces and torques
        if wrench_pred is not None:
            # difference of the actual and predicted forces
            force_difference = wrench_history[:, :3] - wrench_pred[:, :3]
            ax.cla()
            ax.plot(wrench_times, force_difference[:, 0], 'r', wrench_times, force_difference[:, 0], 'w|', ms=2, lw=2) # plot the line
            ax.plot(wrench_times, force_difference[:, 1], 'g', wrench_times, force_difference[:, 1], 'w|', ms=2, lw=2) # plot the line
            ax.plot(wrench_times, force_difference[:, 2], 'b', wrench_times, force_difference[:, 2], 'w|', ms=2, lw=2) # plot the line
            ax.grid(visible=True)
            ax.set_ylim(self.forces_ylim)
            ax.set_title('Force Difference (RGB = XYZ)', fontsize=fontsize)
            ax.set_xlabel('Time (s)', fontsize=fontsize)
            ax.set_ylabel('Force (N)', fontsize=fontsize)
            ax.tick_params(axis='both', which='major', labelsize=8)
            canvas.draw()
            force_difference_plot = np.array(canvas.buffer_rgba())[..., :3]

            # difference of the actual and predicted torques
            torque_difference = wrench_history[:, 3:] - wrench_pred[:, 3:]
            ax.cla()
            ax.plot(wrench_times, torque_difference[:, 0], 'r', wrench_times, torque_difference[:, 0], 'w|', ms=2, lw=2) # plot the line
            ax.plot(wrench_times, torque_difference[:, 1], 'g', wrench_times, torque_difference[:, 1], 'w|', ms=2, lw=2) # plot the line
            ax.plot(wrench_times, torque_difference[:, 2], 'b', wrench_times, torque_difference[:, 2], 'w|', ms=2, lw=2) # plot the line
            ax.grid(visible=True)
            ax.set_ylim(self.torques_ylim)
            ax.set_title('Torque Difference (RGB = XYZ)', fontsize=fontsize)
            ax.set_xlabel('Time (s)', fontsize=fontsize)
            ax.set_ylabel('Torque (Nm)', fontsize=fontsize)
            ax.tick_params(axis='both', which='major', labelsize=8)
            canvas.draw()
            torque_difference_plot = np.array(canvas.buffer_rgba())[..., :3]
            wrench_plots_dict.update(
            {'forces_difference_plot': force_difference_plot,
            'torques_difference_plot': torque_difference_plot,}
            )
        return wrench_plots_dict

    def draw_heatmaps(self, depth_tensor, 
                      pred_tensor=None, cond_pred_tensor=None, global_pred_prob=None, 
                      target_tensor=None, 
                      color_path=None, 
                      contact_pxls_flt=None,  
                      flow_image=None, bb_x1x2y1y2_list=None, 
                      context_depth_frame=None, context_color_path=None, 
                      context_bb_x1x2y1y2_list=None,
                      NMS_pred_pxl_coord_list=None, cross_length=3, save_dir=None, 
                      preprocessing_dict = None
                      ):
        # backward compatibility
        if preprocessing_dict is not None:
            self.color_image_settings_dict = preprocessing_dict

        # accepts tensors of HxW dim which should be put on cpu
        # pred_tensor should be prob, not logit
        heatmaps_dict = {}
        im_size = depth_tensor.shape # H x W 

        if not pred_tensor is None:
            pred_prob_heatmap_np = pred_tensor.numpy() # index batch and channel
            pred_im, pred_mask = prob_heatmap_viz(pred_prob_heatmap_np, self.pred_cmap, self.near_zero_tol, self.max_local_prob)
            if save_dir is not None:
                # self.save_image(os.path.join(save_dir, 'pred_heatmap_images', str(self.idx) + '.png'), pred_im)
                save_pickleable(os.path.join(save_dir, 'pred_prob_np', str(self.idx) + '.pkl'), pred_prob_heatmap_np)
            # pred_mask = np.array(pred_loc_map_tensor > self.near_zero_tol)
            # pred_loc_map_np = np.clip((np.array(pred_loc_map_tensor)*(255./self.max_local_prob)), 0, 255)
            if not global_pred_prob is None:
                global_pred_im, global_pred_mask = global_prob_viz(global_pred_prob, self.pred_cmap, im_size)
                heatmaps_dict['global_pred_im'] = cv2.cvtColor(global_pred_im, cv2.COLOR_BGR2RGB)
                pred_im = masked_overlay_im_list(pred_im, 0.0, [global_pred_im], [1.], [global_pred_mask], dont_flip_RGB=True)
            if not NMS_pred_pxl_coord_list is None:
                pred_cross_image, pred_cross_image_mask = prediction_pixel_coordinates_to_image(im_size, NMS_pred_pxl_coord_list, cross_length=cross_length, output_BGR=True)
                heatmaps_dict['pred_NMS_im'] = pred_cross_image
                if save_dir is not None:
                    # save the NMS_pred_pxl_coord_list as pickle
                    save_pickleable(os.path.join(save_dir, 'NMS_pred_pxl_coord_list', str(self.idx) + '.pkl'), NMS_pred_pxl_coord_list)
                    # self.save_image(os.path.join(save_dir, 'pred_NMS_images', str(self.idx) + '.png'), pred_cross_image)

                heatmaps_dict['pred_im'] = cv2.cvtColor(masked_overlay_im_list(pred_im, 0.0, [pred_cross_image], [1.0], [pred_cross_image_mask], dont_flip_RGB=True), cv2.COLOR_BGR2RGB)
                # update the pred_mask
                # pred_mask = np.logical_or(pred_mask, pred_cross_image_mask)
                # if save_dir is not None:
                #     self.save_image(os.path.join(save_dir, 'pred_heatmap_NMS_overlay_images', str(self.idx) + '.png'), pred_im)

        if not cond_pred_tensor is None:
            cond_pred_loc_map_tensor = cond_pred_tensor.numpy() # index batch and channel
            cond_pred_im, cond_pred_mask = prob_heatmap_viz(cond_pred_loc_map_tensor, self.cond_pred_cmap, self.near_zero_tol, self.max_local_prob)
            heatmaps_dict['cond_pred_im'] = cv2.cvtColor(cond_pred_im, cv2.COLOR_BGR2RGB)

        elif global_pred_prob is not None:
            global_pred_prob = global_pred_prob.numpy() 
            # global_pred_im, global_pred_mask = self.global_prob_viz(global_pred_prob, self.pred_cmap, im_size)
            pred_im, pred_mask = global_prob_viz(global_pred_prob, self.pred_cmap, im_size)
            heatmaps_dict['pred_im'] = cv2.cvtColor(pred_im, cv2.COLOR_BGR2RGB)
            
        # if not self.real_dataset:
        if not target_tensor is None:
            target_im, target_mask = prob_heatmap_viz(target_tensor, self.target_cmap, max_local_prob=1)
            if save_dir is not None:
                # save the contact_pxls_flt as pickle
                save_pickleable(os.path.join(save_dir, 'contact_pxls_flt', str(self.idx) + '.pkl'), contact_pxls_flt)
            #     self.save_image(os.path.join(save_dir, 'target_heatmap_images', str(self.idx) + '.png'), target_im)
            circled_contacts_im, circled_contacts_mask = self.gt_contact_pxls_to_im(contact_pxls_flt, im_size)
            # circled_contacts_im = cv2.cvtColor(circled_contacts_im, cv2.COLOR_RGB2BGR) 
        if not color_path is None:
            image_np_color = color_image_from_path(color_path, im_size=im_size, return_grayscale=self.convert_color_to_gray, return_BGR=True)

            if not self.color_image_settings_dict is None and not self.convert_color_to_gray:
                # Please pass in a dict as follows to draw_heatmap function to active preprocessing here:
                # preprocessing_dict = {'saturation_coefficient': 0.3 (lower, less color), 'brightness_coefficient': 30 (lower, darker)}
                image_np_color = desaturate_color_image(image_np_color)
        
        if not context_color_path is None:
            image_np_context_color = color_image_from_path(context_color_path, im_size=im_size, return_grayscale=self.convert_color_to_gray, return_BGR=False)
            if context_bb_x1x2y1y2_list is not None:
                context_rectangle_overlay_im, context_rectangle_overlay_mask = generate_viz_rectangle(context_bb_x1x2y1y2_list, im_size)
                image_np_context_color = masked_overlay_im_list(image_np_context_color, 0.4, [context_rectangle_overlay_im], [1.], [context_rectangle_overlay_mask], dont_flip_RGB=True)
            heatmaps_dict['context_color_im'] = image_np_context_color
        
        image_np_depth = self.tensor_to_depth_im(depth_tensor, self.image_cmap, return_BGR=True)
        if not context_depth_frame is None:
            context_depth_im = self.tensor_to_depth_im(context_depth_frame, self.image_cmap, return_BGR=False)
            if context_bb_x1x2y1y2_list is not None:
                # ASSUME THE BB HAS ALREADY BEEN GENERATED 
                context_depth_im = masked_overlay_im_list(context_depth_im, 0.4, [context_rectangle_overlay_im], [1.], [context_rectangle_overlay_mask], dont_flip_RGB=True)
            heatmaps_dict['context_depth_im'] = context_depth_im

        if not flow_image is None:
            flow_image = cv2.cvtColor(flow_image.astype(np.uint8), cv2.COLOR_RGB2BGR)
            # image_np_depth = self.overlay_im_list([image_np_depth, flow_image], [0.6, 0.4], return_BGR=True)
            image_np_color_flow = self.overlay_im_list([image_np_color, flow_image], [0.5, 0.5], return_BGR=True)

        if not bb_x1x2y1y2_list is None:
            # TODO change this to a masked overlay
            rectangle_overlay_im, rectangle_overlay_mask = generate_viz_rectangle(bb_x1x2y1y2_list, im_size)
            if save_dir is not None:
                # save the bb_x1x2y1y2_list as pickle
                save_pickleable(os.path.join(save_dir, 'bb_x1x2y1y2_list', str(self.idx) + '.pkl'), bb_x1x2y1y2_list)
                # self.save_image(os.path.join(save_dir, 'crop_bb_images', str(self.idx) + '.png'), rectangle_overlay_im)
        # if self.real_dataset:
        if not pred_tensor is None or not global_pred_prob is None:
            heatmaps_dict['pred_depth_overlay_im'] = masked_overlay_im_list(image_np_depth, 0.5, [pred_im], [1.0], [pred_mask])
            heatmaps_dict['pred_color_overlay_im'] = masked_overlay_im_list(image_np_color, 0.5, [pred_im], [1.0], [pred_mask])
            # pred_depth_overlay_im = masked_overlay_im_list(image_np_depth, 0.5, [pred_im], [1.0], [pred_mask], dont_flip_RGB=True)
            # pred_color_overlay_im = masked_overlay_im_list(image_np_color, 0.5, [pred_im], [1.0], [pred_mask], dont_flip_RGB=True)
            
            pred_depth_overlay_im = cv2.cvtColor(heatmaps_dict['pred_depth_overlay_im'], cv2.COLOR_RGB2BGR)
            pred_color_overlay_im = cv2.cvtColor(heatmaps_dict['pred_color_overlay_im'], cv2.COLOR_RGB2BGR)
            
            if not bb_x1x2y1y2_list is None:
                heatmaps_dict['pred_color_overlay_im'] = masked_overlay_im_list(heatmaps_dict['pred_color_overlay_im'], 0.2, [rectangle_overlay_im], [1.0], [rectangle_overlay_mask], dont_flip_RGB=True)
                heatmaps_dict['pred_depth_overlay_im'] = masked_overlay_im_list(heatmaps_dict['pred_depth_overlay_im'], 0.2, [rectangle_overlay_im], [1.0], [rectangle_overlay_mask], dont_flip_RGB=True)
            
        if target_tensor is None:
            if save_dir is not None:
                self.save_image(os.path.join(save_dir, 'pred_color_overlay_images', str(self.idx) + '.png'), cv2.cvtColor(heatmaps_dict['pred_color_overlay_im'], cv2.COLOR_RGB2BGR))
                self.save_image(os.path.join(save_dir, 'pred_depth_overlay_images', str(self.idx) + '.png'), cv2.cvtColor(heatmaps_dict['pred_depth_overlay_im'], cv2.COLOR_RGB2BGR))
                                  
            if not cond_pred_tensor is None:
                heatmaps_dict['cond_pred_depth_overlay_im'] = masked_overlay_im_list(image_np_depth, 0.4, [cond_pred_im], [1.0], [cond_pred_mask])
                heatmaps_dict['cond_pred_color_overlay_im'] = masked_overlay_im_list(image_np_color, 0.4, [cond_pred_im], [1.0], [cond_pred_mask])
                if not bb_x1x2y1y2_list is None:
                    heatmaps_dict['cond_pred_color_overlay_im'] = masked_overlay_im_list(pred_depth_overlay_im, 0.5, [rectangle_overlay_im], [1.0], [rectangle_overlay_mask], dont_flip_RGB=True)
                    heatmaps_dict['cond_pred_depth_overlay_im'] = masked_overlay_im_list(pred_color_overlay_im, 0.5, [rectangle_overlay_im], [1.0], [rectangle_overlay_mask], dont_flip_RGB=True)

        # if not self.real_dataset:
        if target_tensor is not None:
            # H x W x 3
            if not pred_tensor is None or not global_pred_prob is None:
                pass
            # if self.circle_dict['enable']:
            if not pred_tensor is None or not global_pred_prob is None:
                # heatmaps_dict['circled_target_pred_depth_overlay_im'] = masked_overlay_im_list(image_np_depth, 0.4, [pred_im, target_im, circled_contacts_im], [0.6, 0.4, 1.0], [pred_mask, target_mask, circled_contacts_mask])
                # heatmaps_dict['circled_target_pred_color_overlay_im'] = masked_overlay_im_list(image_np_color, 0.4, [pred_im, target_im, circled_contacts_im], [0.6, 0.4, 1.0], [pred_mask, target_mask, circled_contacts_mask])
                # do the masked overlay of the pred_im over color and depth first
                heatmaps_dict['circled_target_pred_depth_overlay_im'] = masked_overlay_im_list(pred_depth_overlay_im, 0.3, [pred_cross_image, circled_contacts_im], [1.0, 1.0], [pred_cross_image_mask, circled_contacts_mask])
                heatmaps_dict['circled_target_pred_color_overlay_im'] = masked_overlay_im_list(pred_color_overlay_im, 0.3, [pred_cross_image, circled_contacts_im], [1.0, 1.0], [pred_cross_image_mask, circled_contacts_mask])

            if not cond_pred_tensor is None:
                heatmaps_dict['circled_target_pred_depth_overlay_im'] = masked_overlay_im_list(image_np_depth, 0.4, [cond_pred_im, target_im, circled_contacts_im], [0.6, 0.4, 1.0], [cond_pred_mask, target_mask, circled_contacts_mask])
                heatmaps_dict['circled_target_pred_color_overlay_im'] = masked_overlay_im_list(image_np_color, 0.4, [cond_pred_im, target_im, circled_contacts_im], [0.6, 0.4, 1.0], [cond_pred_mask, target_mask, circled_contacts_mask])
            
            if not bb_x1x2y1y2_list is None:
                heatmaps_dict['circled_target_pred_color_overlay_im'] = masked_overlay_im_list(heatmaps_dict['circled_target_pred_color_overlay_im'], 0.4, [rectangle_overlay_im], [1.0], [rectangle_overlay_mask], dont_flip_RGB=True)
                heatmaps_dict['circled_target_pred_depth_overlay_im'] = masked_overlay_im_list(heatmaps_dict['circled_target_pred_depth_overlay_im'], 0.4, [rectangle_overlay_im], [1.0], [rectangle_overlay_mask], dont_flip_RGB=True)
            
            if save_dir is not None:
                self.save_image(os.path.join(save_dir, 'circled_target_pred_color_overlay_images', str(self.idx) + '.png'), cv2.cvtColor(heatmaps_dict['circled_target_pred_color_overlay_im'], cv2.COLOR_RGB2BGR))
                self.save_image(os.path.join(save_dir, 'circled_target_pred_depth_overlay_images', str(self.idx) + '.png'), cv2.cvtColor(heatmaps_dict['circled_target_pred_depth_overlay_im'], cv2.COLOR_RGB2BGR))
            else:
                raise NotImplementedError
        return heatmaps_dict
    
    def draw_global_contact(self, ax, wrench_times, start_times, end_times):
        # len of start times and end times can only be off by one!
        if len(start_times) == len(end_times):
            for i in range(len(start_times)):
                ax.axvspan(start_times[i], end_times[i], alpha=0.5, color='pink')
        elif len(start_times) == len(end_times) - 1:
            for i in range(len(start_times)):
                if i == len(end_times):
                    ax.axvspan(start_times[i], wrench_times[0], alpha=0.5, color='pink')
                else:
                    ax.axvspan(start_times[i], end_times[i], alpha=0.5, color='pink')
        elif len(end_times) == len(start_times) - 1: # this means that the start time is out of the range of the wrench history so we can use wrench_times[-1] for start time
            for i in range(len(end_times)):
                if i == len(start_times):
                    ax.axvspan(wrench_times[-1], end_times[i], alpha=0.5, color='pink')
                else:
                    ax.axvspan(start_times[i], end_times[i], alpha=0.5, color='pink')
        else:
            raise ValueError('start times and end times are not of the same length or off by one!')
        return ax

    def viz_stn(self, model, im_tensor):
        with torch.no_grad():
            tfed_im_tensor = model.stn(im_tensor[None, None, ...]).cpu() # fake the batch and channel dims
        return self.tensor_to_depth_im(tfed_im_tensor[0, 0, ...], self.image_cmap, return_BGR=False) # remove the fake
    
