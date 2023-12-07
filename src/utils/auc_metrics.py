from torchmetrics import MetricCollection
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision
import torch
class AUCMetrics():
    def __init__(self, crop_shift, crop_size) -> None:
        self.crop_shift = torch.tensor(crop_shift)
        self.crop_size = crop_size

    def sample_pixels_from_image(self, prediction_batch, target_batch, ee_pxls):
        # expect images in B,H,W format
        assert prediction_batch.shape[0] == target_batch.shape[0], "prediction and target batch must have the same batch size"
        assert prediction_batch.shape[1] == target_batch.shape[1], "prediction and target batch must have the same height"
        assert prediction_batch.shape[2] == target_batch.shape[2], "prediction and target batch must have the same width"

        for idx in range(prediction_batch.shape[0]):
            pred = prediction_batch[idx, ...]
            target = target_batch[idx, ...]
            img_height = pred.shape[0]
            img_width = pred.shape[1]
            center_pxl = ee_pxls[idx, ...]
            center_pxl = center_pxl + self.crop_shift
            # first ensure that the crop is within the image using clamp
            bottom = torch.clamp(center_pxl[1] - self.crop_size[0]//2, min=0)
            top = torch.clamp(center_pxl[1] + self.crop_size[0]//2, max=img_height)
            left = torch.clamp(center_pxl[0] - self.crop_size[1]//2, min=0)
            right = torch.clamp(center_pxl[0] + self.crop_size[1]//2, max=img_width)
            # then crop the image
            crop_pred = pred[bottom:top, left:right]
            crop_target = target[bottom:top, left:right]
            # concat along the batch dimension and preserve the batch dimension
            if idx == 0:
                crop_batch_pred = crop_pred.unsqueeze(0)
                crop_batch_target = crop_target.unsqueeze(0)
            else:
                crop_batch_pred = torch.cat((crop_batch_pred, crop_pred.unsqueeze(0)), dim=0)
                crop_batch_target = torch.cat((crop_batch_target, crop_target.unsqueeze(0)), dim=0)
        return crop_batch_pred, crop_batch_target
