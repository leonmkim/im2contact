import torch 
import torchmetrics

# compute root sum of squared errors (RSSE) and then average over the batches
class MRSSE(torchmetrics.Metric):
    is_differentiable = True
    higher_is_better = False
    full_state_update = False
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("sum_squared_error", default=torch.tensor(()), dist_reduce_fx="cat")

    def update(self, preds, target):
        # reduce across vector dimension
        sum_squared_error = torch.sum((preds - target)**2, dim=1)
        self.sum_squared_error = torch.cat((self.sum_squared_error, sum_squared_error), dim=0)

    def compute(self):
        return torch.mean(torch.sqrt(self.sum_squared_error))

class MSSE(torchmetrics.Metric):
    is_differentiable = True
    higher_is_better = False
    full_state_update = False
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("sum_squared_error", default=torch.tensor(()), dist_reduce_fx="cat")

    def update(self, preds, target):
        sum_squared_error = torch.sum((preds - target)**2, dim=1)
        self.sum_squared_error = torch.cat((self.sum_squared_error, sum_squared_error), dim=0)

    def compute(self):
        return torch.mean(self.sum_squared_error)