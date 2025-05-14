from abc import ABC

import torch
import torch.nn as nn

class FocalLoss(nn.Module, ABC):
    def __init__(self, alpha=2, beta=4):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, prediction, target, mask=None):
        if mask is not None:
            if mask.sum() == 0:
                return torch.tensor(0.0, requires_grad=True, device=prediction.device)
            prediction = prediction[mask!=0]
            target = target[mask!=0]
        positive_index = target.eq(1).float()
        negative_index = target.lt(1).float()
        negative_weights = torch.pow(1 - target, self.beta)
        prediction = torch.clamp(prediction, 1e-12, 0.9999999)
        positive_loss = torch.log(prediction) * torch.pow(1 - prediction, self.alpha) * positive_index
        negative_loss = torch.log(1 - prediction) * torch.pow(prediction,
                                                              self.alpha) * negative_weights * negative_index
        
        num_positive = positive_index.float().sum()
        positive_loss = positive_loss.sum()
        negative_loss = negative_loss.sum()

        if num_positive == 0:
            loss = -negative_loss
        else:
            loss = -(positive_loss + negative_loss) / num_positive
        return loss