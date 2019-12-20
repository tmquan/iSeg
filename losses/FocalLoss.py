# Using torch
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma
        
    def forward(self, inputs, target):
        if not (target.size() == inputs.size()):
            raise ValueError("Target size ({}) must be the same as inputs size ({})"
                             .format(target.size(), inputs.size()))
 
        max_val = (-inputs).clamp(min=0)
        loss = inputs - inputs * target + max_val + \
            ((-max_val).exp() + (-inputs - max_val).exp()).log()
 
        invprobs = F.logsigmoid(-inputs * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss
        
        return loss.sum(dim=1).mean()