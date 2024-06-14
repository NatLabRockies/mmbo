import torch
import torchbnn as bnn
from torch import nn
import torch.nn.init as init

class NaNGaussianLoss(nn.Module):

    def __init__(self, logsigma = None):
        super(NaNGaussianLoss, self).__init__()
        self._logsigma = logsigma

    def forward(self, input, target, logsigma = None):
        if logsigma is None:
            if self._logsigma is None:
                raise Exception("Must supply logsigma")
            else:
                logsigma = self._logsigma

        # For some reason torch.nanmean and torch.nansum don't work here. Reproduce solution from:
        # https://discuss.pytorch.org/t/why-all-my-parameters-are-nan-after-using-torch-nanmean-in-my-loss-function/151837
        mask_input = torch.logical_not(torch.isnan(input))
        mask_target = torch.logical_not(torch.isnan(target))
        mask = torch.logical_and(mask_input, mask_target)
        
        n = mask.sum()
        input_masked = input[mask]
        target_masked = target[mask]

        return (input_masked - target_masked).pow(2).sum() / (2 * torch.exp(2 * logsigma)) + n * logsigma


class BayesianLoss(nn.Module):

    def __init__(self):
        super(BayesianLoss, self).__init__()
        self._likelihood = NaNGaussianLoss()
        self._prior_kl = bnn.BKLLoss(reduction="sum")
        self.logsigma = nn.Parameter(torch.Tensor(1))
    
    def forward(self, model, input, target):
        return self._prior_kl(model) + self._likelihood(input, target, self.logsigma)