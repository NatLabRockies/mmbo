import torch
import torchbnn as bnn
from torch import nn
import torch.nn.init as init

class GaussianLoss(nn.Module):

    def __init__(self, logsigma = None, ignorenan=True):
        super(GaussianLoss, self).__init__()
        self._logsigma = logsigma
        self._ignorenan = ignorenan

    def forward(self, input, target, logsigma = None):
        if logsigma is None:
            if self._logsigma is None:
                raise Exception("Must supply logsigma")
            else:
                logsigma = self._logsigma

        if self._ignorenan:
            # For some reason torch.nanmean and torch.nansum don't work here. Reproduce solution from:
            # https://discuss.pytorch.org/t/why-all-my-parameters-are-nan-after-using-torch-nanmean-in-my-loss-function/151837
            mask_input = torch.logical_not(torch.isnan(input))
            mask_target = torch.logical_not(torch.isnan(target))
            mask = torch.logical_and(mask_input, mask_target)
            n = mask.sum()
            input_masked = input[mask]
            target_masked = target[mask]
        else:
            n = torch.ones_like(target).sum()
            input_masked, target_masked = input, target
        

        return (input_masked - target_masked).pow(2).sum() / (2 * torch.exp(2 * logsigma)) + n * logsigma
    
class NetworkGaussianLoss(nn.Module):

    def __init__(self, splitdim = -1, ignorenan = True):
        super(NetworkGaussianLoss, self).__init__()
        self._ignorenan = ignorenan
        self._splitdim = splitdim

    def forward(self, input, target):
        if input.shape()[self._splitdim] != 2:
            raise Exception("Expecting length 2 in dimension {}".format(self._splitdim))
        
        input_mean, input_sd = torch.chunk(input, 2, dim = self._splitdim)
        input_mean = input_mean.squeeze(dim = self._splitdim)
        input_sd = input_sd.squeeze(dim = self._splitdim)

        if self._ignorenan:
            mask_input_mean = torch.logical_not(torch.isnan(input_mean))
            mask_input_sd = torch.logcial_not(torch.isnan(input_sd))
            mask_target = torch.logical_not(torch.isnan(target))
            mask = torch.logical_and(
                mask_target,
                torch.logical_and(
                    mask_input_mean,
                    mask_input_sd
                )
            )
            input_mean_masked = input_mean[mask]
            input_sd_masked = input_sd[mask]
            target_masked = target[mask]
        else:
            input_mean_masked, input_sd_masked, target_masked = input_mean, input_sd, target
        
        return ((input_mean_masked - target_masked) / input_sd_masked).pow(2).sum() / 2 + input_sd_masked.log().sum()


class BayesianLoss(nn.Module):

    def __init__(self):
        super(BayesianLoss, self).__init__()
        self._likelihood = GaussianLoss()
        self._prior_kl = bnn.BKLLoss(reduction="sum")
        self.logsigma = nn.Parameter(torch.Tensor(1))
    
    def forward(self, model, input, target):
        return self._prior_kl(model) + self._likelihood(input, target, self.logsigma)
    
class NetworkBayesianLoss(nn.Module):

    def __init__(self):
        super(NetworkBayesianLoss, self).__init__()
        self._likelihood = NetworkGaussianLoss()
        self._prior_kl = bnn.BKLLoss(reduction="sum")
    
    def forward(self, model, input, target):
        return self._prior_kl(model) + self._likelihood(input, target)