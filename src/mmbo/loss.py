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

    def forward(self, mean_est, sd_est, target):
        if input.shape()[self._splitdim] != 2:
            raise Exception("Expecting length 2 in dimension {}".format(self._splitdim))

        if self._ignorenan:
            mask_mean_est = torch.logical_not(torch.isnan(mean_est))
            mask_sd_est = torch.logcial_not(torch.isnan(sd_est))
            mask_target = torch.logical_not(torch.isnan(target))
            mask = torch.logical_and(
                mask_target,
                torch.logical_and(
                    mask_mean_est,
                    mask_sd_est
                )
            )
            mean_est_masked = mean_est[mask]
            sd_est_masked = sd_est[mask]
            target_masked = target[mask]
        else:
            mean_est_masked, sd_est_masked, target_masked = mean_est, sd_est, target
        
        return ((mean_est_masked - target_masked) / sd_est_masked).pow(2).sum() / 2 + sd_est_masked.log().sum()


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
    
    def forward(self, model, mean_est, sd_est, target):
        return self._prior_kl(model) + self._likelihood(mean_est, sd_est, target)