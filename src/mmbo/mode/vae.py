import torch
from torch import nn
import pyro
import pyro.distributions as dist
import pyro.distributions.constraints as constraints
from pyro.nn import PyroModule, PyroParam
from pyro.nn.module import pyro_method

__all__ = ["VAE"]

###############################################################################
## Variational Autoencoder
###############################################################################

class Residual(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.module(inputs) + inputs

class MLP(nn.Sequential):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int, depth: int = 1):
        super().__init__()
        # Add in layers
        self.append(nn.Linear(in_dim, hidden_dim))
        self.append(nn.ReLU())
        for ll in range(depth - 1):
            self.append(nn.Linear(hidden_dim, hidden_dim))
            self.append(nn.ReLU())
        self.append(nn.Linear(hidden_dim, out_dim))

class VAE(PyroModule):
    def __init__(self, data_dim: int, z_dim: int, hidden_dim: int, depth: int = 1):
        super().__init__()
        # create the encoder and decoder networks
        self.mlp_encoder = MLP(data_dim, z_dim * 2, hidden_dim, depth) # Output mean/sd pairs
        self.mlp_decoder = MLP(z_dim, data_dim, hidden_dim, depth)
        self.error_scale_loc = PyroParam(torch.zeros(data_dim))
        self.error_scale_scale = PyroParam(torch.ones(data_dim), constraint=constraints.positive)
        self.z_dim = z_dim
        self.data_dim = data_dim
        self.register_buffer("z_prior_loc", torch.zeros(z_dim))
        self.register_buffer("z_prior_scale", torch.ones(z_dim))
        self.register_buffer("error_scale_prior_scale", torch.ones(data_dim))

    # define the model p(x|z)p(z)
    @pyro_method
    def model(self, x: torch.Tensor):
        assert x.ndim == 2
        n_data = x.shape[0]
        # Sample from prior of error scale
        error_scale = pyro.sample("error_scale", dist.HalfCauchy(self.error_scale_prior_scale).to_event(1))
        with pyro.plate("data", n_data):
            # sample from prior (value will be sampled by guide when computing the ELBO)
            z = pyro.sample("latent", dist.Normal(self.z_prior_loc, self.z_prior_scale).to_event(1).expand((n_data,)))
            # score against observations
            pyro.sample("obs", dist.Normal(self.decode(z), error_scale).to_event(1), obs = x)

    # define the guide (i.e. variational distribution) q(z|x)
    @pyro_method
    def guide(self, x: torch.Tensor):
        # use the encoder to get the parameters used to define q(z|x)
        assert x.ndim == 2
        n_data = x.shape[0]
        z_loc, z_scale = self.encode(x)
        error_scale = pyro.sample("error_scale", dist.LogNormal(self.error_scale_loc, self.error_scale_scale).to_event(1))
        with pyro.plate("data", n_data):
            # sample the latent code z
            pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))

    def generate(self, nsample):
        """Generate 'prior' samples according to the learned decoder."""
        return self.decode(dist.Normal(self.z_prior_loc, self.z_prior_scale).sample((nsample,)))

    def encode(self, x):
        z_loc, z_scale_raw = torch.chunk(self.mlp_encoder(x), chunks = 2, dim=-1)
        return z_loc, torch.nn.functional.softplus(z_scale_raw)
    
    def decode(self, x):
        return self.mlp_decoder(x)