import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_from_disk
from collections import namedtuple
import random
Rawdata = namedtuple('Rawdata', ['x', 'y'])
Prefdata = namedtuple('Prefdata', ['x', 'y', 'pref', 'full_pref'])
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import os
import torch.distributed as dist
import gpytorch
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.models import ExactGP
from gpytorch.distributions import MultivariateNormal
from torch.distributions import MultivariateNormal as MultivariateNormalTorch

def scalar_collate_fn(batch):
    x, y = zip(*batch)
    x = torch.stack(x, dim=0)
    y = torch.stack(y, dim=0)
    return Rawdata(x, y)

class GaussianProcessConstant(gpytorch.models.ExactGP):
    def __init__(self, x, y, likelihood, mean_constant=0.0, lengthscale=1.0, outputscale=1.0, noise=0.1):
        super().__init__(x, y, likelihood)

        # Initialize mean module
        self.mean_module = ConstantMean()
        self.mean_module.constant = mean_constant
        self.covar_module = ScaleKernel(
            RBFKernel()
        )
        self.covar_module.base_kernel.lengthscale = lengthscale
        self.covar_module.outputscale = outputscale
        # Set noise
        self.likelihood = likelihood
        self.likelihood.noise_covar.noise = noise


    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

class GPSamplerConstantDataset(Dataset):
    def __init__(self, num_samples, dimension=1, mean_constant=0.0, length_scale=1.0, output_scale=1.0, noise=0.1, x_range=(-2, 2), horizon=100, seed=None):
        self.num_samples = num_samples
        self.dimension = dimension
        self.x_range = x_range
        self.horizon = horizon
        print(f'noise: {noise}')

        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        
        self.likelihood = GaussianLikelihood()
        self.gp = GaussianProcessConstant(None, None, self.likelihood, mean_constant, length_scale, output_scale, noise)
        self.gp.eval()

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        lb, ub = self.x_range
        x = lb + (ub - lb) * torch.rand([self.horizon, self.dimension])
        y_distribution = self.likelihood(self.gp(x))
        y_mean = y_distribution.mean
        y_covariance = y_distribution.covariance_matrix
        y_distribution_pyotrch = MultivariateNormalTorch(y_mean, y_covariance)
        y = y_distribution_pyotrch.sample().unsqueeze(-1)

        return x, y
    
def load_gpsampler_constant_data(training_args, data_args, model_args):
    train_batch_size = training_args.total_train_batch_size // training_args.num_process
    test_batch_size = training_args.total_test_batch_size // training_args.num_process
    train_dataset = GPSamplerConstantDataset(
        num_samples=training_args.num_train_samples, 
        noise = data_args.noise_scale,
        dimension=model_args.dim_llm_embedding, 
        horizon=training_args.train_horizon)
    test_dataset = GPSamplerConstantDataset(
        num_samples=training_args.num_test_samples, 
        noise = data_args.noise_scale,
        dimension=model_args.dim_llm_embedding, 
        horizon=training_args.test_horizon)
    train_data_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, collate_fn=scalar_collate_fn, num_workers=data_args.num_test_workers)
    test_data_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, collate_fn=scalar_collate_fn, num_workers=data_args.num_test_workers)

    return train_data_loader, test_data_loader
