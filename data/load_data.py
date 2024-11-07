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

class Reward_Dataset(Dataset):
    def __init__(self, dataset_dir, dim_llm_embedding, num_samples, horizon, train_split):
        dataset_dir = os.path.join(dataset_dir, 'reward_{}'.format(dim_llm_embedding))
        self.dataset = load_from_disk(dataset_dir)[train_split]
        self.embeddings = torch.tensor(self.dataset['embedding'])
        self.rewards = torch.tensor(self.dataset['rewards'])
        self.horizon = horizon
        self.num_classes = self.rewards.shape[1]  # Assuming rewards is a 2D tensor
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Randomly sample a list of horizon data from the dataset
        indices = torch.randint(0, len(self.embeddings), (self.horizon,))
        # print(indices)
        
        # Sample W from Dirichlet distribution
        alpha = 0.05
        w = torch.distributions.Dirichlet(torch.full((self.num_classes,), alpha)).sample()
        
        # Get embeddings and rewards for the sampled indices
        x = self.embeddings[indices]
        y = self.rewards[indices]
        
        # Calculate reweighted reward
        reweighted_y = torch.matmul(y, w.unsqueeze(1))

        # print(f'x shape: {x.shape}, y shape: {y.shape}, reweighted_y shape: {reweighted_y.shape}')

        return x, reweighted_y
    

def load_reward_data(training_args, data_args, model_args):
    train_batch_size = training_args.total_train_batch_size // training_args.num_process
    test_batch_size = training_args.total_test_batch_size // training_args.num_process
    train_dataset = Reward_Dataset(data_args.dataset_dir, model_args.dim_llm_embedding, training_args.num_train_samples,  training_args.train_horizon, 'train')
    test_dataset = Reward_Dataset(data_args.dataset_dir, model_args.dim_llm_embedding, training_args.num_test_samples, training_args.test_horizon, 'test')
    train_data_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, collate_fn=scalar_collate_fn, num_workers=data_args.num_test_workers)
    test_data_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, collate_fn=scalar_collate_fn, num_workers=data_args.num_test_workers)

    return train_data_loader, test_data_loader


class ToyRewardDataset(Dataset):
    def __init__(self, num_samples, dim_llm_embedding, horizon, w_file_path, alpha, noise_std=0.1):
        self.num_samples = num_samples
        self.dim_llm_embedding = dim_llm_embedding
        self.horizon = horizon
        self.noise_std = noise_std
        self.alpha = alpha

        self.Reward_W = torch.load(w_file_path)

        self.num_reward_model = self.Reward_W.shape[0]



    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):

        x = torch.randn(self.horizon, self.dim_llm_embedding)

        y = torch.matmul(x, self.Reward_W.T)

        Dirichlet_W = torch.distributions.Dirichlet(torch.full((self.num_reward_model,), self.alpha)).sample()

        reweighted_y = torch.matmul(y, Dirichlet_W.unsqueeze(1))

        return x, reweighted_y
    

def load_toy_reward_data(training_args, data_args, model_args):
    train_batch_size = training_args.total_train_batch_size // training_args.num_process
    test_batch_size = training_args.total_test_batch_size // training_args.num_process
    w_file_path = "/user/al4263/rlhf/TPU/data/W.pt"
    train_dataset = ToyRewardDataset(training_args.num_train_samples, model_args.dim_llm_embedding, training_args.train_horizon, w_file_path, data_args.alpha)
    test_dataset = ToyRewardDataset(training_args.num_test_samples, model_args.dim_llm_embedding, training_args.test_horizon, w_file_path, data_args.alpha)
    train_data_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, collate_fn=scalar_collate_fn, num_workers=data_args.num_test_workers)
    test_data_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, collate_fn=scalar_collate_fn, num_workers=data_args.num_test_workers)

    return train_data_loader, test_data_loader


class ContextualBanditDataset(Dataset):
    def __init__(self, num_samples, context_dim, action_dim, horizon, theta_file_path, noise_std):
        self.num_samples = num_samples
        self.context_dim = context_dim
        self.action_dim = action_dim
        self.horizon = horizon
        self.noise_std = noise_std
        self.arms = torch.load(theta_file_path, weights_only=True)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):

        context = torch.randn(self.horizon, self.context_dim)
        # select a random arm
        arm_indices = torch.randint(0, self.arms.shape[0], (self.horizon,))
        selected_arms = self.arms[arm_indices]
        # response is arm multiplied by context
        response = torch.einsum('hc,hca->ha', context, selected_arms)
        x = torch.cat([context, response], dim=1)
        # w is a random vector form the normal distribution
        W = torch.randn(1, self.context_dim + self.action_dim)
        W = W / torch.norm(W, p='fro')
        y = torch.matmul(x, W.T) 

        noise = torch.randn(y.shape) * self.noise_std
        noisy_y = y + noise

        return x, noisy_y
    

def load_contextual_bandit_data(training_args, data_args, model_args):
    train_batch_size = training_args.total_train_batch_size // training_args.num_process
    test_batch_size = training_args.total_test_batch_size // training_args.num_process
    theta_file_path = data_args.dataset_dir
    train_dataset = ContextualBanditDataset(training_args.num_train_samples, model_args.context_dim, model_args.action_dim, training_args.train_horizon, theta_file_path, data_args.noise_scale)
    test_dataset = ContextualBanditDataset(training_args.num_test_samples, model_args.context_dim, model_args.action_dim, training_args.test_horizon, theta_file_path, data_args.noise_scale)
    train_data_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, collate_fn=scalar_collate_fn, num_workers=data_args.num_test_workers)
    test_data_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, collate_fn=scalar_collate_fn, num_workers=data_args.num_test_workers)

    return train_data_loader, test_data_loader

# class ClassicalContextualBanditDataset(Dataset):
#     def __init__(self, num_samples, context_dim, horizon, noise_std):
#         self.num_samples = num_samples
#         self.context_dim = context_dim
#         self.horizon = horizon
#         self.noise_std = noise_std

#     def __len__(self):
#         return self.num_samples

#     def __getitem__(self, idx):

#         funcs, X_test = setup_gp(1)
#         x_indices = torch.randint(0, X_test.shape[0], (self.horizon,))
#         context = X_test[x_indices]
#         y = funcs[:, x_indices].T

#         noise = torch.randn(y.shape) * self.noise_std
#         noisy_y = y + noise

#         return context, noisy_y

class HyperbolicFunction(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.randn(1))
        self.b = nn.Parameter(torch.randn(1))
        self.c = nn.Parameter(torch.randn(1))

        self.a.requires_grad = False
        self.b.requires_grad = False
        self.c.requires_grad = False
        
    def forward(self, x):
        return self.a * x**2 + self.b * x + self.c
    
class ClassicalContextualBanditDataset(Dataset):
    def __init__(self, num_samples, context_dim, horizon, noise_std):
        self.num_samples = num_samples
        self.context_dim = context_dim
        self.horizon = horizon
        self.noise_std = noise_std

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):

        context = torch.randn(self.horizon, self.context_dim)
        func = HyperbolicFunction()
        y = func(context)

        noise = torch.randn(y.shape) * self.noise_std
        noisy_y = y + noise

        return context, noisy_y

class ClassicalContextualBanditDataset(Dataset):
    def __init__(self, num_samples, context_dim, horizon, noise_std):
        self.num_samples = num_samples
        self.context_dim = context_dim
        self.horizon = horizon
        self.noise_std = noise_std

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):

        context = torch.randn(self.horizon, self.context_dim)
        theta_true = torch.randn(1, self.context_dim)
        theta_true = theta_true / torch.norm(theta_true, p='fro')
        y = torch.matmul(context, theta_true.T)

        noise = torch.randn(y.shape) * self.noise_std
        noisy_y = y + noise

        return context, noisy_y
    

def load_classical_contextual_bandit_data(training_args, data_args, model_args):
    train_batch_size = training_args.total_train_batch_size // training_args.num_process
    test_batch_size = training_args.total_test_batch_size // training_args.num_process
    train_dataset = ClassicalContextualBanditDataset(training_args.num_train_samples, model_args.context_dim, training_args.train_horizon, data_args.noise_scale)
    test_dataset = ClassicalContextualBanditDataset(training_args.num_test_samples, model_args.context_dim, training_args.test_horizon, data_args.noise_scale)
    train_data_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, collate_fn=scalar_collate_fn, num_workers=data_args.num_test_workers)
    test_data_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, collate_fn=scalar_collate_fn, num_workers=data_args.num_test_workers)

    return train_data_loader, test_data_loader

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


