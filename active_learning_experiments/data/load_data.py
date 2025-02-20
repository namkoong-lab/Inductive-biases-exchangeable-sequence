import random
from collections import namedtuple

import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader, Dataset

Rawdata = namedtuple("Rawdata", ["x", "y"])
Prefdata = namedtuple("Prefdata", ["x", "y", "pref", "full_pref"])
import os

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


def scalar_collate_fn(batch):
    x, y = zip(*batch)
    x = torch.stack(x, dim=0)
    y = torch.stack(y, dim=0)
    return Rawdata(x, y)


def pref_collate_fn(batch):
    x, y, pref, full_pref = zip(*batch)
    x = torch.stack(x, dim=0)
    y = torch.stack(y, dim=0)
    pref = torch.stack(pref, dim=0)
    full_pref = torch.stack(full_pref, dim=0)
    return Prefdata(x, y, pref, full_pref)


class Reward_Dataset(Dataset):
    def __init__(
        self, dataset_dir, dim_llm_embedding, num_samples, horizon, train_split
    ):
        dataset_dir = os.path.join(dataset_dir, "reward_{}".format(dim_llm_embedding))
        self.dataset = load_from_disk(dataset_dir)[train_split]
        self.embeddings = torch.tensor(self.dataset["embedding"])
        self.rewards = torch.tensor(self.dataset["rewards"])
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
        w = torch.distributions.Dirichlet(
            torch.full((self.num_classes,), alpha)
        ).sample()

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
    train_dataset = Reward_Dataset(
        data_args.dataset_dir,
        model_args.dim_llm_embedding,
        training_args.num_train_samples,
        training_args.train_horizon,
        "train",
    )
    test_dataset = Reward_Dataset(
        data_args.dataset_dir,
        model_args.dim_llm_embedding,
        training_args.num_test_samples,
        training_args.test_horizon,
        "test",
    )
    train_data_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        collate_fn=scalar_collate_fn,
        num_workers=data_args.num_test_workers,
    )
    test_data_loader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        collate_fn=scalar_collate_fn,
        num_workers=data_args.num_test_workers,
    )

    return train_data_loader, test_data_loader


class ToyRewardDataset(Dataset):
    def __init__(
        self, num_samples, dim_llm_embedding, horizon, w_file_path, alpha, noise_std=0.1
    ):
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

        Dirichlet_W = torch.distributions.Dirichlet(
            torch.full((self.num_reward_model,), self.alpha)
        ).sample()

        reweighted_y = torch.matmul(y, Dirichlet_W.unsqueeze(1))

        return x, reweighted_y


def load_toy_reward_data(training_args, data_args, model_args):
    train_batch_size = training_args.total_train_batch_size // training_args.num_process
    test_batch_size = training_args.total_test_batch_size // training_args.num_process
    w_file_path = "/user/al4263/rlhf/TPU/data/W.pt"
    train_dataset = ToyRewardDataset(
        training_args.num_train_samples,
        model_args.dim_llm_embedding,
        training_args.train_horizon,
        w_file_path,
        data_args.alpha,
    )
    test_dataset = ToyRewardDataset(
        training_args.num_test_samples,
        model_args.dim_llm_embedding,
        training_args.test_horizon,
        w_file_path,
        data_args.alpha,
    )
    train_data_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        collate_fn=scalar_collate_fn,
        num_workers=data_args.num_test_workers,
    )
    test_data_loader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        collate_fn=scalar_collate_fn,
        num_workers=data_args.num_test_workers,
    )

    return train_data_loader, test_data_loader


class ContextualBanditDataset(Dataset):
    def __init__(
        self, num_samples, context_dim, action_dim, horizon, theta_file_path, noise_std
    ):
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
        response = torch.einsum("hc,hca->ha", context, selected_arms)
        x = torch.cat([context, response], dim=1)
        # w is a random vector form the normal distribution
        W = torch.randn(1, self.context_dim + self.action_dim)
        W = W / torch.norm(W, p="fro")
        y = torch.matmul(x, W.T)

        noise = torch.randn(y.shape) * self.noise_std
        noisy_y = y + noise

        return x, noisy_y


def load_contextual_bandit_data(training_args, data_args, model_args):
    train_batch_size = training_args.total_train_batch_size // training_args.num_process
    test_batch_size = training_args.total_test_batch_size // training_args.num_process
    theta_file_path = data_args.dataset_dir
    train_dataset = ContextualBanditDataset(
        training_args.num_train_samples,
        model_args.dim_llm_embedding,
        model_args.action_dim,
        training_args.train_horizon,
        theta_file_path,
        data_args.noise_scale,
    )
    test_dataset = ContextualBanditDataset(
        training_args.num_test_samples,
        model_args.dim_llm_embedding,
        model_args.action_dim,
        training_args.test_horizon,
        theta_file_path,
        data_args.noise_scale,
    )
    train_data_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        collate_fn=scalar_collate_fn,
        num_workers=data_args.num_test_workers,
    )
    test_data_loader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        collate_fn=scalar_collate_fn,
        num_workers=data_args.num_test_workers,
    )

    return train_data_loader, test_data_loader


# class HyperbolicFunction(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.a = nn.Parameter(torch.randn(1))
#         self.b = nn.Parameter(torch.randn(1))
#         self.c = nn.Parameter(torch.randn(1))

#         self.a.requires_grad = False
#         self.b.requires_grad = False
#         self.c.requires_grad = False

#     def forward(self, x):
#         return self.a * x**2 + self.b * x + self.c

# class ClassicalContextualBanditDataset(Dataset):
#     def __init__(self, num_samples, context_dim, horizon, noise_std):
#         self.num_samples = num_samples
#         self.context_dim = context_dim
#         self.horizon = horizon
#         self.noise_std = noise_std

#     def __len__(self):
#         return self.num_samples

#     def __getitem__(self, idx):

#         context = torch.randn(self.horizon, self.context_dim)
#         func = HyperbolicFunction()
#         y = func(context)

#         noise = torch.randn(y.shape) * self.noise_std
#         noisy_y = y + noise

#         return context, noisy_y


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
        theta_true = theta_true / torch.norm(theta_true, p="fro")
        y = torch.matmul(context, theta_true.T)

        noise = torch.randn(y.shape) * self.noise_std
        noisy_y = y + noise

        return context, noisy_y


def load_classical_contextual_bandit_data(training_args, data_args, model_args):
    train_batch_size = training_args.total_train_batch_size // training_args.num_process
    test_batch_size = training_args.total_test_batch_size // training_args.num_process
    train_dataset = ClassicalContextualBanditDataset(
        training_args.num_train_samples,
        model_args.dim_llm_embedding,
        training_args.train_horizon,
        data_args.noise_scale,
    )
    test_dataset = ClassicalContextualBanditDataset(
        training_args.num_test_samples,
        model_args.dim_llm_embedding,
        training_args.test_horizon,
        data_args.noise_scale,
    )
    train_data_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        collate_fn=scalar_collate_fn,
        num_workers=data_args.num_test_workers,
    )
    test_data_loader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        collate_fn=scalar_collate_fn,
        num_workers=data_args.num_test_workers,
    )

    return train_data_loader, test_data_loader


class MixedContextualBanditDataset(Dataset):
    def __init__(self, num_samples, context_dim, horizon, noise_std):
        self.num_samples = num_samples
        self.context_dim = context_dim
        self.horizon = horizon
        self.noise_std = noise_std

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):

        context = torch.randn(self.horizon, self.context_dim)
        theta_true = torch.randn(1, self.context_dim) * torch.tensor(
            random.choice([1.0, 1.5, 2.0, 2.5, 3.0])
        )
        y = torch.matmul(context, theta_true.T)

        noise = torch.randn(y.shape) * torch.tensor(
            random.choice(torch.linspace(0.1, 1.0, 10).tolist())
        )
        noisy_y = y + noise

        return context, noisy_y


def load_mixed_contextual_bandit_data(training_args, data_args, model_args):
    train_batch_size = training_args.total_train_batch_size // training_args.num_process
    test_batch_size = training_args.total_test_batch_size // training_args.num_process
    train_dataset = MixedContextualBanditDataset(
        training_args.num_train_samples,
        model_args.dim_llm_embedding,
        training_args.train_horizon,
        data_args.noise_scale,
    )
    test_dataset = MixedContextualBanditDataset(
        training_args.num_test_samples,
        model_args.dim_llm_embedding,
        training_args.test_horizon,
        data_args.noise_scale,
    )
    train_data_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        collate_fn=scalar_collate_fn,
        num_workers=data_args.num_test_workers,
    )
    test_data_loader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        collate_fn=scalar_collate_fn,
        num_workers=data_args.num_test_workers,
    )

    return train_data_loader, test_data_loader


def generate_al_matmul_data(context_dim, horizon, noise_std):
    context = torch.randn(horizon, context_dim)
    theta_true = torch.randn(1, context_dim)
    y = torch.matmul(context, theta_true.T)

    noise_scales = torch.norm(context, dim=1, keepdim=True) * noise_std
    noise = torch.randn(size=(horizon, 1)) * noise_scales
    noisy_y = y + noise

    return context, noisy_y


class ActiveLearningMatMulDataset(Dataset):
    SUB_FOLDER_PATH = "al_matmul"

    def __init__(
        self, num_samples, context_dim, horizon, noise_std, data_folder, train=True
    ):
        self.num_samples = num_samples
        self.context_dim = context_dim
        self.horizon = horizon
        self.noise_std = noise_std

        self.data_folder = f"{data_folder}/{self.SUB_FOLDER_PATH}/context_dim={context_dim}_horizon={horizon}_noise={noise_std}"

        if train:
            self.contexts = torch.load(f"{self.data_folder}/train_contexts.pt")
            self.y = torch.load(f"{self.data_folder}/train_y.pt")
        else:
            self.contexts = torch.load(f"{self.data_folder}/test_contexts.pt")
            self.y = torch.load(f"{self.data_folder}/test_y.pt")

        self.available_num_samples = self.contexts.shape[0]
        self.contexts = self.contexts[: self.num_samples]
        self.y = self.y[: self.num_samples]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.contexts[idx], self.y[idx]


def load_active_learning_matmul_data(training_args, data_args, model_args):
    train_batch_size = training_args.total_train_batch_size // training_args.num_process
    test_batch_size = training_args.total_test_batch_size // training_args.num_process
    train_dataset = ActiveLearningMatMulDataset(
        training_args.num_train_samples,
        model_args.dim_llm_embedding,
        training_args.train_horizon,
        data_args.noise_scale,
        data_args.dataset_dir,
        train=True,
    )
    test_dataset = ActiveLearningMatMulDataset(
        training_args.num_test_samples,
        model_args.dim_llm_embedding,
        training_args.test_horizon,
        data_args.noise_scale,
        data_args.dataset_dir,
        train=False,
    )

    train_data_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        collate_fn=scalar_collate_fn,
        num_workers=data_args.num_test_workers,
    )
    test_data_loader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        collate_fn=scalar_collate_fn,
        num_workers=data_args.num_test_workers,
    )

    return train_data_loader, test_data_loader


def generate_al_additive_data(context_dim, horizon, noise_std):
    context = torch.randn(horizon, context_dim)
    theta_true = torch.randn(1)
    y = torch.sum(context, dim=1) + theta_true
    y = y.reshape(-1, 1)

    noise_scales = torch.norm(context, dim=1, keepdim=True) * noise_std
    noise = torch.randn(size=(horizon, 1)) * noise_scales
    noisy_y = y + noise

    return context, noisy_y


class ActiveLearningAddDataset(Dataset):
    SUB_FOLDER_PATH = "al_add"

    def __init__(
        self, num_samples, context_dim, horizon, noise_std, data_folder, train=True
    ):
        self.num_samples = num_samples
        self.context_dim = context_dim
        self.horizon = horizon
        self.noise_std = noise_std

        self.data_folder = f"{data_folder}/{self.SUB_FOLDER_PATH}/context_dim={context_dim}_horizon={horizon}_noise={noise_std}"
        if train:
            self.contexts = torch.load(f"{self.data_folder}/train_contexts.pt")
            self.y = torch.load(f"{self.data_folder}/train_y.pt")
        else:
            self.contexts = torch.load(f"{self.data_folder}/test_contexts.pt")
            self.y = torch.load(f"{self.data_folder}/test_y.pt")

        self.available_num_samples = self.contexts.shape[0]
        self.contexts = self.contexts[: self.num_samples]
        self.y = self.y[: self.num_samples]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.contexts[idx], self.y[idx]


def load_active_learning_add_data(training_args, data_args, model_args):
    train_batch_size = training_args.total_train_batch_size // training_args.num_process
    test_batch_size = training_args.total_test_batch_size // training_args.num_process
    train_dataset = ActiveLearningAddDataset(
        training_args.num_train_samples,
        model_args.dim_llm_embedding,
        training_args.train_horizon,
        data_args.noise_scale,
        data_args.dataset_dir,
        train=True,
    )
    test_dataset = ActiveLearningAddDataset(
        training_args.num_test_samples,
        model_args.dim_llm_embedding,
        training_args.test_horizon,
        data_args.noise_scale,
        data_args.dataset_dir,
        train=False,
    )
    train_data_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        collate_fn=scalar_collate_fn,
        num_workers=data_args.num_test_workers,
    )
    test_data_loader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        collate_fn=scalar_collate_fn,
        num_workers=data_args.num_test_workers,
    )

    return train_data_loader, test_data_loader


def generate_al_regions_data(
    context_dim, horizon, noise_std, with_high_aleatoric=False
):
    """Here we want to build a dataset with 10 regions, each with a different epistemic and aleatoric uncertainty."""
    if context_dim != 1:
        raise ValueError("Context dimension must be 1 for this dataset")
    context = torch.rand(size=(horizon, context_dim)) * 10 - 5

    if not with_high_aleatoric:
        # Keeping sum of variance constant across regions
        theta_scales = torch.tensor(
            [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
        )
        noise_scales = torch.tensor(
            [0.95, 0.85, 0.75, 0.65, 0.55, 0.45, 0.35, 0.25, 0.15, 0.05]
        )
    else:
        # Have the first region with high aleatoric uncertainty and total uncertainty
        theta_scales = torch.tensor(
            [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
        )
        noise_scales = torch.tensor(
            [1.95, 0.85, 0.75, 0.65, 0.55, 0.45, 0.35, 0.25, 0.15, 0.05]
        )

    theta_true = torch.randn(10) * theta_scales
    theta_shared = torch.randn(1)

    # split regions into [-5, -4], (-4, -3], ..., (3, 4], (4, 5]
    def is_in_region(context, i):
        if i == 0:
            return torch.logical_and(context >= -5, context <= -4)
        else:
            return torch.logical_and(context > -5 + i, context <= -5 + i + 1)

    y = torch.zeros(horizon, 1)
    for i in range(10):
        noise = torch.randn(size=(horizon, 1)) * noise_scales[i]
        y += is_in_region(context, i).float() * (theta_true[i] + theta_shared + noise)

    return context, y


def generate_al_hregions_data(context_dim, horizon, noise_std, num_regions=100):
    if context_dim != 1:
        raise ValueError("Context dimension must be 1 for this dataset")

    # x ranges from -5 to 5
    context = torch.rand(size=(horizon, context_dim)) * 10 - 5

    # half with low and other half with high epistemic uncertainty
    theta_scales = torch.cat(
        [torch.ones(num_regions // 2) * 0.05, torch.ones(num_regions // 2) * 0.95]
    )
    noise_scales = torch.cat(
        [torch.ones(num_regions // 2) * 1.55, torch.ones(num_regions // 2) * 0.05]
    )

    theta_true = torch.randn(num_regions) * theta_scales

    def is_in_region(context, i, num_regions):
        interval = 10 / num_regions
        if i == 0:
            return torch.logical_and(context >= -5, context <= -5 + interval)
        else:
            return torch.logical_and(
                context > -5 + i * interval, context <= -5 + (i + 1) * interval
            )

    y = torch.zeros(horizon, 1)
    for i in range(num_regions):
        noise = torch.randn(size=(horizon, 1)) * noise_scales[i]
        iir = is_in_region(context, i, num_regions)
        y += iir.float() * (theta_true[i] + noise)
    return context, y


class ActiveLearningRegionDataset(Dataset):
    def __init__(
        self,
        num_samples,
        context_dim,
        horizon,
        noise_std,
        data_folder,
        with_high_aleatoric=False,
        with_high_regions=False,
        num_regions=100,
        train=True,
    ):
        if with_high_regions:
            self.SUB_FOLDER_PATH = f"al_hregions_{num_regions}"
        else:
            if with_high_aleatoric:
                self.SUB_FOLDER_PATH = "al_regions_ha"
            else:
                self.SUB_FOLDER_PATH = "al_regions"
        self.num_samples = num_samples
        self.context_dim = context_dim
        self.horizon = horizon
        self.noise_std = noise_std

        self.data_folder = f"{data_folder}/{self.SUB_FOLDER_PATH}/context_dim={context_dim}_horizon={horizon}_noise={noise_std}"
        if train:
            self.contexts = torch.load(f"{self.data_folder}/train_contexts.pt")
            self.y = torch.load(f"{self.data_folder}/train_y.pt")
        else:
            self.contexts = torch.load(f"{self.data_folder}/test_contexts.pt")
            self.y = torch.load(f"{self.data_folder}/test_y.pt")

        self.available_num_samples = self.contexts.shape[0]
        self.contexts = self.contexts[: self.num_samples]
        self.y = self.y[: self.num_samples]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.contexts[idx], self.y[idx]


def load_active_learning_region_data(
    training_args,
    data_args,
    model_args,
    with_high_aleatoric=False,
    with_high_regions=False,
    num_regions=100,
):
    train_batch_size = training_args.total_train_batch_size // training_args.num_process
    test_batch_size = training_args.total_test_batch_size // training_args.num_process
    train_dataset = ActiveLearningRegionDataset(
        training_args.num_train_samples,
        model_args.dim_llm_embedding,
        training_args.train_horizon,
        data_args.noise_scale,
        data_args.dataset_dir,
        with_high_aleatoric=with_high_aleatoric,
        with_high_regions=with_high_regions,
        num_regions=num_regions,
        train=True,
    )
    test_dataset = ActiveLearningRegionDataset(
        training_args.num_test_samples,
        model_args.dim_llm_embedding,
        training_args.test_horizon,
        data_args.noise_scale,
        data_args.dataset_dir,
        with_high_aleatoric=with_high_aleatoric,
        with_high_regions=with_high_regions,
        num_regions=num_regions,
        train=False,
    )

    train_data_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        collate_fn=scalar_collate_fn,
        num_workers=data_args.num_test_workers,
    )
    test_data_loader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        collate_fn=scalar_collate_fn,
        num_workers=data_args.num_test_workers,
    )
    return train_data_loader, test_data_loader


def generate_al_regions_v2_data(context_dim, horizon, noise_std, num_regions=50):
    if context_dim != 1:
        raise ValueError("Context dimension must be 1 for this dataset")

    # x ranges from -5 to 5
    context = torch.rand(size=(horizon, context_dim)) * 10 - 5

    # half with low and other half with high epistemic uncertainty
    theta_scales = torch.cat(
        [torch.ones(num_regions // 2) * 0.9, torch.ones(num_regions // 2) * 0.5]
    )
    noise_scales = torch.cat(
        [torch.ones(num_regions // 2) * 0.1, torch.ones(num_regions // 2) * 0.7]
    )

    theta_true = torch.randn(num_regions) * theta_scales

    def is_in_region(context, i, num_regions):
        interval = 10 / num_regions
        if i == 0:
            return torch.logical_and(context >= -5, context <= -5 + interval)
        else:
            return torch.logical_and(
                context > -5 + i * interval, context <= -5 + (i + 1) * interval
            )

    y = torch.zeros(horizon, 1)
    for i in range(num_regions):
        noise = torch.randn(size=(horizon, 1)) * noise_scales[i]
        iir = is_in_region(context, i, num_regions)
        y += iir.float() * (theta_true[i] + noise)
    return context, y


class ActiveLearningDataset(Dataset):
    def __init__(
        self,
        num_samples,
        context_dim,
        horizon,
        noise_std,
        data_folder,
        al_data_name,
        train=True,
    ):
        self.SUB_FOLDER_PATH = al_data_name
        self.num_samples = num_samples
        self.context_dim = context_dim
        self.horizon = horizon
        self.noise_std = noise_std

        self.data_folder = f"{data_folder}/{self.SUB_FOLDER_PATH}/context_dim={context_dim}_horizon={horizon}_noise={noise_std}"
        print(f"Loading active learning data from {self.data_folder}")
        if train:
            self.contexts = torch.load(f"{self.data_folder}/train_contexts.pt")
            self.y = torch.load(f"{self.data_folder}/train_y.pt")
        else:
            self.contexts = torch.load(f"{self.data_folder}/test_contexts.pt")
            self.y = torch.load(f"{self.data_folder}/test_y.pt")

        self.available_num_samples = self.contexts.shape[0]
        self.contexts = self.contexts[: self.num_samples]
        self.y = self.y[: self.num_samples]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.contexts[idx], self.y[idx]


def load_active_learning_data(
    training_args,
    data_args,
    model_args,
    al_data_name,
):
    if al_data_name == "al_matmul":
        return load_active_learning_matmul_data(training_args, data_args, model_args)
    elif al_data_name == "al_add":
        return load_active_learning_add_data(training_args, data_args, model_args)
    elif al_data_name == "al_regions":
        return load_active_learning_region_data(
            training_args, data_args, model_args, with_high_aleatoric=False
        )
    elif al_data_name == "al_regions_ha":
        return load_active_learning_region_data(
            training_args, data_args, model_args, with_high_aleatoric=True
        )
    elif al_data_name == "al_hregions":
        return load_active_learning_region_data(
            training_args, data_args, model_args, with_high_regions=True
        )
    else:
        train_batch_size = (
            training_args.total_train_batch_size // training_args.num_process
        )
        test_batch_size = (
            training_args.total_test_batch_size // training_args.num_process
        )
        train_dataset = ActiveLearningDataset(
            num_samples=training_args.num_train_samples,
            context_dim=model_args.dim_llm_embedding,
            horizon=training_args.train_horizon,
            noise_std=data_args.noise_scale,
            data_folder=data_args.dataset_dir,
            al_data_name=al_data_name,
            train=True,
        )
        test_dataset = ActiveLearningDataset(
            training_args.num_test_samples,
            model_args.dim_llm_embedding,
            training_args.test_horizon,
            data_args.noise_scale,
            data_args.dataset_dir,
            al_data_name=al_data_name,
            train=False,
        )
        train_data_loader = DataLoader(
            train_dataset,
            batch_size=train_batch_size,
            shuffle=True,
            collate_fn=scalar_collate_fn,
            num_workers=data_args.num_test_workers,
        )
        test_data_loader = DataLoader(
            test_dataset,
            batch_size=test_batch_size,
            shuffle=False,
            collate_fn=scalar_collate_fn,
            num_workers=data_args.num_test_workers,
        )
        return train_data_loader, test_data_loader
