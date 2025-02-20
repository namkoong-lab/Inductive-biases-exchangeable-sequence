import torch
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('..')
import torch
import os
from dataclasses import dataclass, field
import yaml
from models.reward_tnp import Reward_TNP
import torch
from types import SimpleNamespace
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.distributions as dist
import torch.multiprocessing as mp
import json
from tqdm import tqdm
from functools import partial
import torch.nn as nn
import math
import gpytorch

@dataclass
class ModelArguments:
    dim_llm_embedding: int = field(default=1024)
    dim_y: int = field(default=1)
    repeat_y: int = field(default=1)
    emb_depth: int = field(default=0)
    d_model: int = field(default=2048)
    dim_feedforward: int = field(default=5120)
    nhead: int = field(default=256)
    dropout: float = field(default=0.2)
    activation: str = field(default="gelu")
    num_layers: int = field(default=4)
    bound_std: bool = field(default=True)
    embed_type: str = field(default="embed_concat")
    uncertainty: str = field(default="normal")
    loss_type: str = field(default="logprob")
    pad_value: float = field(default=0.0)
    context_dim: int = field(default=8)
    action_dim: int = field(default=4)
    gradient_type: str = field(default="std")
    borders_data_dir: str = field(default=None)
        
class LinUCBDisjoint:
    def __init__(self, alpha, d):
        """
        Initializes the LinUCB algorithm with disjoint linear models.

        Args:
            alpha (float): Exploration parameter.
            d (int): Dimension of feature vectors.
        """
        self.alpha = alpha
        self.d = d
        # Using defaultdict to handle new arms dynamically
        self.A = defaultdict(lambda: torch.eye(d))
        self.b = defaultdict(lambda: torch.zeros(d, 1))
        # To store the inverse of A for each arm to save computation
        self.A_inv = defaultdict(lambda: torch.eye(d))
    
    def select_arm(self, feature_matrix):
        """
        Selects the arm with the highest UCB score.

        Args:
            feature_matrix (torch.Tensor): Tensor of shape (num_arms, d) representing features of all arms.

        Returns:
            int: The index of the selected arm.
        """
        num_arms = feature_matrix.size(0)
        p = torch.zeros(num_arms)
        
        for a in range(num_arms):
            x_a = feature_matrix[a].unsqueeze(1)  # Shape: (d, 1)
            theta_a = torch.matmul(self.A_inv[a], self.b[a])  # Shape: (d, 1)
            mean = torch.matmul(theta_a.t(), x_a).item()
            uncertainty = self.alpha * torch.sqrt(torch.matmul(x_a.t(), torch.matmul(self.A_inv[a], x_a))).item()
            p[a] = mean + uncertainty
        
        # Select the arm with the highest p value
        a_t = torch.argmax(p).item()
        return a_t

    def update(self, a_t, x_t, r_t):
        """
        Updates the model parameters for the selected arm.

        Args:
            a_t (int): The index of the selected arm.
            x_t (torch.Tensor): Feature vector of the selected arm, shape (d,).
            r_t (float): Observed reward.
        """
        x_t = x_t.unsqueeze(1)  # Shape: (d, 1)
        self.A[a_t] += torch.matmul(x_t, x_t.t())
        self.b[a_t] += r_t * x_t
        # Update the inverse using the Sherman-Morrison formula for efficiency
        A_inv_a = self.A_inv[a_t]
        x = x_t
        A_inv_x = torch.matmul(A_inv_a, x)  # Shape: (d, 1)
        denominator = 1.0 + torch.matmul(x.t(), A_inv_x)  # Shape: (1, 1)
        numerator = torch.matmul(A_inv_x, A_inv_x.t())  # Shape: (d, d)
        self.A_inv[a_t] = A_inv_a - numerator / denominator    


class LinUCBGreedy:
    def __init__(self, alpha, d):
        """
        Initializes the LinUCB algorithm with disjoint linear models.

        Args:
            alpha (float): Exploration parameter.
            d (int): Dimension of feature vectors.
        """
        self.alpha = alpha
        self.d = d
        # Using defaultdict to handle new arms dynamically
        self.A = defaultdict(lambda: torch.eye(d))
        self.b = defaultdict(lambda: torch.zeros(d, 1))
        # To store the inverse of A for each arm to save computation
        self.A_inv = defaultdict(lambda: torch.eye(d))
    
    def select_arm(self, feature_matrix):
        """
        Selects the arm with the highest UCB score.

        Args:
            feature_matrix (torch.Tensor): Tensor of shape (num_arms, d) representing features of all arms.

        Returns:
            int: The index of the selected arm.
        """
        num_arms = feature_matrix.size(0)
        p = torch.zeros(num_arms)
        
        for a in range(num_arms):
            x_a = feature_matrix[a].unsqueeze(1)  # Shape: (d, 1)
            theta_a = torch.matmul(self.A_inv[a], self.b[a])  # Shape: (d, 1)
            mean = torch.matmul(theta_a.t(), x_a).item()
            # uncertainty = self.alpha * torch.sqrt(torch.matmul(x_a.t(), torch.matmul(self.A_inv[a], x_a))).item()
            p[a] = mean 
        
        # Select the arm with the highest p value
        a_t = torch.argmax(p).item()
        return a_t

    def update(self, a_t, x_t, r_t):
        """
        Updates the model parameters for the selected arm.

        Args:
            a_t (int): The index of the selected arm.
            x_t (torch.Tensor): Feature vector of the selected arm, shape (d,).
            r_t (float): Observed reward.
        """
        x_t = x_t.unsqueeze(1)  # Shape: (d, 1)
        self.A[a_t] += torch.matmul(x_t, x_t.t())
        self.b[a_t] += r_t * x_t
        # Update the inverse using the Sherman-Morrison formula for efficiency
        A_inv_a = self.A_inv[a_t]
        x = x_t
        A_inv_x = torch.matmul(A_inv_a, x)  # Shape: (d, 1)
        denominator = 1.0 + torch.matmul(x.t(), A_inv_x)  # Shape: (1, 1)
        numerator = torch.matmul(A_inv_x, A_inv_x.t())  # Shape: (d, d)
        self.A_inv[a_t] = A_inv_a - numerator / denominator

class TS_machine:
    def __init__(self, uq_model, device, dim_llm_embedding, inner_loop_iterations, num_imag_samples):
        self.uq_model = uq_model
        self.device = device
        self.dim_llm_embedding = dim_llm_embedding
        self.inner_loop_iterations = inner_loop_iterations
        self.num_imag_samples = num_imag_samples
        self.uq_joint_context_x = []
        self.uq_joint_context_y = []
        self.uq_joint_pred_loss = []
        self.uq_joint_actual_rewards = []

    def predict(self, x_t, beta_hat):
        beta_hat = beta_hat.to(self.device)
        x_t = x_t.to(self.device)
        uq_joint_predicted_rewards = torch.matmul(x_t, beta_hat.T)
        return uq_joint_predicted_rewards

    def update_batch(self, x_t, reward_t):
        self.uq_joint_context_x.append(x_t.unsqueeze(0))
        self.uq_joint_context_y.append(torch.tensor(reward_t, device=self.device).unsqueeze(0).unsqueeze(0))
    
    def set_batch(self, x_t, reward_t):
        batch = SimpleNamespace(
        xc=torch.stack(self.uq_joint_context_x, dim=1).repeat(self.num_imag_samples, 1, 1).to(self.device) if self.uq_joint_context_x else torch.empty(self.num_imag_samples, 0, self.dim_llm_embedding, device=self.device),
        yc=torch.stack(self.uq_joint_context_y, dim=1).repeat(self.num_imag_samples, 1, 1).to(self.device) if self.uq_joint_context_y else torch.empty(self.num_imag_samples, 0, 1, device=self.device),
        xt=x_t,
        yt=torch.zeros(reward_t.shape, device=self.device))

        return batch

    def ts_imagination(self, x_t, reward_t):
        inner_context_x = []
        inner_context_y = []
        uq_joint_batch = self.set_batch(x_t, reward_t)
        dim_llm_embedding = self.dim_llm_embedding
        device = self.device
        uq_model = self.uq_model
        inner_loop_iterations = self.inner_loop_iterations
        batch_size = self.num_imag_samples
        uq_context_x = uq_joint_batch.xc
        uq_context_y = uq_joint_batch.yc
        

        for i in range(inner_loop_iterations):

            # Generate random contexts
            context = torch.randn(batch_size, 1, dim_llm_embedding, device=device)

            inner_xc = torch.cat(inner_context_x, dim=1) if inner_context_x else torch.empty(batch_size, 0, dim_llm_embedding, device=device)
            full_xc = torch.cat([uq_context_x, inner_xc], dim=1) 
            inner_yc = torch.cat(inner_context_y, dim=1) if inner_context_y else torch.empty(batch_size, 0, 1, device=device)
            full_yc = torch.cat([uq_context_y, inner_yc], dim=1) 
            full_xc = full_xc.to(device)
            full_yc = full_yc.to(device)
            batch = SimpleNamespace(
                xc=full_xc,
                yc=full_yc,
                xt=context,  # Shape: num_arms,1, dim_llm_embedding
                yt=torch.zeros(batch_size, 1, 1, device=device)  # Shape: num_arms,1,1
            )


            predicted_rewards = uq_model.predict(batch)
            inner_context_x.append(context)
            inner_context_y.append(predicted_rewards)

        # Concatenate all inner context data
        X = torch.cat(inner_context_x, dim=1)  
        y = torch.cat(inner_context_y, dim=1)  
        X = torch.cat([uq_context_x, X], dim=1)
        y = torch.cat([uq_context_y, y], dim=1)

        beta_hat = []
        for i in range(X.shape[0]):
            beta = torch.linalg.lstsq(X[i], y[i]).solution
            beta_hat.append(beta)
        beta_hats = torch.stack(beta_hat).squeeze()

        beta_mean = torch.mean(beta_hats, dim=0)
        beta_cov = torch.cov(beta_hats.T)

        sampled_beta = beta_mean.unsqueeze(0).unsqueeze(-1)

        if beta_mean.dim() == 0:
            beta_mean = beta_mean.unsqueeze(0)
        # Ensure beta_cov is at least 2-dimensional
        if beta_cov.dim() < 2:
            beta_cov = beta_cov.unsqueeze(0).unsqueeze(0)

        mvn = dist.MultivariateNormal(beta_mean, beta_cov)
        sampled_beta = mvn.sample().unsqueeze(0).unsqueeze(-1)  # Shape: [1, dim_llm_embedding, 1]

        return X, y, sampled_beta

def run_exp(process_id):
    # Parameters
    alpha = 1.0
    d = 1        # Feature dimension
    num_arms = 10
    T = 100   # Number of time steps
    inner_loop_iterations = 50
    num_imag_samples = 100
    noise_scale = 0.2

    gpu_id = process_id % 8
    torch.cuda.set_device(gpu_id)
    # set seed
    torch.manual_seed(process_id)
    torch.cuda.manual_seed(process_id)
    torch.cuda.manual_seed_all(process_id)
    device = torch.device(f'cuda:{gpu_id}') if torch.cuda.is_available() else torch.device('cpu')

    uq_config = yaml.load(open('/user/al4263/rlhf/TPU/scripts/ccb_uq_normal.yaml', 'r'), Loader=yaml.FullLoader)
    # use config to create model arguments
    uq_model_args = ModelArguments(**uq_config['model_args'])

    uq_models = []
    for _ in range(num_arms):
        uq_model = Reward_TNP(uq_model_args)
        uq_checkpoint_dir = f'/shared/share_mala/Leon/CCB_1/UQ-Horizon-2000_Noise_0.2/model_checkpoint_85.pt'
        uq_model.load_state_dict(torch.load(uq_checkpoint_dir, weights_only=True))
        uq_model.eval()
        uq_model.to(device)
        uq_models.append(uq_model)

    ts_machine_list = [TS_machine(uq_model, device, d, inner_loop_iterations, num_imag_samples) for uq_model in uq_models]
    linucb = LinUCBDisjoint(alpha=alpha, d=d)
    linucb_greedy = LinUCBGreedy(alpha=alpha, d=d)
    linucb_eps = LinUCBGreedy(alpha=alpha, d=d)

    # Define a true theta for each arm (for simulation purposes)
    theta_true = torch.randn(num_arms, d)
    for arm in range(num_arms):
        theta_true[arm] = theta_true[arm] / torch.norm(theta_true[arm], p='fro')
    # Initialize cumulative rewards\
    uq_rewards = []
    cum_rewards = []
    cum_rewards_greedy = []
    cum_rewards_eps = []
    oracle_cum_rewards = []
    env_cum_rewards = []
    total_reward = 0.0
    total_reward_greedy = 0.0
    total_reward_eps = 0.0
    total_oracle_reward = 0.0
    total_env_reward = 0.0
    total_uq_reward = 0.0

    for t in tqdm(range(1, T + 1), desc=f"Running experiment {process_id}"):
        # Simulate feature vectors for each arm
        # In practice, these would come from the environment
        feature_matrix = torch.randn(num_arms, d)
        expected_rewards = torch.matmul(feature_matrix, theta_true.T)
        expected_rewards = torch.diag(expected_rewards)
        env_rewards = expected_rewards + torch.randn(expected_rewards.shape) * noise_scale

        # Select an arm using LinUCB
        a_t = linucb.select_arm(feature_matrix)
        x_t = feature_matrix[a_t]
        r_t = env_rewards[a_t]
        linucb.update(a_t, x_t, r_t)
        total_reward += r_t.item()
        cum_rewards.append(total_reward)

        ### select arms with LinUCB greedy
        a_t = linucb_greedy.select_arm(feature_matrix)
        x_t = feature_matrix[a_t]
        r_t = env_rewards[a_t]
        linucb_greedy.update(a_t, x_t, r_t)
        total_reward_greedy += r_t.item()
        cum_rewards_greedy.append(total_reward_greedy)

        ### select arms with LinUCB epsilon greedy
        if np.random.rand() < 0.1:
            a_t = np.random.randint(num_arms)
        else:
            a_t = linucb_eps.select_arm(feature_matrix)
        x_t = feature_matrix[a_t]
        r_t = env_rewards[a_t]
        linucb_eps.update(a_t, x_t, r_t)
        total_reward_eps += r_t.item()
        cum_rewards_eps.append(total_reward_eps)

        ### select arms with TS imagination
        with torch.no_grad():
            reward_list = []
            for arm in range(num_arms):
                x_t = feature_matrix[arm]
                X, y, beta_hat = ts_machine_list[arm].ts_imagination(x_t, r_t)
                predicted_rewards = ts_machine_list[arm].predict(x_t, beta_hat)
                reward_list.append(predicted_rewards)
            
            predicted_rewards_list = torch.stack(reward_list, dim=0)
            a_t = torch.argmax(predicted_rewards_list).item()
            total_uq_reward += env_rewards[a_t].item()
            uq_rewards.append(total_uq_reward)
            ts_machine_list[a_t].update_batch(feature_matrix[a_t], env_rewards[a_t].item())
        # for arm in range(num_arms):
        #     ts_machine_list[arm].update_batch(feature_matrix[arm], env_rewards[arm].item())

        # Oracle selects the best arm based on true theta
        oracle_a = np.argmax(expected_rewards)
        oracle_reward = expected_rewards[oracle_a]
        total_oracle_reward += oracle_reward.item()
        oracle_cum_rewards.append(total_oracle_reward)

        # Update cumulative environment reward, which is the max of the 
        env_a = np.argmax(env_rewards)
        env_reward = env_rewards[env_a]
        total_env_reward += env_reward.item()
        env_cum_rewards.append(total_env_reward)

    results = {
        'uq_joint_cumulative': uq_rewards,
        'oracle_cumulative': oracle_cum_rewards,
        'env_cumulative': env_cum_rewards,
        'linucb_cumulative': cum_rewards,
        'linucb_greedy_cumulative': cum_rewards_greedy,
        'linucb_eps_cumulative': cum_rewards_eps,
    }
    return results

def run_wrapper(process_id):
    result = run_exp(process_id)
    return result

def main():
    num_jobs = 100
    d = 1
    noise_scale = 0.2
    # pool = mp.Pool(processes=num_jobs)
    # run_experiment_with_args = partial(run_exp)
    # all_results = list(tqdm(pool.imap(run_experiment_with_args, range(num_jobs)), total=num_jobs, desc="Running experiments"))

    # pool.close()
    # pool.join()
    ctx = mp.get_context("spawn")

    with ctx.Pool(processes=num_jobs) as pool:
        run_experiment_with_args = partial(run_wrapper)
        all_results = list(tqdm(pool.imap(run_experiment_with_args, range(num_jobs)), total=num_jobs, desc="Running experiments"))

    # Combine results from all jobs
    combined_results = {key: [] for key in all_results[0].keys()}
    for result in all_results:
        for key in result:
            combined_results[key].append(result[key])

    # Calculate mean and standard deviation
# Calculate mean and standard deviation
    final_results = {}
    for key in combined_results:
        # Convert list of lists to 2D numpy array
        data = np.array(combined_results[key])
        # Calculate mean and std along the first axis (across experiments)
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        final_results[key] = (mean, std)

    # Plot results
    plt.figure(figsize=(12, 6))
    T = len(final_results['linucb_cumulative'][0])
    x = np.arange(T)

    for label, color, key in [('LinUCB', 'red', 'linucb_cumulative'), 
                              ('LinUCB Greedy', 'orange', 'linucb_greedy_cumulative'), 
                              ('LinUCB Eps Greedy', 'green', 'linucb_eps_cumulative'), 
                              ('UQ TS', 'blue', 'uq_joint_cumulative'), 
                              ('Oracle', 'purple', 'oracle_cumulative'), 
                              ('Environment', 'gray', 'env_cumulative')]:
        mean, std = final_results[key]
        plt.plot(x, mean, label=label, color=color)
        plt.fill_between(x, mean - std, mean + std, alpha=0.3, color=color)

    plt.title("Average Cumulative Rewards over Time for Different Strategies")
    plt.xlabel("Time Step")
    plt.ylabel("Cumulative Reward")
    plt.legend()
    plt.grid(True)
    plt.savefig(f'reward_plot_dim{d}_noise{noise_scale}.png')
    plt.close()

if __name__ == '__main__':
    main()
