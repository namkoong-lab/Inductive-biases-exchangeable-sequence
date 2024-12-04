import sys
import torch
import matplotlib.pyplot as plt 
import yaml
import numpy as np
import random
from tqdm import tqdm
import matplotlib.gridspec as gridspec
from torch.distributions import Normal
from torch.utils.data import DataLoader
import gpytorch
from types import SimpleNamespace

sys.path.append('..')
from utils.load_model import load_model
from data.load_data import GPSamplerConstantDataset, scalar_collate_fn

class CustomizableGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, mean_module, base_kernel, likelihood):
        super(CustomizableGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = mean_module
        self.covar_module = base_kernel
        self.likelihood = likelihood

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(self.mean_module(x), self.covar_module(x))

class JointPredictionMachine:
    def __init__(self, uq_model, device):
        self.uq_model = uq_model
        self.device = device

    def joint_prediction(self, xc_data, yc_data, xt_data, T):
        inner_context_x = []
        inner_context_y = []
        device = self.device
        uq_model = self.uq_model
        batch_size = xt_data.shape[0]
        D = xt_data.shape[2]

        with torch.no_grad():
            for i in range(T):
                target_x = xt_data[:, i, :].unsqueeze(1)
                
                inner_xc = torch.cat(inner_context_x, dim=1) if inner_context_x else torch.empty(batch_size, 0, D, device=device)
                full_xc = torch.cat([xc_data, inner_xc], dim=1) 
                inner_yc = torch.cat(inner_context_y, dim=1) if inner_context_y else torch.empty(batch_size, 0, 1, device=device)
                full_yc = torch.cat([yc_data, inner_yc], dim=1) 
                
                full_xc = full_xc.to(device)
                full_yc = full_yc.to(device)
                
                batch = SimpleNamespace(
                    xc=full_xc,
                    yc=full_yc,
                    xt=target_x,
                    yt=torch.zeros(batch_size, 1, 1, device=device)
                )

                predicted_rewards = uq_model.predict(batch)
                inner_context_x.append(target_x)
                inner_context_y.append(predicted_rewards)

        X = torch.cat(inner_context_x, dim=1)
        y = torch.cat(inner_context_y, dim=1)

        return X, y

    def marginal_prediction(self, xc_data, yc_data, xt_data, T):
        device = self.device
        uq_model = self.uq_model
        batch_size = xt_data.shape[0]

        with torch.no_grad():
            batch = SimpleNamespace(
                xc=xc_data,
                yc=yc_data,
                xt=xt_data,
                yt=torch.zeros(batch_size, T, 1, device=device)
            )
            predicted_rewards = uq_model.predict(batch)

        return xt_data, predicted_rewards

    def permutation_prediction(self, eval_batch, P, T, marginal=False):
        xc_data = eval_batch.xc
        yc_data = eval_batch.yc
        xt_raw = eval_batch.xt
        B = xt_raw.shape[0]

        y_P_BxTxD = torch.zeros(P, B, xt_raw.shape[1], yc_data.shape[2], device=self.device)

        for p in range(P):
            perm_indices = torch.randperm(xt_raw.shape[1])
            xt_data = xt_raw[:, perm_indices, :]
            
            if marginal:
                X, y = self.marginal_prediction(xc_data, yc_data, xt_data, T)
            else:
                X, y = self.joint_prediction(xc_data, yc_data, xt_data, T)
                
            y = y[:, perm_indices.argsort(), :]
            y = y.unsqueeze(0)
            y_P_BxTxD[p, :, :, :] = y

        return y_P_BxTxD

def oracle_model_predict(oracle_model, likelihood, xc, yc, xt):
    oracle_model.to(device)
    likelihood.to(device)
    oracle_model.set_train_data(inputs=xc, targets=yc, strict=False)

    oracle_model.eval()
    likelihood.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        posterior = likelihood(oracle_model(xt))
        posterior_mean = posterior.mean
        posterior_var = posterior.variance
        return posterior_mean, posterior_var

def oracle_model_predict_B(oracle_model, likelihood, xc, yc, xt, B):
    means_B_oracle = torch.zeros(B, xt.shape[1])
    vars_B_oracle = torch.zeros(B, xt.shape[1])
    
    for b in range(B):
        means_B_oracle[b, :], vars_B_oracle[b, :] = oracle_model_predict(
            oracle_model, likelihood, xc[b, :, :], yc[b, :, 0], xt[b, :, :]
        )
    return means_B_oracle, vars_B_oracle

def estimating_losses(batch, B, N, T, P, D, y_P_BxTxD_list, context_continuous, oracle_model, likelihood):
    P = y_P_BxTxD_list[0].shape[0]
    a1_loss_list = []
    a2_loss_list = []

    for y_P_BxTxD in tqdm(y_P_BxTxD_list, desc="Processing predictions"):
        a1_loss = 0
        a2_loss = 0

        for p in tqdm(range(P), desc="Processing permutations", leave=False):
            yt_true_oracle_BxT = torch.zeros(B, T).to(device)

            for t in range(T):
                if context_continuous:
                    curr_xc_BxNptxD = torch.cat([batch.xc, batch.xt[:, :t, :]], dim=1)
                    curr_yc_BxNpt = torch.cat([batch.yc, y_P_BxTxD[p, :, :t, :]], dim=1).squeeze(2)
                else:
                    curr_xc_BxNptxD = batch.xc
                    curr_yc_BxNpt = batch.yc.squeeze(2)

                curr_xt_BxD = batch.xt[:, t, :]
                curr_yt_tnp_B = y_P_BxTxD[p, :, t, 0].to(device)

                means_B_oracle, vars_B_oracle = oracle_model_predict_B(
                    oracle_model,
                    likelihood,
                    curr_xc_BxNptxD,
                    curr_yc_BxNpt.unsqueeze(2),
                    curr_xt_BxD.unsqueeze(1),
                    B
                )
                
                means_B_oracle = means_B_oracle.squeeze(1).to(device)
                sds_B_oracle = torch.sqrt(vars_B_oracle).squeeze(1).to(device)

                distributions = Normal(means_B_oracle, sds_B_oracle)
                log_probs = distributions.log_prob(curr_yt_tnp_B)
                a1_loss += log_probs.mean()

                samples_B = torch.normal(means_B_oracle, sds_B_oracle)
                log_probs = distributions.log_prob(samples_B)
                a2_loss += log_probs.mean()

        a1_loss_list.append(a1_loss.detach().cpu() / (P * T))
        a2_loss_list.append(a2_loss.detach().cpu() / (P * T))

    # Oracle Baseline (Single Trajectory)
    b_loss = 0
    yt_true_oracle_BxT = torch.zeros(B, T).to(device)
    
    for t in tqdm(range(T), desc="Computing oracle baseline"):
        if context_continuous:
            curr_xc_BxNptxD = torch.cat([batch.xc, batch.xt[:, :t, :]], dim=1)
            curr_yc_oracle_BxNpt = torch.cat([batch.yc.squeeze(2), yt_true_oracle_BxT[:, :t]], dim=1)
        else:
            curr_xc_BxNptxD = batch.xc
            curr_yc_oracle_BxNpt = batch.yc.squeeze(2)

        curr_xt_BxD = batch.xt[:, t, :]

        means_B_oracle_oracle, vars_B_oracle_oracle = oracle_model_predict_B(
            oracle_model,
            likelihood,
            curr_xc_BxNptxD,
            curr_yc_oracle_BxNpt.unsqueeze(2),
            curr_xt_BxD.unsqueeze(1),
            B
        )
        means_B_oracle_oracle = means_B_oracle_oracle.squeeze(1)
        sds_B_oracle_oracle = torch.sqrt(vars_B_oracle_oracle).squeeze(1)

        distributions = Normal(means_B_oracle_oracle, sds_B_oracle_oracle)
        samples_B = torch.normal(means_B_oracle_oracle, sds_B_oracle_oracle)
        yt_true_oracle_BxT[:, t] = samples_B
        log_probs = distributions.log_prob(samples_B)
        b_loss += log_probs.mean()
        
    b_loss = b_loss.detach().cpu() / T

    return a1_loss_list, a2_loss_list, b_loss

def create_batch(original_batch, N, T):
    return SimpleNamespace(
        xc=original_batch.x[:, :N, :].clone().to(device),
        yc=original_batch.y[:, :N, :].clone().to(device),
        xt=original_batch.x[:, N:T+N, :].clone().to(device),
        yt=original_batch.y[:, N:T+N, :].clone().to(device)
    )

def collect_joint_context_data(eval_batch, start, end, step):
    T = 10
    auto_reg_a1_list = []
    excg_a1_list = []
    excg_marginal_a1_list = []
    oracle_list = []
    auto_reg_a2_list = []
    excg_a2_list = []
    excg_marginal_a2_list = []
    
    for n in tqdm(range(start, end, step)):
        batch = create_batch(eval_batch, n, T)

        auto_reg_y_P_BxTxD = auto_reg_joint_prediction_machine.permutation_prediction(batch, P, T, marginal=False)
        excg_y_P_BxTxD = excg_joint_prediction_machine.permutation_prediction(batch, P, T, marginal=False)
        excg_marginal_y_P_BxTxD = excg_joint_prediction_machine.permutation_prediction(batch, P, T, marginal=True)

        y_P_BxTxD_list = [auto_reg_y_P_BxTxD, excg_y_P_BxTxD, excg_marginal_y_P_BxTxD]

        a1_list, a2_list, oracle_loss = estimating_losses(
            batch, B, n, T, P, D, y_P_BxTxD_list, context_continuous,
            oracle_model, likelihood)
        
        auto_reg_a1_list.append(a1_list[0])
        excg_a1_list.append(a1_list[1])
        excg_marginal_a1_list.append(a1_list[2])
        oracle_list.append(oracle_loss)
        auto_reg_a2_list.append(a2_list[0])
        excg_a2_list.append(a2_list[1])
        excg_marginal_a2_list.append(a2_list[2])

    return auto_reg_a1_list, excg_a1_list, excg_marginal_a1_list, oracle_list, auto_reg_a2_list, excg_a2_list, excg_marginal_a2_list

def collect_joint_target_data(eval_batch, start, end, step):
    N = 10
    auto_reg_a1_list = []
    excg_a1_list = []
    excg_marginal_a1_list = []
    oracle_list = []
    auto_reg_a2_list = []
    excg_a2_list = []
    excg_marginal_a2_list = []
    
    for t in tqdm(range(start, end, step)):
        batch = create_batch(eval_batch, N, t)

        auto_reg_y_P_BxTxD = auto_reg_joint_prediction_machine.permutation_prediction(batch, P, t, marginal=False)
        excg_y_P_BxTxD = excg_joint_prediction_machine.permutation_prediction(batch, P, t, marginal=False)
        excg_marginal_y_P_BxTxD = excg_joint_prediction_machine.permutation_prediction(batch, P, t, marginal=True)

        y_P_BxTxD_list = [auto_reg_y_P_BxTxD, excg_y_P_BxTxD, excg_marginal_y_P_BxTxD]

        a1_list, a2_list, oracle_loss = estimating_losses(
            batch, B, N, t, P, D, y_P_BxTxD_list, context_continuous,
            oracle_model, likelihood)
        
        auto_reg_a1_list.append(a1_list[0])
        excg_a1_list.append(a1_list[1])
        excg_marginal_a1_list.append(a1_list[2])
        oracle_list.append(oracle_loss)
        auto_reg_a2_list.append(a2_list[0])
        excg_a2_list.append(a2_list[1])
        excg_marginal_a2_list.append(a2_list[2])
    
    return auto_reg_a1_list, excg_a1_list, excg_marginal_a1_list, oracle_list, auto_reg_a2_list, excg_a2_list, excg_marginal_a2_list

def collect_marginal_context_data(eval_batch, start, end, step):
    T = 1
    auto_reg_a1_list = []
    excg_a1_list = []
    excg_marginal_a1_list = []
    oracle_list = []
    auto_reg_a2_list = []
    excg_a2_list = []
    excg_marginal_a2_list = []
    
    for n in tqdm(range(start, end, step)):
        batch = create_batch(eval_batch, n, T)

        auto_reg_y_P_BxTxD = auto_reg_joint_prediction_machine.permutation_prediction(batch, P, T, marginal=False)
        excg_y_P_BxTxD = excg_joint_prediction_machine.permutation_prediction(batch, P, T, marginal=False)
        excg_marginal_y_P_BxTxD = excg_joint_prediction_machine.permutation_prediction(batch, P, T, marginal=True)

        y_P_BxTxD_list = [auto_reg_y_P_BxTxD, excg_y_P_BxTxD, excg_marginal_y_P_BxTxD]

        a1_list, a2_list, oracle_loss = estimating_losses(
            batch, B, n, T, P, D, y_P_BxTxD_list, context_continuous,
            oracle_model, likelihood)
        
        auto_reg_a1_list.append(a1_list[0])
        excg_a1_list.append(a1_list[1])
        excg_marginal_a1_list.append(a1_list[2])
        oracle_list.append(oracle_loss)
        auto_reg_a2_list.append(a2_list[0])
        excg_a2_list.append(a2_list[1])
        excg_marginal_a2_list.append(a2_list[2])
    
    return auto_reg_a1_list, excg_a1_list, excg_marginal_a1_list, oracle_list, auto_reg_a2_list, excg_a2_list, excg_marginal_a2_list

def create_combined_plots(start, end, step, joint_context_data, joint_target_data, marginal_context_data):
    """Creates combined plots for perplexity and difference analysis."""
    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(2, 3, figure=fig)
    
    # Unpack all data
    (auto_reg_a1_list_jc, excg_a1_list_jc, excg_marginal_a1_list_jc, 
     oracle_list_jc, auto_reg_a2_list_jc, excg_a2_list_jc, 
     excg_marginal_a2_list_jc) = joint_context_data
    
    (auto_reg_a1_list_jt, excg_a1_list_jt, excg_marginal_a1_list_jt,
     oracle_list_jt, auto_reg_a2_list_jt, excg_a2_list_jt,
     excg_marginal_a2_list_jt) = joint_target_data
    
    (auto_reg_a1_list_mc, excg_a1_list_mc, excg_marginal_a1_list_mc,
     oracle_list_mc, auto_reg_a2_list_mc, excg_a2_list_mc,
     excg_marginal_a2_list_mc) = marginal_context_data

    # Calculate differences for difference plots
    diff_configs = {
        'jc': {
            'auto_reg': [(b - a) for a, b in zip(auto_reg_a1_list_jc, oracle_list_jc)],
            'excg': [(b - a) for a, b in zip(excg_a1_list_jc, oracle_list_jc)],
            'pfn': [(b - a) for a, b in zip(excg_marginal_a1_list_jc, oracle_list_jc)],
            'auto_reg_a2': [(b - a) for a, b in zip(auto_reg_a2_list_jc, oracle_list_jc)],
            'excg_a2': [(b - a) for a, b in zip(excg_a2_list_jc, oracle_list_jc)],
            'pfn_a2': [(b - a) for a, b in zip(excg_marginal_a2_list_jc, oracle_list_jc)]
        },
        'jt': {
            'auto_reg': [(b - a) for a, b in zip(auto_reg_a1_list_jt, oracle_list_jt)],
            'excg': [(b - a) for a, b in zip(excg_a1_list_jt, oracle_list_jt)],
            'pfn': [(b - a) for a, b in zip(excg_marginal_a1_list_jt, oracle_list_jt)],
            'auto_reg_a2': [(b - a) for a, b in zip(auto_reg_a2_list_jt, oracle_list_jt)],
            'excg_a2': [(b - a) for a, b in zip(excg_a2_list_jt, oracle_list_jt)],
            'pfn_a2': [(b - a) for a, b in zip(excg_marginal_a2_list_jt, oracle_list_jt)]
        },
        'mc': {
            'auto_reg': [(b - a) for a, b in zip(auto_reg_a1_list_mc, oracle_list_mc)],
            'excg': [(b - a) for a, b in zip(excg_a1_list_mc, oracle_list_mc)],
            'pfn': [(b - a) for a, b in zip(excg_marginal_a1_list_mc, oracle_list_mc)],
            'auto_reg_a2': [(b - a) for a, b in zip(auto_reg_a2_list_mc, oracle_list_mc)],
            'excg_a2': [(b - a) for a, b in zip(excg_a2_list_mc, oracle_list_mc)],
            'pfn_a2': [(b - a) for a, b in zip(excg_marginal_a2_list_mc, oracle_list_mc)]
        }
    }

    # Define all plot configurations
    plot_configs = [
        # Top row - Perplexity plots
        {
            'data': [
                (auto_reg_a1_list_jc, "AutoReg a1", 'green', '-'),
                (excg_a1_list_jc, "ExcG a1", 'blue', '-'),
                (excg_marginal_a1_list_jc, "PFN a1", 'red', '-'),
                (oracle_list_jc, "Oracle", 'black', '-'),
                (auto_reg_a2_list_jc, "AutoReg a2", 'green', '--'),
                (excg_a2_list_jc, "ExcG a2", 'blue', '--'),
                (excg_marginal_a2_list_jc, "PFN a2", 'red', '--')
            ],
            'title': 'Joint Prediction: Fixed Target (10)\nContext Points vs Perplexity',
            'xlabel': 'Number of Context Points',
            'ylabel': 'Perplexity',
            'position': (0, 0)
        },
        {
            'data': [
                (auto_reg_a1_list_jt, "AutoReg a1", 'green', '-'),
                (excg_a1_list_jt, "ExcG a1", 'blue', '-'),
                (excg_marginal_a1_list_jt, "PFN a1", 'red', '-'),
                (oracle_list_jt, "Oracle", 'black', '-'),
                (auto_reg_a2_list_jt, "AutoReg a2", 'green', '--'),
                (excg_a2_list_jt, "ExcG a2", 'blue', '--'),
                (excg_marginal_a2_list_jt, "PFN a2", 'red', '--')
            ],
            'title': 'Joint Prediction: Fixed Context (10)\nTarget Points vs Perplexity',
            'xlabel': 'Number of Target Points',
            'ylabel': 'Perplexity',
            'position': (0, 1)
        },
        {
            'data': [
                (auto_reg_a1_list_mc, "AutoReg a1", 'green', '-'),
                (excg_a1_list_mc, "ExcG a1", 'blue', '-'),
                (excg_marginal_a1_list_mc, "PFN a1", 'red', '-'),
                (oracle_list_mc, "Oracle", 'black', '-'),
                (auto_reg_a2_list_mc, "AutoReg a2", 'green', '--'),
                (excg_a2_list_mc, "ExcG a2", 'blue', '--'),
                (excg_marginal_a2_list_mc, "PFN a2", 'red', '--')
            ],
            'title': 'Marginal Prediction: Fixed Target (1)\nContext Points vs Perplexity',
            'xlabel': 'Number of Context Points',
            'ylabel': 'Perplexity',
            'position': (0, 2)
        },
        # Bottom row - Difference plots
        {
            'data': [
                (diff_configs['jc']['auto_reg'], "AutoReg Diff", 'green', '-'),
                (diff_configs['jc']['excg'], "ExcG Diff", 'blue', '-'),
                (diff_configs['jc']['pfn'], "PFN Diff", 'red', '-'),
                (diff_configs['jc']['auto_reg_a2'], "AutoReg a2 Diff", 'green', '--'),
                (diff_configs['jc']['excg_a2'], "ExcG a2 Diff", 'blue', '--'),
                (diff_configs['jc']['pfn_a2'], "PFN a2 Diff", 'red', '--')
            ],
            'title': 'Joint Prediction: Fixed Target (10)\nDifference from Oracle',
            'xlabel': 'Number of Context Points',
            'ylabel': 'Perplexity Difference',
            'position': (1, 0)
        },
        {
            'data': [
                (diff_configs['jt']['auto_reg'], "AutoReg Diff", 'green', '-'),
                (diff_configs['jt']['excg'], "ExcG Diff", 'blue', '-'),
                (diff_configs['jt']['pfn'], "PFN Diff", 'red', '-'),
                (diff_configs['jt']['auto_reg_a2'], "AutoReg a2 Diff", 'green', '--'),
                (diff_configs['jt']['excg_a2'], "ExcG a2 Diff", 'blue', '--'),
                (diff_configs['jt']['pfn_a2'], "PFN a2 Diff", 'red', '--')
            ],
            'title': 'Joint Prediction: Fixed Context (10)\nDifference from Oracle',
            'xlabel': 'Number of Target Points',
            'ylabel': 'Perplexity Difference',
            'position': (1, 1)
        },
        {
            'data': [
                (diff_configs['mc']['auto_reg'], "AutoReg Diff", 'green', '-'),
                (diff_configs['mc']['excg'], "ExcG Diff", 'blue', '-'),
                (diff_configs['mc']['pfn'], "PFN Diff", 'red', '-'),
                (diff_configs['mc']['auto_reg_a2'], "AutoReg a2 Diff", 'green', '--'),
                (diff_configs['mc']['excg_a2'], "ExcG a2 Diff", 'blue', '--'),
                (diff_configs['mc']['pfn_a2'], "PFN a2 Diff", 'red', '--')
            ],
            'title': 'Marginal Prediction: Fixed Target (1)\nDifference from Oracle',
            'xlabel': 'Number of Context Points',
            'ylabel': 'Perplexity Difference',
            'position': (1, 2)
        }
    ]

    # Create all plots
    for config in plot_configs:
        ax = fig.add_subplot(gs[config['position'][0], config['position'][1]])
        for data, label, color, style in config['data']:
            ax.plot(range(start, end, step), data, label=label, c=color, linewidth=2, linestyle=style)
        ax.set_title(config['title'], fontsize=12)
        ax.set_xlabel(config['xlabel'], fontsize=10)
        ax.set_ylabel(config['ylabel'], fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()

    plt.tight_layout()
    plt.savefig(f'model_checkpoint_{Checkpoint_iter}.png', bbox_inches='tight', dpi=300)

def main():
    # Set seeds for reproducibility
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # Device configuration
    global device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Model parameters
    global Checkpoint_iter, dim_llm_embedding, B, P, D
    Checkpoint_iter = 400
    dim_llm_embedding = 4
    B = 512  # batch size
    P = 50  # number of permutations
    D = 4   # dimension
    
    # Load models
    AUTO_REG_CKPT_PATH = f"/shared/share_mala/Leon/GP_{dim_llm_embedding}/autoreg-UQ-normal-Gradient-full-Loss-logprob-Horizon-2000_Noise_0.1/model_checkpoint_{Checkpoint_iter}.pt"
    EXCG_CKPT_PATH = f"/shared/share_mala/Leon/GP_{dim_llm_embedding}/excg-UQ-normal-Gradient-full-Loss-logprob-Horizon-2000_Noise_0.1/model_checkpoint_{Checkpoint_iter}.pt"
    Auto_reg_config_path = f"../scripts/gp_uq_normal_autoreg.yaml"
    Excg_config_path = f"../scripts/gp_uq_normal_excg.yaml"

    print("Loading models...")
    with tqdm(total=2, desc="Loading models") as pbar:
        auto_reg_model = load_model(Auto_reg_config_path, AUTO_REG_CKPT_PATH, "autoreg", device)
        pbar.update(1)
        excg_model = load_model(Excg_config_path, EXCG_CKPT_PATH, "excg", device)
        pbar.update(1)
    
    # Set up oracle model
    mean_module = gpytorch.means.ConstantMean()
    base_kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
    global likelihood
    likelihood = gpytorch.likelihoods.GaussianLikelihood()

    mean_module.constant = 0.0  # mean_constant
    base_kernel.base_kernel.lengthscale = 1.0  # lengthscale
    base_kernel.outputscale = 1.0  # outputscale
    likelihood.noise_covar.noise = 0.1  # noise

    global oracle_model
    oracle_model = CustomizableGPModel(None, None, mean_module, base_kernel, likelihood)
    
    # Initialize prediction machines
    global auto_reg_joint_prediction_machine, excg_joint_prediction_machine, context_continuous
    auto_reg_joint_prediction_machine = JointPredictionMachine(auto_reg_model, device)
    excg_joint_prediction_machine = JointPredictionMachine(excg_model, device)
    context_continuous = True
    
    # Create dataset
    test_horizon = 2000
    num_test_samples = 8192
    noise = 0.1
    
    print("Creating dataset...")
    with tqdm(total=2, desc="Creating dataset") as pbar:
        eval_dataset = GPSamplerConstantDataset(
            num_samples=num_test_samples,
            noise=noise,
            dimension=dim_llm_embedding,
            horizon=test_horizon
        )
        pbar.update(1)
        
        eval_data_loader = DataLoader(eval_dataset, batch_size=B, shuffle=True, collate_fn=scalar_collate_fn)
        eval_batch = next(iter(eval_data_loader))
        pbar.update(1)
    
    # Run experiments
    start, end, step = 1, 31, 5
    
    print("\nRunning experiments...")
    with tqdm(total=3, desc="Collecting data") as pbar:
        print("\nCollecting Joint Context Data...")
        joint_context_data = collect_joint_context_data(eval_batch, start, end, step)
        pbar.update(1)
        
        print("\nCollecting Joint Target Data...")
        joint_target_data = collect_joint_target_data(eval_batch, start, end, step)
        pbar.update(1)
        
        print("\nCollecting Marginal Context Data...")
        marginal_context_data = collect_marginal_context_data(eval_batch, start, end, step)
        pbar.update(1)
    
    print("\nCreating plots...")
    create_combined_plots(start, end, step, joint_context_data, joint_target_data, marginal_context_data)
    print("\nDone! Plots saved.")

if __name__ == "__main__":
    main()