import torch
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
sys.path.append('..')
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch.multiprocessing as mp
from tqdm import tqdm
from functools import partial
from models.LinUCB import LinUCBDisjoint
from models.TS_model import TS_machine
from utils.load_model import load_model
import random
import pandas as pd

def run_exp(process_id, base_seed):
    # Parameters
    alpha = 1.0
    d = 1        
    num_arms = 2
    T = 100   # Number of time steps
    batch_exp = 1
    imag_horizon = 100
    nl = False
    sigma_1 = 0.5
    sigma_2 = 0.9
    tau_1 = 0.5
    tau_2 = 0.1
    checkpoint_iter = 200
    horizon = 200

    # Specify available GPU IDs
    available_gpus = [0, 1, 2, 3, 4, 5, 6, 7]  # Modify this list based on your available GPUs
    gpu_id = available_gpus[process_id % len(available_gpus)]
    torch.cuda.set_device(gpu_id)
    
    # Modify the seed calculation to ensure no overlap
    experiment_seed = base_seed + (process_id * 500)  # Each process gets its own range of 500 seeds
    
    torch.manual_seed(experiment_seed)
    torch.cuda.manual_seed(experiment_seed)
    torch.cuda.manual_seed_all(experiment_seed)
    np.random.seed(experiment_seed)
    random.seed(experiment_seed)

    device = torch.device(f'cuda:{gpu_id}') if torch.cuda.is_available() else torch.device('cpu')

    ### TODO: change the checkpoint path
    arm_1_autoreg_config_path = '../scripts/uq_normal_autoreg.yaml'
    arm_1_autoreg_checkpoint_dir = f'autoreg-Horizon_{horizon}-Sigma_{sigma_1}-Noise_{tau_1}/model_checkpoint_{checkpoint_iter}.pt'

    arm_2_autoreg_config_path = '../scripts/uq_normal_autoreg.yaml'
    arm_2_autoreg_checkpoint_dir = f'autoreg-Horizon_{horizon}-Sigma_{sigma_2}-Noise_{tau_2}/model_checkpoint_{checkpoint_iter}.pt'

    arm_1_excg_config_path = '../scripts/uq_normal_excg.yaml'
    arm_1_excg_checkpoint_dir = f'excg-Horizon_{horizon}-Sigma_{sigma_1}-Noise_{tau_1}/model_checkpoint_{checkpoint_iter}.pt'

    arm_2_excg_config_path = '../scripts/uq_normal_excg.yaml'
    arm_2_excg_checkpoint_dir = f'excg-Horizon_{horizon}-Sigma_{sigma_2}-Noise_{tau_2}/model_checkpoint_{checkpoint_iter}.pt'

    autoreg_models = [load_model(arm_1_autoreg_config_path, arm_1_autoreg_checkpoint_dir), load_model(arm_2_autoreg_config_path, arm_2_autoreg_checkpoint_dir)]
    excg_models = [load_model(arm_1_excg_config_path, arm_1_excg_checkpoint_dir), load_model(arm_2_excg_config_path, arm_2_excg_checkpoint_dir)]
    pfn_models = [load_model(arm_1_excg_config_path, arm_1_excg_checkpoint_dir), load_model(arm_2_excg_config_path, arm_2_excg_checkpoint_dir)]

    # Move models to the correct device after loading
    for model in autoreg_models + excg_models + pfn_models:
        model.to(device)

    autoreg_ts_machine_list = [TS_machine(autoreg_model, device, d, imag_horizon, nl) for autoreg_model in autoreg_models]
    excg_ts_machine_list = [TS_machine(excg_model, device, d, imag_horizon, nl) for excg_model in excg_models]
    pfn_ts_machine_list = [TS_machine(pfn_model, device, d, 1, nl) for pfn_model in pfn_models]
    linucb = LinUCBDisjoint(alpha=alpha, d=d)

    theta_true = torch.randn(num_arms, d)
    # print(f'theta_true: {theta_true}')
    scaling_factors = torch.tensor([sigma_1, sigma_2])
    # print(f'scaling_factors: {scaling_factors}')
    theta_true = theta_true * scaling_factors.unsqueeze(1)  # Broadcasting to match dimensions
    # print(f'theta_true after scaling: {theta_true}')
    max_theta_true = torch.max(theta_true)
    # print(f'max_theta_true: {max_theta_true}')
    arm_noise_scales = torch.tensor([tau_1, tau_2])
    # print(f'arm_noise_scales: {arm_noise_scales}')
    
    # Initialize cumulative rewards
    autoreg_rewards = []
    excg_rewards = []
    pfn_rewards = []
    linucb_rewards = []
    oracle_rewards = []
    total_reward = 0.0
    total_oracle_reward = 0.0
    total_autoreg_reward = 0.0
    total_excg_reward = 0.0
    total_pfn_reward = 0.0
    linucb_batch_buffer = []
    ts_machine_batch_buffer = [[] for _ in range(num_arms)]
    excg_ts_machine_batch_buffer = [[] for _ in range(num_arms)]
    pfn_ts_machine_batch_buffer = [[] for _ in range(num_arms)]
    for t in tqdm(range(1, T + 1), desc=f"Running experiment {process_id}"):
        # Simulate feature vectors for each arm
        # feature_matrix = torch.randn(num_arms, d)
        # expected_rewards = torch.tensor([theta_true[i] @ feature_matrix[i] for i in range(num_arms)])
        feature_matrix = torch.zeros(num_arms, d)
        # expected_rewards = torch.tensor([theta_true[i] + feature_matrix[i] for i in range(num_arms)])
        expected_rewards = theta_true.squeeze()
        # print(f'expected_rewards: {expected_rewards}')
        # Apply different noise scales for each arm
        noise = torch.randn(expected_rewards.shape) 
        # print(f'noise: {noise}')
        scaled_noise = noise * arm_noise_scales
        # print(f'scaled_noise: {scaled_noise}')
        env_rewards = expected_rewards + scaled_noise
        # print(f'env_rewards: {env_rewards}')

        ### Select an arm using LinUCB
        # a_t = linucb.select_arm(feature_matrix)
        # x_t = feature_matrix[a_t]
        # r_t = env_rewards[a_t]
        # linucb_batch_buffer.append((a_t, x_t, r_t))
        # if t % batch_exp == 0:
        #     for a_t, x_t, r_t in linucb_batch_buffer:
        #         linucb.update(a_t, x_t, r_t)
        #     linucb_batch_buffer = []
        # total_reward += r_t.item()
        linucb_rewards.append(0)


        ### select arms with TS imagination
        reward_list = []

        ### Exploration and draw arms
        if t % batch_exp == 0 or t == 1:
            autoreg_theta_pred_list = []
            for arm in range(num_arms):
                autoreg_theta_pred = autoreg_ts_machine_list[arm].ts_imagination_joint()
                autoreg_theta_pred_list.append(autoreg_theta_pred)

        for arm in range(num_arms):
            x_t = feature_matrix[arm]
            autoreg_theta_a = autoreg_theta_pred_list[arm]
            # print(f'autoreg_theta_a: {autoreg_theta_a}')
            predicted_rewards = autoreg_ts_machine_list[arm].predict(x_t, autoreg_theta_a)
            reward_list.append(predicted_rewards)

        predicted_rewards_list = torch.stack(reward_list, dim=0)
        # print(f'predicted_rewards_list: {predicted_rewards_list}')
        a_t = torch.argmax(predicted_rewards_list).item()
        # print(f'a_t: {a_t}')
        # print(f'predicted_rewards[a_t]: {predicted_rewards_list[a_t]}')
        # print(f'expected_rewards[a_t]: {expected_rewards[a_t]}')
        # print(f'env_rewards[a_t]: {env_rewards[a_t]}')
        # print(f'max_theta_true: {max_theta_true}')
        total_autoreg_reward += max_theta_true - expected_rewards[a_t].item()
        # print(f'total_autoreg_reward: {total_autoreg_reward}')
        autoreg_rewards.append(total_autoreg_reward.item())
        ts_machine_batch_buffer[a_t].append((feature_matrix[a_t], env_rewards[a_t].item()))


        # print("uq context of arms")
        # for arm in range(num_arms):
        #     print(f'arm {arm} uq_joint_context_x: {autoreg_ts_machine_list[arm].uq_joint_context_x}')
        #     print(f'arm {arm} uq_joint_context_y: {autoreg_ts_machine_list[arm].uq_joint_context_y}')
        # print('-------------------------------------------------')
        ### Batched Update
        if t % batch_exp == 0:
            for arm in range(num_arms):
                for x_t, r_t in ts_machine_batch_buffer[arm]:  
                    autoreg_ts_machine_list[arm].update_batch(x_t, r_t)
            ts_machine_batch_buffer = [[] for _ in range(num_arms)]

        ### excg
        excg_reward_list = []

        ### Exploration and draw arms
        if t % batch_exp == 0 or t == 1:
            excg_theta_pred_list = []
            for arm in range(num_arms):
                excg_theta_pred = excg_ts_machine_list[arm].ts_imagination_joint()
                excg_theta_pred_list.append(excg_theta_pred)

        for arm in range(num_arms):
            x_t = feature_matrix[arm]
            excg_theta_a = excg_theta_pred_list[arm]
            predicted_rewards = excg_ts_machine_list[arm].predict(x_t, excg_theta_a)
            excg_reward_list.append(predicted_rewards)

        predicted_rewards_list = torch.stack(excg_reward_list, dim=0)
        a_t = torch.argmax(predicted_rewards_list).item()
        # total_excg_reward += env_rewards[a_t].item()
        total_excg_reward += max_theta_true - expected_rewards[a_t].item()
        excg_rewards.append(total_excg_reward.item())
        excg_ts_machine_batch_buffer[a_t].append((feature_matrix[a_t], env_rewards[a_t].item()))

        ### Batched Update
        if t % batch_exp == 0:
            for arm in range(num_arms):
                for x_t, r_t in excg_ts_machine_batch_buffer[arm]:  
                    excg_ts_machine_list[arm].update_batch(x_t, r_t)
            excg_ts_machine_batch_buffer = [[] for _ in range(num_arms)]

        ### pfn
        pfn_reward_list = []

        ### Exploration and draw arms
        if t % batch_exp == 0 or t == 1:
            pfn_theta_pred_list = []
            for arm in range(num_arms):
                pfn_theta_pred = pfn_ts_machine_list[arm].ts_imagination_marginal()
                pfn_theta_pred_list.append(pfn_theta_pred)

        for arm in range(num_arms):
            x_t = feature_matrix[arm]
            pfn_theta_a = pfn_theta_pred_list[arm]
            predicted_rewards = pfn_ts_machine_list[arm].predict(x_t, pfn_theta_a)
            pfn_reward_list.append(predicted_rewards)

        predicted_rewards_list = torch.stack(pfn_reward_list, dim=0)
        # print(f'predicted_rewards_list: {predicted_rewards_list}')
        a_t = torch.argmax(predicted_rewards_list).item()
        # print(f'a_t: {a_t}')
        # total_pfn_reward += env_rewards[a_t].item()
        total_pfn_reward += max_theta_true - expected_rewards[a_t].item()
        # print(f'total_pfn_reward: {total_pfn_reward}')
        pfn_rewards.append(total_pfn_reward.item())
        pfn_ts_machine_batch_buffer[a_t].append((feature_matrix[a_t], env_rewards[a_t].item()))

        ### Batched Update
        if t % batch_exp == 0:
            for arm in range(num_arms):
                for x_t, r_t in pfn_ts_machine_batch_buffer[arm]:  
                    pfn_ts_machine_list[arm].update_batch(x_t, r_t)
            pfn_ts_machine_batch_buffer = [[] for _ in range(num_arms)] 

        # print("uq context of arms")
        # for arm in range(num_arms):
        #     print(f'arm {arm} uq_joint_context_x: {pfn_ts_machine_list[arm].uq_joint_context_x}')
        #     print(f'arm {arm} uq_joint_context_y: {pfn_ts_machine_list[arm].uq_joint_context_y}')
        # print('-------------------------------------------------')

        # Update cumulative environment reward, which is the max of the 
        oracle_a = torch.argmax(expected_rewards)
        oracle_reward = expected_rewards[oracle_a]
        total_oracle_reward += oracle_reward.item()
        oracle_rewards.append(total_oracle_reward)

    results = {
        'autoreg_joint_cumulative': autoreg_rewards,
        'oracle_cumulative': oracle_rewards,
        'linucb_cumulative': linucb_rewards,
        'excg_cumulative': excg_rewards,
        'pfn_cumulative': pfn_rewards,
    }

    return results

def run_wrapper(args):
    process_id, base_seed = args
    result = run_exp(process_id, base_seed)
    return result

def main():
    num_jobs = 100
    total_experiments = 10  # or whatever number we set
    total_runs = num_jobs * total_experiments

    # Store all results for final analysis
    all_data = {
        'excg_cumulative': [],  # Will be a list of lists, each inner list is one complete run
        'pfn_cumulative': []
    }

    for exp_num in range(total_experiments):
        base_seed = 1000 + (exp_num * num_jobs)  # Each experiment gets its own range of seeds
        print(f"\nRunning Experiment Set {exp_num + 1}/{total_experiments} with base_seed {base_seed}")
        
        # Create output directory for this experiment
        output_dir = os.path.join('outputs/all_experiments', f'experiment_seed_{base_seed}')
        os.makedirs(output_dir, exist_ok=True)

        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=num_jobs) as pool:
            args_list = [(i, base_seed) for i in range(num_jobs)]
            all_results = list(tqdm(pool.imap(run_wrapper, args_list), 
                                  total=num_jobs, 
                                  desc="Running parallel jobs"))

        # Combine results from all jobs
        combined_results = {key: [] for key in all_results[0].keys()}
        for result in all_results:
            for key in result:
                combined_results[key].append(result[key])

        # Store raw results from each batch
        for key in ['excg_cumulative', 'pfn_cumulative']:
            all_data[key].extend(combined_results[key])  # Add all num_jobs batch results

    # Calculate final statistics
    for key in ['excg_cumulative', 'pfn_cumulative']:
        data_array = np.array(all_data[key])  # Shape: (total_runs, num_timesteps)
        mean = np.mean(data_array, axis=0)
        std = np.std(data_array, axis=0)
        ci = 1.96 * (std / np.sqrt(total_runs))  # Using total_runs = num_jobs * total_experiments
        
        # Save final statistics to CSV
        stats_df = pd.DataFrame({
            'timestep': range(len(mean)),
            'mean': mean,
            'std': std,
            'ci_lower': mean - ci,
            'ci_upper': mean + ci
        })
        stats_df.to_csv(os.path.join('outputs/all_experiments', f'{key}_final_statistics.csv'), index=False)

        # Save all raw data
        raw_data = []
        for run_idx, run_data in enumerate(all_data[key]):
            for t, value in enumerate(run_data):
                raw_data.append({
                    'run': run_idx,
                    'timestep': t,
                    'value': value
                })
        pd.DataFrame(raw_data).to_csv(os.path.join('outputs/all_experiments', f'{key}_all_raw_data.csv'), index=False)

    # Create final plot using the calculated statistics
    plt.figure(figsize=(12, 6))
    
    colors = {
        'excg_cumulative': '#2ca02c',  # deep green
        'pfn_cumulative': '#9467bd'    # purple
    }

    labels = {
        'excg_cumulative': 'Multi Step',
        'pfn_cumulative': 'One Step'
    }

    for key in ['excg_cumulative', 'pfn_cumulative']:
        data_array = np.array(all_data[key])
        mean = np.mean(data_array, axis=0)
        std = np.std(data_array, axis=0)
        ci = 1.96 * (std / np.sqrt(total_runs))
        
        x = np.arange(len(mean))
        plt.plot(x, mean, label=labels[key], color=colors[key], linewidth=2.5)
        plt.fill_between(x, mean - ci, mean + ci, alpha=0.15, color=colors[key])

    plt.title(f"Average Cumulative Regret over Time ({total_runs} Total Runs)")
    plt.xlabel("Time Step")
    plt.ylabel("Cumulative Regret")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the final plot
    plt.savefig(os.path.join('outputs/all_experiments', 'final_aggregated_plot.png'), dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    main()
