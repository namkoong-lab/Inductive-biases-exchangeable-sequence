import torch
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('..')
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch.multiprocessing as mp
from tqdm import tqdm
from functools import partial
from LinUCB import LinUCBDisjoint
from TS_model import TS_machine
from utils.load_model import load_model

def run_exp(process_id):
    # Parameters
    alpha = 1.0
    d = 1        
    num_arms = 2
    T = 100   # Number of time steps
    batch_exp = 10
    imag_horizon = 100
    nl = False

    gpu_id = process_id % 8
    torch.cuda.set_device(gpu_id)
    # set seed
    torch.manual_seed(process_id)
    torch.cuda.manual_seed(process_id)
    torch.cuda.manual_seed_all(process_id)
    device = torch.device(f'cuda:{gpu_id}') if torch.cuda.is_available() else torch.device('cpu')

    autoreg_config_path = '../scripts/uq_normal_autoreg.yaml'
    autoreg_checkpoint_dir = '/shared/share_mala/Leon/CCB_1/UQ-normal-Gradient-full-Loss-logprob-Horizon-2000_Noise_0.1/model_checkpoint_100.pt'

    excg_config_path = '../scripts/uq_normal_excg.yaml'
    excg_checkpoint_dir = '/shared/share_mala/Leon/CCB_1/UQ-normal-Gradient-full-Loss-logprob-Horizon-2000_Noise_0.1/model_checkpoint_70.pt'

    autoreg_models = []
    for _ in range(num_arms):
        autoreg_model = load_model(autoreg_config_path, autoreg_checkpoint_dir)
        autoreg_model.to(device)
        autoreg_models.append(autoreg_model)

    excg_models = []
    for _ in range(num_arms):
        excg_model = load_model(excg_config_path, excg_checkpoint_dir)
        excg_model.to(device)
        excg_models.append(excg_model)

    pfn_models = []
    for _ in range(num_arms):
        pfn_model = load_model(excg_config_path, excg_checkpoint_dir)
        pfn_model.to(device)
        pfn_models.append(pfn_model)

    autoreg_ts_machine_list = [TS_machine(autoreg_model, device, d, imag_horizon, nl) for autoreg_model in autoreg_models]
    excg_ts_machine_list = [TS_machine(excg_model, device, d, imag_horizon, nl) for excg_model in excg_models]
    pfn_ts_machine_list = [TS_machine(pfn_model, device, d, imag_horizon, nl) for pfn_model in pfn_models]
    linucb = LinUCBDisjoint(alpha=alpha, d=d)

    # Define a true theta for each arm (for simulation purposes)
    theta_true = torch.randn(num_arms, d)
    for arm in range(num_arms):
        theta_true[arm] = theta_true[arm] / torch.norm(theta_true[arm], p='fro')
    
    # Define different noise scales for each arm
    # arm_noise_scales = torch.linspace(0.1, 1.0, num_arms)  # Creates evenly spaced noise scales from 0.1 to 1.0
    arm_noise_scales = torch.tensor([0.1, 0.5])
    
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
        feature_matrix = torch.randn(num_arms, d)
        expected_rewards = torch.tensor([theta_true[i] @ feature_matrix[i] for i in range(num_arms)])
        # Apply different noise scales for each arm
        env_rewards = expected_rewards + torch.randn(expected_rewards.shape) * arm_noise_scales

        ### Select an arm using LinUCB
        a_t = linucb.select_arm(feature_matrix)
        x_t = feature_matrix[a_t]
        r_t = env_rewards[a_t]
        linucb_batch_buffer.append((a_t, x_t, r_t))
        if t % batch_exp == 0:
            for a_t, x_t, r_t in linucb_batch_buffer:
                linucb.update(a_t, x_t, r_t)
            linucb_batch_buffer = []
        total_reward += r_t.item()
        linucb_rewards.append(total_reward)


        ### select arms with TS imagination
        reward_list = []

        ### Exploration and draw arms
        if t % batch_exp == 0 or t == 1:
            theta_pred_list = []
            for arm in range(num_arms):
                theta_pred = autoreg_ts_machine_list[arm].ts_imagination_joint()
                theta_pred_list.append(theta_pred)

        for arm in range(num_arms):
            x_t = feature_matrix[arm]
            theta_a = theta_pred_list[arm]
            predicted_rewards = autoreg_ts_machine_list[arm].predict(x_t, theta_a)
            reward_list.append(predicted_rewards)

        predicted_rewards_list = torch.stack(reward_list, dim=0)
        a_t = torch.argmax(predicted_rewards_list).item()
        total_autoreg_reward += env_rewards[a_t].item()
        autoreg_rewards.append(total_autoreg_reward)
        ts_machine_batch_buffer[a_t].append((feature_matrix[a_t], env_rewards[a_t].item()))

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
            theta_pred_list = []
            for arm in range(num_arms):
                theta_pred = excg_ts_machine_list[arm].ts_imagination_joint()
                theta_pred_list.append(theta_pred)

        for arm in range(num_arms):
            x_t = feature_matrix[arm]
            theta_a = theta_pred_list[arm]
            predicted_rewards = excg_ts_machine_list[arm].predict(x_t, theta_a)
            excg_reward_list.append(predicted_rewards)

        predicted_rewards_list = torch.stack(excg_reward_list, dim=0)
        a_t = torch.argmax(predicted_rewards_list).item()
        total_excg_reward += env_rewards[a_t].item()
        excg_rewards.append(total_excg_reward)
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
            theta_pred_list = []
            for arm in range(num_arms):
                theta_pred = pfn_ts_machine_list[arm].ts_imagination_marginal()
                theta_pred_list.append(theta_pred)

        for arm in range(num_arms):
            x_t = feature_matrix[arm]
            theta_a = theta_pred_list[arm]
            predicted_rewards = pfn_ts_machine_list[arm].predict(x_t, theta_a)
            pfn_reward_list.append(predicted_rewards)

        predicted_rewards_list = torch.stack(pfn_reward_list, dim=0)
        a_t = torch.argmax(predicted_rewards_list).item()
        total_pfn_reward += env_rewards[a_t].item()
        pfn_rewards.append(total_pfn_reward)
        pfn_ts_machine_batch_buffer[a_t].append((feature_matrix[a_t], env_rewards[a_t].item()))

        ### Batched Update
        if t % batch_exp == 0:
            for arm in range(num_arms):
                for x_t, r_t in pfn_ts_machine_batch_buffer[arm]:  
                    pfn_ts_machine_list[arm].update_batch(x_t, r_t)
            pfn_ts_machine_batch_buffer = [[] for _ in range(num_arms)] 

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

def run_wrapper(process_id):
    result = run_exp(process_id)
    return result

def main():
    num_jobs = 8
    d = 1
    num_arms = 2
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
                              ('AutoReg TS', 'blue', 'autoreg_joint_cumulative'),
                              ('ExcG TS', 'green', 'excg_cumulative'),
                              ('PFN TS', 'purple', 'pfn_cumulative'),
                              ('Oracle', 'gray', 'oracle_cumulative')]:
        mean, std = final_results[key]
        plt.plot(x, mean, label=label, color=color)
        plt.fill_between(x, mean - std, mean + std, alpha=0.3, color=color)

    plt.title("Average Cumulative Rewards over Time for Different Strategies")
    plt.xlabel("Time Step")
    plt.ylabel("Cumulative Reward")
    plt.legend()
    plt.grid(True)
    plt.savefig(f'batched_reward_plot_dim{d}_num_arms{num_arms}.png')
    plt.close()

if __name__ == '__main__':
    main()
