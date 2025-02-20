import argparse
import json
import os
import sys
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

sys.path.append("..")
import random
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import torch.multiprocessing as mp
from torch.distributions.normal import Normal
from tqdm import tqdm

from models.LinUCB import LinUCBDisjoint
from models.TS_model import TS_machine
from utils.load_model import load_model


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_type", type=str, default="al_regions_v2")
    parser.add_argument("--num_train", type=int, default=5)
    parser.add_argument("--num_pool", type=int, default=95)
    parser.add_argument("--max_choice", type=int, default=20)
    parser.add_argument("--num_test", type=int, default=100)
    parser.add_argument("--num_rollouts", type=int, default=20)
    parser.add_argument("--rollout_length", type=int, default=20)
    parser.add_argument("--model_epoch", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--load_eu", type=bool, default=False)
    parser.add_argument("--new_max_choice", type=int, default=50)
    return parser.parse_args()


def generate_al_data(context_dim, horizon, noise_std):
    context = torch.randn(horizon, context_dim)
    theta_true = torch.randn(1, context_dim)
    y = torch.matmul(context, theta_true.T)

    noise = torch.norm(context, dim=1) * noise_std
    noisy_y = y + noise

    return context, noisy_y


def get_mean_std_prediction(model, batch):
    with torch.no_grad():
        xt = batch.xt
        device = next(model.encoder.parameters()).device
        embeddings = model.construct_prediction_input(batch)
        mask = model.create_prediction_mask(batch)
        mask = mask.to(embeddings.dtype)
        mask = mask.to(device)
        encoding = model.encoder(embeddings, mask=mask)
        prediction = encoding[:, -xt.shape[1] :, :]
        mean = model.mean_predictor(prediction)
        std = model.std_predictor(prediction)

        if model.bound_std:
            std = 0.05 + 0.95 * F.softplus(std)
        else:
            std = std = torch.exp(std)

        pred_dist = Normal(mean, std)
        sample = pred_dist.sample()
    return mean, std, sample


def compute_rollout_uncertainty(
    model,
    train_context,
    train_y,
    pool_context,
    num_pool,
    num_rollouts,
    rollout_length,
    device,
):
    rollout_history = torch.zeros(
        (train_context.shape[0], num_pool, num_rollouts, rollout_length, 3)
    )

    epistemic_uncertainty = torch.zeros(train_context.shape[0], num_pool)
    for i in tqdm(range(num_pool), desc="Iterating over pool", position=0):
        converged_means = torch.zeros(train_context.shape[0], num_rollouts)
        for j in tqdm(
            range(num_rollouts), desc="Iterating over rollouts", position=1, leave=False
        ):
            cur_context, cur_y = train_context, train_y
            for k in tqdm(
                range(rollout_length), desc="Rolling out", position=2, leave=False
            ):
                batch = SimpleNamespace(
                    xc=cur_context,
                    yc=cur_y,
                    xt=pool_context[:, i : i + 1, :],
                    yt=torch.zeros(pool_context.shape[0], 1, 1, device=device),
                )
                mean, std, sample = get_mean_std_prediction(model, batch)

                # Record history
                rollout_history[:, i, j, k, 0] = mean.flatten()
                rollout_history[:, i, j, k, 1] = std.flatten()
                rollout_history[:, i, j, k, 2] = sample.flatten()

                cur_context = torch.cat(
                    [cur_context, pool_context[:, i : i + 1, :]], dim=1
                )
                cur_y = torch.cat([cur_y, sample], dim=1)
            converged_means[:, j] = mean.flatten()

        epistemic_uncertainty[:, i] = torch.std(converged_means, dim=1).flatten()
    return epistemic_uncertainty, rollout_history


def compute_oneshot_uncertainty(model, train_context, train_y, pool_context, num_pool):
    epistemic_uncertainty = torch.zeros(train_context.shape[0], num_pool)
    for i in range(num_pool):
        batch = SimpleNamespace(
            xc=train_context,
            yc=train_y,
            xt=pool_context[:, i : i + 1, :],
            yt=torch.zeros(pool_context.shape[0], 1, 1, device=pool_context.device),
        )
        _, std, _ = get_mean_std_prediction(model, batch)
        epistemic_uncertainty[:, i] = std.flatten()
    return epistemic_uncertainty


def eval_model(model, context, y, test_context, test_y):
    batch = SimpleNamespace(
        xc=context, yc=y, xt=test_context, yt=torch.zeros_like(test_y)
    )
    mean, std, sample = get_mean_std_prediction(model, batch)
    pred_dist = Normal(mean, std)
    return -pred_dist.log_prob(test_y)


def pick_context(context, y, idx):
    idx = idx.to(context.device)
    idx = idx.unsqueeze(-1).expand(-1, -1, context.shape[-1])
    return torch.gather(context, 1, idx), torch.gather(y, 1, idx)


def get_file_folder(args):
    save_folder = None
    save_folder = os.path.join(
        save_folder,
        f"num_train={args.num_train}_num_pool={args.num_pool}_max-choice={args.max_choice}_num_test={args.num_test}_num_rollouts={args.num_rollouts}_rollout_length={args.rollout_length}_model-epoch={args.model_epoch}_seed={args.seed}",
    )
    return save_folder


def run_exp(
    data_type,
    num_train,
    num_pool,
    max_choice,
    num_test,
    num_rollouts,
    rollout_length,
    epoch,
    seed,
    gpu_id,
    args,
):
    torch.cuda.set_device(gpu_id)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = (
        torch.device(f"cuda:{gpu_id}")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    # Load Models
    model_path = f"/shared/share_mala/tyen/seqb/dataset={data_type}_1"
    autoreg_config_path = "../scripts/uq_al-matmul_autoreg.yaml"
    autoreg_checkpoint_dir = (
        f"{model_path}/Model-autoreg-Horizon-200_Noise_0.1/model_checkpoint_{epoch}.pt"
    )

    excg_config_path = "../scripts/uq_al-matmul_excg.yaml"
    excg_checkpoint_dir = (
        f"{model_path}/Model-excg-Horizon-200_Noise_0.1/model_checkpoint_{epoch}.pt"
    )

    print(
        f"Loading models from checkpoint:"
        f"\n{autoreg_checkpoint_dir}\n{excg_checkpoint_dir}"
    )
    autoreg_model = load_model(autoreg_config_path, autoreg_checkpoint_dir)
    autoreg_model.to(device)

    excg_model = load_model(excg_config_path, excg_checkpoint_dir)
    excg_model.to(device)
    print("Models loaded")

    # Load data
    DATA_FOLDER = None

    # Taking data that model has not seen
    #   dim: [NUM_SAMPLES, [Train, Pool, Test], context_dim]
    contexts = torch.load(f"{DATA_FOLDER}/test_contexts.pt", weights_only=True)
    y = torch.load(f"{DATA_FOLDER}/test_y.pt", weights_only=True)
    print(
        f"Data loaded. Context shape: {contexts.shape}. y shape: {y.shape}",
    )

    train_context = contexts[:, :num_train, :]
    train_y = y[:, :num_train]
    pool_context = contexts[:, num_train : num_train + num_pool, :]
    pool_y = y[:, num_train : num_train + num_pool]
    test_context = contexts[:, num_train + num_pool :, :]
    test_y = y[:, num_train + num_pool :]

    # Rank pool data by epistemic uncertainty
    train_context = train_context.to(device)
    train_y = train_y.to(device)
    pool_context = pool_context.to(device)
    pool_y = pool_y.to(device)

    if not args.load_eu:
        # Do epistemic uncertainty for both models
        print("Computing autoregressive rollout uncertainty")
        autoreg_rollout_eu, autoreg_rollout_history = compute_rollout_uncertainty(
            autoreg_model,
            train_context,
            train_y,
            pool_context,
            num_pool,
            num_rollouts,
            rollout_length,
            device,
        )
        print("Autoreg rollout uncertainty computed")

        print("Computing autoregressive oneshot uncertainty")
        autoreg_oneshot_eu = compute_oneshot_uncertainty(
            autoreg_model, train_context, train_y, pool_context, num_pool
        )
        print("Autoreg uncertainty computed")

        print("Computing excg rollout uncertainty")
        excg_rollout_eu, excg_rollout_history = compute_rollout_uncertainty(
            excg_model,
            train_context,
            train_y,
            pool_context,
            num_pool,
            num_rollouts,
            rollout_length,
            device,
        )
        print("excg rollout uncertainty computed")

        print("Computing excg oneshot uncertainty")
        excg_oneshot_eu = compute_oneshot_uncertainty(
            excg_model, train_context, train_y, pool_context, num_pool
        )
        print("excg oneshot uncertainty computed")
    else:
        # Load eu from file
        print(f"Loading epistemic uncertainty from files in {args.save_folder}")
        autoreg_rollout_eu = torch.load(
            f"{args.save_folder}/autoreg_rollout_eu.pt", weights_only=True
        )
        autoreg_rollout_history = torch.load(
            f"{args.save_folder}/autoreg_rollout_history.pt", weights_only=True
        )
        autoreg_oneshot_eu = torch.load(
            f"{args.save_folder}/autoreg_oneshot_eu.pt", weights_only=True
        )
        excg_rollout_eu = torch.load(
            f"{args.save_folder}/excg_rollout_eu.pt", weights_only=True
        )
        excg_rollout_history = torch.load(
            f"{args.save_folder}/excg_rollout_history.pt", weights_only=True
        )
        excg_oneshot_eu = torch.load(
            f"{args.save_folder}/excg_oneshot_eu.pt", weights_only=True
        )

        # Update max choice for later experiments
        print(f"Resetting max choice to {args.new_max_choice}")
        args.max_choice = args.new_max_choice
        max_choice = args.new_max_choice

        new_folder = get_file_folder(args)
        print(f"Resetting file folder to {new_folder}")
        args.save_folder = new_folder

    # Ablate over number of pool choice
    autoreg_rollout_losses = torch.zeros(
        max_choice - 1, train_context.shape[0], num_test, device=device
    )
    autoreg_oneshot_losses = torch.zeros(
        max_choice - 1, train_context.shape[0], num_test, device=device
    )
    excg_rollout_losses = torch.zeros(
        max_choice - 1, train_context.shape[0], num_test, device=device
    )
    excg_oneshot_losses = torch.zeros(
        max_choice - 1, train_context.shape[0], num_test, device=device
    )

    for num_pool_choice in tqdm(
        range(1, max_choice), desc="Iterating over pool choice"
    ):
        # Pick highest uncertainty
        autoreg_rollout_chosen_idx = torch.argsort(
            autoreg_rollout_eu, dim=1, descending=True
        )[:, :num_pool_choice]
        autoreg_oneshot_chosen_idx = torch.argsort(
            autoreg_oneshot_eu, dim=1, descending=True
        )[:, :num_pool_choice]
        excg_rollout_chosen_idx = torch.argsort(
            excg_rollout_eu, dim=1, descending=True
        )[:, :num_pool_choice]
        excg_oneshot_chosen_idx = torch.argsort(
            excg_oneshot_eu, dim=1, descending=True
        )[:, :num_pool_choice]

        # Build context with chosen idx
        auto_rollout_context, auto_rollout_y = pick_context(
            pool_context, pool_y, autoreg_rollout_chosen_idx
        )
        auto_rollout_context = torch.cat([train_context, auto_rollout_context], dim=1)
        auto_rollout_y = torch.cat([train_y, auto_rollout_y], dim=1)

        auto_oneshot_context, auto_oneshot_y = pick_context(
            pool_context, pool_y, autoreg_oneshot_chosen_idx
        )
        auto_oneshot_context = torch.cat([train_context, auto_oneshot_context], dim=1)
        auto_oneshot_y = torch.cat([train_y, auto_oneshot_y], dim=1)

        excg_rollout_context, excg_rollout_y = pick_context(
            pool_context, pool_y, excg_rollout_chosen_idx
        )
        excg_rollout_context = torch.cat([train_context, excg_rollout_context], dim=1)
        excg_rollout_y = torch.cat([train_y, excg_rollout_y], dim=1)

        excg_oneshot_context, excg_oneshot_y = pick_context(
            pool_context, pool_y, excg_oneshot_chosen_idx
        )
        excg_oneshot_context = torch.cat([train_context, excg_oneshot_context], dim=1)
        excg_oneshot_y = torch.cat([train_y, excg_oneshot_y], dim=1)

        # Evaluate models with chosen idx
        test_context = test_context.to(device)
        test_y = test_y.to(device)
        for i in tqdm(
            range(num_test), desc="Iterating over test data", position=1, leave=False
        ):
            # Evaluate everything using autoreg model but different contexts
            autoreg_rollout_losses[num_pool_choice - 1, :, i] = eval_model(
                autoreg_model,
                auto_rollout_context,
                auto_rollout_y,
                test_context[:, i : i + 1, :],
                test_y[:, i : i + 1, :],
            ).flatten()

            autoreg_oneshot_losses[num_pool_choice - 1, :, i] = eval_model(
                autoreg_model,
                auto_oneshot_context,
                auto_oneshot_y,
                test_context[:, i : i + 1, :],
                test_y[:, i : i + 1, :],
            ).flatten()

            excg_rollout_losses[num_pool_choice - 1, :, i] = eval_model(
                autoreg_model,
                excg_rollout_context,
                excg_rollout_y,
                test_context[:, i : i + 1, :],
                test_y[:, i : i + 1, :],
            ).flatten()

            excg_oneshot_losses[num_pool_choice - 1, :, i] = eval_model(
                autoreg_model,
                excg_oneshot_context,
                excg_oneshot_y,
                test_context[:, i : i + 1, :],
                test_y[:, i : i + 1, :],
            ).flatten()
    return {
        "autoreg_rollout_losses": autoreg_rollout_losses,
        "autoreg_oneshot_losses": autoreg_oneshot_losses,
        "excg_rollout_losses": excg_rollout_losses,
        "excg_oneshot_losses": excg_oneshot_losses,
        "autoreg_oneshot_eu": autoreg_oneshot_eu,
        "autoreg_rollout_eu": autoreg_rollout_eu,
        "excg_oneshot_eu": excg_oneshot_eu,
        "excg_rollout_eu": excg_rollout_eu,
        "autoreg_rollout_history": autoreg_rollout_history,
        "excg_rollout_history": excg_rollout_history,
    }


if __name__ == "__main__":
    args = get_args()
    args.save_folder = get_file_folder(args)

    print(f"Evaluating with params:\n{json.dumps(vars(args), indent=2)}")

    results = run_exp(
        data_type=args.data_type,
        num_train=args.num_train,
        num_pool=args.num_pool,
        max_choice=args.max_choice,
        num_test=args.num_test,
        num_rollouts=args.num_rollouts,
        rollout_length=args.rollout_length,
        epoch=args.model_epoch,
        seed=args.seed,
        gpu_id=args.gpu_id,
        args=args,
    )

    os.makedirs(args.save_folder, exist_ok=True)

    print(f"Saving results to {args.save_folder}")
    for key in results:
        print(f"{key} shape: {results[key].shape}. Saving {key} to {key}.pt")
        torch.save(results[key].cpu(), os.path.join(args.save_folder, f"{key}.pt"))
