import argparse
import json
import os

import torch
from load_data import (
    generate_al_additive_data,
    generate_al_hregions_data,
    generate_al_matmul_data,
    generate_al_regions_data,
    generate_al_regions_v2_data,
)
from tqdm import tqdm

DATA_FUNC = {
    "al_matmul": generate_al_matmul_data,
    "al_add": generate_al_additive_data,
    "al_regions": lambda **kwargs: generate_al_regions_data(
        with_high_aleatoric=False, **kwargs
    ),
    "al_regions_ha": lambda **kwargs: generate_al_regions_data(
        with_high_aleatoric=True, **kwargs
    ),
    "al_hregions": generate_al_hregions_data,
    "al_regions_v2": generate_al_regions_v2_data,
}


def make_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


def get_data_function(data_type, args):
    if data_type == "al_hregions":
        data_folder = f"{args.data_dir}/{data_type}_{args.al_hregions_num}/context_dim={args.context_dim}_horizon={args.horizon}_noise={args.noise_std}"
        f = lambda **kwargs: generate_al_hregions_data(
            num_regions=args.al_hregions_num, **kwargs
        )
    elif data_type.startswith("al_regions_v2"):
        expected_underscores = 3
        if data_type.count("_") != expected_underscores:
            raise ValueError(
                f"Expected {expected_underscores} underscores in {data_type}"
            )
        num_regions = int(data_type.split("_")[-1])
        data_folder = f"{args.data_dir}/{data_type}/context_dim={args.context_dim}_horizon={args.horizon}_noise={args.noise_std}"
        f = lambda **kwargs: generate_al_regions_v2_data(
            num_regions=num_regions, **kwargs
        )
        print(f"Generating region-based data v2 with {num_regions} regions")
    else:
        data_folder = f"{args.data_dir}/{args.data_type}/context_dim={args.context_dim}_horizon={args.horizon}_noise={args.noise_std}"
        f = DATA_FUNC[data_type]
    return f, data_folder


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", type=str, default="/shared/share_mala/tyen/seqb/data"
    )
    parser.add_argument(
        "--data_type",
        type=str,
        default="al_hregions",
        help=f"Type of data to generate. Options: {list(DATA_FUNC.keys())}",
    )
    parser.add_argument("--al_hregions_num", type=int, default=50)
    parser.add_argument("--num_train_data", type=int, default=50000)
    parser.add_argument("--num_test_data", type=int, default=50000)

    parser.add_argument("--context_dim", type=int, default=1)
    parser.add_argument("--horizon", type=int, default=200)
    parser.add_argument("--noise_std", type=float, default=0.1)
    return parser.parse_args()


if __name__ == "__main__":
    # Generate the data
    args = get_args()

    data_func, data_folder = get_data_function(args.data_type, args)
    make_folder(data_folder)

    all_contexts = []
    all_y = []
    for _ in tqdm(range(args.num_train_data), desc="Generating train data"):
        context, y = data_func(
            context_dim=args.context_dim, horizon=args.horizon, noise_std=args.noise_std
        )
        all_contexts.append(context)
        all_y.append(y)
    all_contexts = torch.stack(all_contexts, dim=0)
    all_y = torch.stack(all_y, dim=0)

    torch.save(all_contexts, f"{data_folder}/train_contexts.pt")
    torch.save(all_y, f"{data_folder}/train_y.pt")

    all_contexts = []
    all_y = []
    for _ in tqdm(range(args.num_test_data), desc="Generating test data"):
        context, y = data_func(
            context_dim=args.context_dim, horizon=args.horizon, noise_std=args.noise_std
        )
        all_contexts.append(context)
        all_y.append(y)
    all_contexts = torch.stack(all_contexts, dim=0)
    all_y = torch.stack(all_y, dim=0)

    torch.save(all_contexts, f"{data_folder}/test_contexts.pt")
    torch.save(all_y, f"{data_folder}/test_y.pt")

    print(f"Data saved to {data_folder}")

    # Record number of data points and config to a file
    with open(f"{data_folder}/config.json", "w") as f:
        json.dump(vars(args), f, indent=2)
        print(f"Config saved to {data_folder}/config.json")
