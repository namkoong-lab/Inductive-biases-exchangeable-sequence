import sys
sys.path.append('..')
import torch
from accelerate import Accelerator
import os
from trainers.trainer import Trainer
from dataclasses import dataclass, field
from transformers import HfArgumentParser

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
    gradient_type: str = field(default="full")
    borders_data_dir: str = field(default=None)
    model_type: str = field(default="autoreg")

@dataclass
class TrainingArguments:
    lr: float = field(default=0.0009)
    seed: int = field(default=3004)
    weight_decay: float = field(default=1e-4)
    epochs: int = field(default=100)
    warmup_ratio: float = field(default=0.03)
    min_lr: float = field(default=1e-6)
    total_train_batch_size: int = field(default=64)
    total_test_batch_size: int = field(default=64)
    num_process: int = field(default=8)
    eval_steps: int = field(default=100)
    train_horizon: int = field(default=64)
    test_horizon: int = field(default=64)
    eval_func: str = field(default="eval_func")
    context_size: int = field(default=100)
    target_size: int = field(default=100)
    num_train_samples: int = field(default=1000)
    num_test_samples: int = field(default=1000)
    load_from_checkpoint: bool = field(default=False)
    checkpoint_path: str = field(default=None)

@dataclass
class DataArguments:
    dataset_name: str = field(default="Pretrain_Dataset")
    num_train_workers: int = field(default=32)
    num_test_workers: int = field(default=8)
    noise_scale: float = field(default=0.1)
    alpha: float = field(default=1.0)

@dataclass
class LoggingArguments:
    task: str = field(default="offline")
    wandb_project: str = field(default="Training_Logs")
    eval_log_step: int = field(default=100)

def main():
    parser = HfArgumentParser((ModelArguments, TrainingArguments, DataArguments, LoggingArguments))
    model_args, training_args, data_args, logging_args = parser.parse_args_into_dataclasses()


    accelerator = Accelerator(log_with="wandb")


    # set the seed for each process
    seed = training_args.seed
    accelerator.wait_for_everyone()
    process_seed = seed + accelerator.process_index
    torch.manual_seed(process_seed)
    torch.cuda.manual_seed(process_seed)
    torch.cuda.manual_seed_all(process_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Print verification for each process
    print(f"Process {accelerator.process_index}: Seed set to {process_seed}")
    accelerator.wait_for_everyone()

    wandb_run_name = f"{model_args.model_type}_X_{model_args.dim_llm_embedding}_Y_{model_args.dim_y}_Horizon_{training_args.train_horizon}_Noise_{data_args.noise_scale}_seed_{training_args.seed}"
    # Initialize wandb
    accelerator.init_trackers(
        project_name=logging_args.wandb_project,
        init_kwargs={"wandb": {
            "name": wandb_run_name,
        }}
    )

    trainer = Trainer(
        accelerator = accelerator,
        model_args = model_args,
        training_args = training_args,
        data_args = data_args,
        logging_args = logging_args
    )

    trainer.train()

    accelerator.end_training()

if __name__ == '__main__':
    main() 