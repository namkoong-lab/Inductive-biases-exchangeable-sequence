import sys
sys.path.append('..')

from data.load_data import load_reward_data, load_toy_reward_data, load_contextual_bandit_data, load_classical_contextual_bandit_data, load_gpsampler_constant_data
from models.autoreg_model import Autoreg_Model
from models.excg_model import ExCg_Model
from utils.scheduler import CosineWarmupScheduler
import torch
from tqdm import tqdm
import wandb
import os

class Trainer:
    def __init__(self, accelerator, model_args, training_args, data_args, logging_args):
        self.accelerator = accelerator
        self.model_args = model_args
        self.training_args = training_args
        self.data_args = data_args
        self.logging_args = logging_args
        
        self.set_up(self.training_args)
        if self.accelerator.is_main_process:
            wandb.define_metric("avg_sample_loss", step_metric="custom_step")
            wandb.define_metric("avg_mean_loss", step_metric="custom_step")
            wandb.define_metric("avg_perplexity", step_metric="custom_step")
            wandb.define_metric("avg_uncertainty", step_metric="custom_step")


    def set_up(self, training_args):
        if self.model_args.model_type == "autoreg":
            self.model = Autoreg_Model(self.model_args, self.accelerator)
        elif self.model_args.model_type == "excg":
            self.model = ExCg_Model(self.model_args, self.accelerator)

        if training_args.load_from_checkpoint:
            self.model.load_state_dict(torch.load(training_args.checkpoint_path, weights_only=True))
        
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.training_args.lr, weight_decay=self.training_args.weight_decay)

        if self.data_args.dataset_name == "toy_data":
            self.training_dataloader, self.test_dataloader = load_toy_reward_data(self.training_args, self.data_args, self.model_args)
        elif self.data_args.dataset_name == "contextual_bandit":
            self.training_dataloader, self.test_dataloader = load_contextual_bandit_data(self.training_args, self.data_args, self.model_args)
        elif self.data_args.dataset_name == "classical_contextual_bandit":
            self.training_dataloader, self.test_dataloader = load_classical_contextual_bandit_data(self.training_args, self.data_args, self.model_args)
        elif self.data_args.dataset_name == "gp":
            self.training_dataloader, self.test_dataloader = load_gpsampler_constant_data(self.training_args, self.data_args, self.model_args)
        else:
            self.training_dataloader, self.test_dataloader = load_reward_data(self.training_args, self.data_args, self.model_args)

        self.scheduler = CosineWarmupScheduler(self.optimizer, self.training_args)

        self.model, self.optimizer, self.training_dataloader, self.test_dataloader, self.scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.training_dataloader, self.test_dataloader, self.scheduler
        )

        self.accelerator.print(f'[MODEL ARCHITECTURE] {self.model}')
        # calculate the total number of parameters of tnp
        total_params = sum(p.numel() for p in self.model.parameters())
        self.accelerator.print(f'[MODEL PARAMS] Total number of parameters: {total_params}')


    def train(self):
        self.evaluate(self.test_dataloader, initial=True, iter=0, prefix="test")

        for epoch in range(1, self.training_args.epochs + 1):
            self.accelerator.print(f'Epoch {epoch}')
            for batch in tqdm(self.training_dataloader, leave=False, desc=f"Training", disable=not self.accelerator.is_local_main_process):
                self.optimizer.zero_grad()
                loss = self.model(batch)
                self.accelerator.backward(loss)
                gathered_loss = self.accelerator.gather(loss).cpu().mean().item()
                self.optimizer.step()
                self.scheduler.step()

                self.accelerator.log({f"lr": self.optimizer.param_groups[0]['lr']})
                self.accelerator.log({f"total loss trajectory": gathered_loss})

            self.evaluate(self.test_dataloader, final=True, iter=epoch, prefix="test")
            # ## NOT SAVING MODEL FOR NOW TODO Later
            self.save_model(checkpoint_iter=epoch)

    
    def evaluate(self, data_loaders, initial=False, final=False, iter=None, prefix=None):
        if initial:
            self.accelerator.print(f'[INITIAL EVALUATION] {prefix}')
        elif final:
            self.accelerator.print(f'[FINAL EVALUATION] {prefix}')

        dataloader = data_loaders
        eval_metric_list, eval_metric_mean = self.eval_metric_func(dataloader, self.accelerator, self.model, self.training_args.test_horizon)
        if self.accelerator.is_main_process:
            for metric, values in eval_metric_list.items():
                data = [[x, y.item()] for x, y in enumerate(values)]
                table = wandb.Table(data=data, columns=["time_step", "value"])
                line_series = wandb.plot.line(table, "time_step", "value", title=f"{prefix} {metric} T{iter}")
                wandb.log({f"{prefix}_{metric}_series_{iter}": line_series})

            for metric, values in eval_metric_mean.items():
                wandb.log({f"{metric}": values.item(), "custom_step": iter})

    def eval_metric_func(self,eval_dataloader, accelerator, model, T):
        eval_metric_list = {
            'sample_loss': torch.zeros(T, device=accelerator.device),
            'mean_loss': torch.zeros(T, device=accelerator.device),
            'perplexity': torch.zeros(T, device=accelerator.device),
            'uncertainty': torch.zeros(T, device=accelerator.device)
        }

        eval_metric_mean = {
            'avg_sample_loss': torch.zeros(1, device=accelerator.device),
            'avg_mean_loss': torch.zeros(1, device=accelerator.device),
            'avg_perplexity': torch.zeros(1, device=accelerator.device),
            'avg_uncertainty': torch.zeros(1, device=accelerator.device),
        }
        
        n = torch.tensor(0, device=accelerator.device)
        model.eval()
        
        with torch.no_grad():
            for batch in tqdm(eval_dataloader, leave=False, desc="Evaluation", disable=not accelerator.is_local_main_process):
                n += 1
                std_list, log_prob_list, sample_loss_list, loss_list, std_mean, log_prob_mean, sample_loss_mean, mean_loss_mean = model.evaluate(batch)
                # print(f"std_list: {std_list}, shape: {std_list.shape}, device: {std_list.device}")
                # print(f"log_prob_list: {log_prob_list}, shape: {log_prob_list.shape}, device: {log_prob_list.device}")
                # print(f"sample_loss_list: {sample_loss_list}, shape: {sample_loss_list.shape}, device: {sample_loss_list.device}")
                # print(f"loss_list: {loss_list}, shape: {loss_list.shape}, device: {loss_list.device}")

                
                eval_metric_list['uncertainty'] += std_list
                eval_metric_list['perplexity'] += log_prob_list
                eval_metric_list['sample_loss'] += sample_loss_list
                eval_metric_list['mean_loss'] += loss_list

                eval_metric_mean['avg_mean_loss'] += mean_loss_mean
                eval_metric_mean['avg_perplexity'] += log_prob_mean
                eval_metric_mean['avg_uncertainty'] += std_mean
                eval_metric_mean['avg_sample_loss'] += sample_loss_mean


        for metric in eval_metric_list:
            eval_metric_list[metric] = accelerator.gather(eval_metric_list[metric])

        for metric in eval_metric_mean:
            eval_metric_mean[metric] = accelerator.gather(eval_metric_mean[metric])

        ## reshape to keep track of the metrics over time
        if accelerator.is_main_process:
            for metric in eval_metric_list:
                eval_metric_list[metric] = eval_metric_list[metric].view(-1, T)

            for metric in eval_metric_mean:
                eval_metric_mean[metric] = eval_metric_mean[metric].view(-1, 1)
            
        n = accelerator.gather(n)
        # accelerator.print(f"n: {n}, shape: {n.shape}")
        
        if accelerator.is_main_process:
            for metric in eval_metric_list:
                eval_metric_list[metric] = (eval_metric_list[metric].sum(0) / n.sum()).detach().cpu().numpy()
            for metric in eval_metric_mean:
                eval_metric_mean[metric] = (eval_metric_mean[metric].sum(0) / n.sum()).detach().cpu().numpy()
                
        model.train()
        
        return eval_metric_list, eval_metric_mean

    def save_model(self, checkpoint_iter = None):
        self.accelerator.wait_for_everyone()
        unwrapped_model = self.accelerator.unwrap_model(self.model)

        if self.accelerator.is_local_main_process:
            if self.data_args.dataset_name == "contextual_bandit":
                saving_dir = f"/shared/share_mala/Leon/CB_{self.model_args.dim_llm_embedding}/UQ-{self.model_args.uncertainty}-Loss-{self.model_args.loss_type}-Horizon-{self.training_args.train_horizon}-grad-{self.model_args.gradient_type}_Noise_{self.data_args.noise_scale}"
                if not os.path.exists(saving_dir):
                    os.makedirs(saving_dir)
                if checkpoint_iter == None:
                    torch.save(unwrapped_model.state_dict(), f"{saving_dir}/model.pt")

                else: 
                    torch.save(unwrapped_model.state_dict(), f"{saving_dir}/model_checkpoint_{checkpoint_iter}.pt")
            
            elif self.data_args.dataset_name == "classical_contextual_bandit":
                saving_dir = f"/shared/share_mala/Leon/CCB_{self.model_args.dim_llm_embedding}/UQ-{self.model_args.uncertainty}-Gradient-{self.model_args.gradient_type}-Loss-{self.model_args.loss_type}-Horizon-{self.training_args.train_horizon}_Noise_{self.data_args.noise_scale}"
                if not os.path.exists(saving_dir):
                    os.makedirs(saving_dir)
                if checkpoint_iter == None:
                    torch.save(unwrapped_model.state_dict(), f"{saving_dir}/model.pt")

                else: 
                    torch.save(unwrapped_model.state_dict(), f"{saving_dir}/model_checkpoint_{checkpoint_iter}.pt")
            
            elif self.data_args.dataset_name == "gp":
                saving_dir = f"/shared/share_mala/Leon/GP_{self.model_args.dim_llm_embedding}/{self.model_args.model_type}-UQ-{self.model_args.uncertainty}-Gradient-{self.model_args.gradient_type}-Loss-{self.model_args.loss_type}-Horizon-{self.training_args.train_horizon}_Noise_{self.data_args.noise_scale}"
                if not os.path.exists(saving_dir):
                    os.makedirs(saving_dir)
                if checkpoint_iter == None:
                    torch.save(unwrapped_model.state_dict(), f"{saving_dir}/model.pt")
                else: 
                    torch.save(unwrapped_model.state_dict(), f"{saving_dir}/model_checkpoint_{checkpoint_iter}.pt")
