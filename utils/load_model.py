
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
from models.autoreg_model import Autoreg_Model
from models.excg_model import ExCg_Model
import torch
import numpy as np

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


def load_model(model_config_dir, model_checkpoint_dir, model_type, device):
    config = yaml.load(open(model_config_dir, 'r'), Loader=yaml.FullLoader)
    model_args = ModelArguments(**config['model_args'])

    if model_type == "autoreg":
        model = Autoreg_Model(model_args)
    elif model_type == "excg":
        model = ExCg_Model(model_args)

    model.load_state_dict(torch.load(model_checkpoint_dir, weights_only=True))
    model.eval()
    model.to(device)
    return model