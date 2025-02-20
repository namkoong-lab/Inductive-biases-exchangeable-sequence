from dataclasses import dataclass, field
from models.excg_model import ExCg_Model
from models.autoreg_model import Autoreg_Model
import yaml
import torch

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
    model_type: str = field(default="autoreg")



def load_model(config_path: str, checkpoint_dir: str):

    uq_config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)
    uq_model_args = ModelArguments(**uq_config['model_args'])

    if uq_model_args.model_type == "autoreg":
        model = Autoreg_Model(uq_model_args)
    elif uq_model_args.model_type == "excg":
        model = ExCg_Model(uq_model_args)
    model.load_state_dict(torch.load(checkpoint_dir, weights_only=True))
    model.eval()
    
    return model