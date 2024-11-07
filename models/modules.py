import torch
import torch.nn as nn
import torch.nn.functional as F

def build_mlp(dim_in, dim_hid, dim_out, depth):
    modules = [nn.Linear(dim_in, dim_hid), nn.SiLU(True)]
    for _ in range(depth-2):
        modules.append(nn.Linear(dim_hid, dim_hid))
        modules.append(nn.SiLU(True))
    modules.append(nn.Linear(dim_hid, dim_out))
    return nn.Sequential(*modules)

class RiemannDistribution(nn.Module):
    def __init__(self, borders, accelerator = None):
        super().__init__()
        assert len(borders.shape) == 1, "Borders should be a 1D tensor."
        self.accelerator = accelerator
        self.register_buffer('borders', borders)
        self.register_buffer('bucket_widths', borders[1:] - borders[:-1])
        self.num_buckets = len(borders) - 1

    def map_to_bucket_idx(self, y):
        # Map each target value y to its corresponding bucket index
        target_sample = torch.searchsorted(self.borders, y, right=False) - 1
        target_sample = target_sample.clamp(0, self.num_buckets - 1)
        return target_sample

    def forward(self, logits, y):
        # logits: (batch_size, seq_len, num_buckets)
        # y: (batch_size, seq_len)
        target_sample = self.map_to_bucket_idx(y)  # (batch_size, seq_len)
        bucket_log_probs = F.log_softmax(logits, dim=-1)  # (batch_size, seq_len, num_buckets)
        log_bucket_widths = torch.log(self.bucket_widths)  # (num_buckets,)
        scaled_log_probs = bucket_log_probs - log_bucket_widths  # Broadcasting over num_buckets
        # Gather the log probabilities for the target buckets
        log_probs = scaled_log_probs.gather(-1, target_sample.unsqueeze(-1)).squeeze(-1)  # (batch_size, seq_len)
        # Compute the negative log-likelihood loss
        loss = -log_probs.mean()
        return log_probs, loss

    def sample(self, logits):

        bucket_indices = torch.distributions.Categorical(logits=logits).sample()  # (batch_size, seq_len)
        # Sample uniformly within the bucket
        bucket_lefts = self.borders[:-1][bucket_indices]  # (batch_size, seq_len)
        bucket_rights = self.borders[1:][bucket_indices]  # (batch_size, seq_len)
        samples = bucket_lefts + (bucket_rights - bucket_lefts) * torch.rand_like(bucket_lefts)
        return samples