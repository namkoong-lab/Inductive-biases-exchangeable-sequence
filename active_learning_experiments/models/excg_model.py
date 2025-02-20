import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

from models.modules import build_mlp
from collections import namedtuple

BatchData = namedtuple('BatchData', ['xc', 'yc', 'xt', 'yt'])


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

class ExCg_Model(nn.Module):
    def __init__(
        self,
        model_args,
        accelerator = None,
):
        super(ExCg_Model, self).__init__()
        self.accelerator = accelerator
        self.setup_configuration(model_args)
        self.setup_embedder()
        self.setup_transformer()
        self.setup_predictor()
        self.setup_gradient()
        self.print_gradient_status()

    def setup_configuration(self, model_args):
        self.dim_llm_embedding = model_args.dim_llm_embedding
        self.dim_y = model_args.dim_y
        self.d_model = model_args.d_model
        self.dim_feedforward = model_args.dim_feedforward
        self.nhead = model_args.nhead
        self.emb_depth = model_args.emb_depth
        self.dropout = model_args.dropout
        self.activation = model_args.activation
        self.num_layers = model_args.num_layers
        self.embed_type = model_args.embed_type
        self.bound_std = model_args.bound_std
        self.pad_value = model_args.pad_value
        self.loss_type = model_args.loss_type
        self.uncertainty = model_args.uncertainty
        self.gradient_type = model_args.gradient_type
        self.borders_data = torch.load(model_args.borders_data_dir, weights_only=True) if model_args.borders_data_dir is not None else None
        self.num_buckets = len(self.borders_data) - 1 if self.borders_data is not None else None

    def setup_embedder(self):
        if self.embed_type == 'embed_llm_embedding':
            if self.emb_depth == 0:
                self.embedder = nn.Linear(self.dim_llm_embedding, self.d_model-self.dim_y)
            else:
                self.embedder = build_mlp(self.dim_llm_embedding, self.dim_feedforward, self.d_model - self.dim_y, self.emb_depth)

        elif self.embed_type == 'embed_concat':
            if self.emb_depth == 0:
                self.embedder = nn.Linear(self.dim_llm_embedding + self.dim_y, self.d_model)
            else:
                self.embedder = build_mlp(self.dim_llm_embedding + self.dim_y, self.dim_feedforward, self.d_model, self.emb_depth)

    def setup_predictor(self):
        if self.uncertainty == 'normal':
            if self.emb_depth == 0:
                self.mean_predictor = nn.Linear(self.d_model, self.dim_y)
                self.std_predictor = nn.Linear(self.d_model, self.dim_y)
            else:
                self.mean_predictor = build_mlp(self.d_model, self.dim_feedforward, self.dim_y, self.emb_depth)
                self.std_predictor = build_mlp(self.d_model, self.dim_feedforward, self.dim_y, self.emb_depth)
            
        elif self.uncertainty == 'riemann':
            if self.emb_depth == 0:
                self.bucket_predictor = nn.Linear(self.d_model, self.num_buckets)
            else:
                self.bucket_predictor = build_mlp(
                    self.d_model, self.dim_feedforward, self.num_buckets, self.emb_depth)
            
            self.setup_distribution()
        else:
            if self.emb_depth == 0:
                self.mean_predictor = nn.Linear(self.d_model, self.dim_y)
            else:
                self.mean_predictor = build_mlp(self.d_model, self.dim_feedforward, self.dim_y, self.emb_depth)

    def setup_distribution(self):
        # Define bucket borders and initialize RiemannDistribution
        self.register_buffer('borders', self.borders_data)
        self.register_buffer('bucket_widths', self.borders_data[1:] - self.borders_data[:-1])
        self.riemann_distribution = RiemannDistribution(self.borders_data, self.accelerator)

    def setup_transformer(self):
        encoder_layer = nn.TransformerEncoderLayer(d_model = self.d_model, 
                                                   nhead = self.nhead,
                                                   dim_feedforward = self.dim_feedforward,
                                                   dropout = self.dropout,
                                                   activation = self.activation,
                                                   batch_first=True)
        
        self.encoder = nn.TransformerEncoder(encoder_layer, self.num_layers)

    def set_requires_grad(self,module, requires_grad):
        if isinstance(module, nn.Linear):
            module.weight.requires_grad = requires_grad
            module.bias.requires_grad = requires_grad
        elif isinstance(module, nn.Sequential):
            for layer in module:
                if isinstance(layer, nn.Linear):
                    layer.weight.requires_grad = requires_grad
                    layer.bias.requires_grad = requires_grad

    def setup_gradient(self):

        if self.gradient_type == 'mean':
            self.set_requires_grad(self.mean_predictor, True)
            self.set_requires_grad(self.std_predictor, False)
        elif self.gradient_type == 'std':
            for param in self.parameters():
                        param.requires_grad = False
                    # Only set std_predictor to True
            self.set_requires_grad(self.std_predictor, True)

    def print_gradient_status(self):
        if self.accelerator is not None:
            self.accelerator.print("Gradient status for each parameter:")
            for name, param in self.named_parameters():
                self.accelerator.print(f"{name}: requires_grad = {param.requires_grad}")
    
    def construct_excg_input(self, batch):
        model_dtype = next(self.encoder.parameters()).dtype

        xc = batch.xc
        yc = batch.yc
        xt = batch.xt
        yt = batch.yt

        if model_dtype != xc.dtype:
            xc = xc.to(model_dtype)
            yc = yc.to(model_dtype)
            xt = xt.to(model_dtype)
            yt = yt.to(model_dtype)

        x_y_ctx = torch.cat((xc, yc), dim=-1) 
        x_0_tar = torch.cat((xt, torch.zeros_like(yt)), dim=-1)

        raw_inp = torch.cat((x_y_ctx, x_0_tar), dim=1)
        embeddings = self.embedder(raw_inp)

        return embeddings 

    def create_excg_mask(self, num_ctx, num_tar):

        num_all = num_ctx + num_tar
        mask = torch.zeros(num_all, num_all).fill_(float('-inf'))
        mask[:, :num_ctx] = 0.0
        mask.diagonal().fill_(0.0)
        return mask

    def forward(self, batch):
        x = batch.x
        y = batch.y

        n = x.shape[1]
        num_ctx = n - torch.randint(low=1, high=n-1, size=(1,)).item()
        xc, yc = x[:, :num_ctx], y[:, :num_ctx]
        xt, yt = x[:, num_ctx:], y[:, num_ctx:]
        batch_data = BatchData(xc, yc, xt, yt)

        embeddings = self.construct_excg_input(batch_data)
        mask = self.create_excg_mask(xc.shape[1], xt.shape[1])
        mask = mask.to(dtype=embeddings.dtype, device=embeddings.device)

        encoding = self.encoder(embeddings, mask=mask)
        prediction = encoding[:, -xt.shape[1]:, :]
        
        if self.uncertainty == 'normal':

            mean = self.mean_predictor(prediction)
            std = self.std_predictor(prediction)
            if mean.dtype != yt.dtype:
                    yt = yt.to(mean.dtype)

            if self.bound_std:
                std = 0.05 + 0.95 * F.softplus(std)
            else:
                std = std = torch.exp(std)

            pred_dist = Normal(mean, std)

            if self.loss_type == 'logprob':
                loss = - pred_dist.log_prob(yt).sum(-1).mean()
            
            elif self.loss_type == 'mse':
                sample = pred_dist.rsample()
                loss = F.mse_loss(sample, yt)

        elif self.uncertainty == 'riemann':
            logits = self.bucket_predictor(prediction)  # (batch_size, seq_len, num_buckets)

            y = yt.squeeze()

            _, loss = self.riemann_distribution(logits, y)

            return loss

        else:
            out = self.mean_predictor(prediction)
            if out.dtype != yt.dtype:
                yt = yt.to(out.dtype)
            loss = F.mse_loss(out, yt)

        return loss
    
    def evaluate(self, batch):
        '''
        For evaluation, this is basically the same as forward, but besides loss, we still want to calculate the mean loss, and sample a point out, and the loss with the true target, and record the mean of std per step.
        So lets say we have x1:t and y1:t, we want to see the plot the std over all data and batch over time t. So when we use wandb log, we will create a diagram for every evaluation step, and that diagram will show the std over sequence t.
        '''

        x = batch.x
        y = batch.y


        device = next(self.encoder.parameters()).device
        std_list = []
        log_prob_list = []
        sample_loss_list = []
        mean_list = []
        for i in range(x.shape[1]):
            num_ctx = i
            xc = x[:, :num_ctx]
            yc = y[:, :num_ctx]
            xt = x[:, num_ctx:num_ctx+1]
            yt = y[:, num_ctx:num_ctx+1]
            batch_data = BatchData(xc, yc, xt, yt)
            embeddings = self.construct_excg_input(batch_data)
            mask = self.create_excg_mask(xc.shape[1], xt.shape[1])
            mask = mask.to(dtype=embeddings.dtype, device=embeddings.device)

            encoding = self.encoder(embeddings, mask=mask)
            prediction = encoding[:, -xt.shape[1]:, :]

            if self.uncertainty == 'normal':

                mean = self.mean_predictor(prediction)
                std = self.std_predictor(prediction)
                if mean.dtype != yt.dtype:
                        yt = yt.to(mean.dtype)

                if self.bound_std:
                    std = 0.05 + 0.95 * F.softplus(std)
                else:
                    std = std = torch.exp(std)

                pred_dist = Normal(mean, std)
                sample = pred_dist.sample()
                sample_loss = F.mse_loss(sample, yt, reduction='none')
                mean_loss = F.mse_loss(mean, yt, reduction='none')

                std_list.append(std.mean(dim=-1).mean(dim = 0))
                log_prob_list.append(pred_dist.log_prob(yt).mean(dim=-1).mean(dim = 0))
                sample_loss_list.append(sample_loss.mean(dim=-1).mean(dim = 0))
                mean_list.append(mean_loss.mean(dim=-1).mean(dim = 0))
            
            
            elif self.uncertainty == 'riemann':
                logits = self.bucket_predictor(prediction)
                yt = yt.squeeze()

                # Compute loss
                log_probs, loss = self.riemann_distribution(logits, yt)
                samples = self.riemann_distribution.sample(logits)

                # Compute mean predictions for MSE
                bucket_probs = F.softmax(logits, dim=-1)  # (batch_size, seq_len, num_buckets)
                bucket_lefts = self.borders[:-1]  # (num_buckets,)
                bucket_rights = self.borders[1:]  # (num_buckets,)
                bucket_centers = (bucket_lefts + bucket_rights) / 2  # (num_buckets,)
                mean_predictions = torch.sum(bucket_probs * bucket_centers, dim=-1)  # (batch_size, seq_len)

                mean_loss = F.mse_loss(mean_predictions, yt.view(mean_predictions.shape), reduction='none')
                mean_list = mean_loss.mean(dim = 0)
                sample_loss = F.mse_loss(samples, yt.view(samples.shape), reduction='none')
                sample_loss_list = sample_loss.mean(dim = 0)
                std_list = torch.zeros(yt.shape[1], device=device)
                log_prob_list = log_probs.mean(dim = 0)

                std_mean = std_list.mean()
                log_prob_mean = log_prob_list.mean()
                sample_loss_mean = sample_loss_list.mean()
                mean_loss_mean = mean_list.mean()


        std_list = torch.tensor(std_list, device=device)
        log_prob_list = torch.tensor(log_prob_list, device=device)
        sample_loss_list = torch.tensor(sample_loss_list, device=device)
        mean_list = torch.tensor(mean_list, device=device)
        std_mean = std_list.mean()
        log_prob_mean = log_prob_list.mean()
        sample_loss_mean = sample_loss_list.mean()
        mean_loss_mean = mean_list.mean()

        return std_list, log_prob_list, sample_loss_list, mean_list, std_mean, log_prob_mean, sample_loss_mean, mean_loss_mean
        
    
    def construct_prediction_input(self, batch):
        xc = batch.xc
        yc = batch.yc
        xt = batch.xt
        yt = batch.yt

        model_dtype = next(self.encoder.parameters()).dtype
        if model_dtype != xc.dtype:
            xc = xc.to(model_dtype)
            yc = yc.to(model_dtype)
            xt = xt.to(model_dtype)
            yt = yt.to(model_dtype)

        x_y_ctx = torch.cat((xc, yc), dim=-1) 
        x_0_tar = torch.cat((xt, torch.zeros_like(yt)), dim=-1)

        raw_inp = torch.cat((x_y_ctx, x_0_tar), dim=1)
        embeddings = self.embedder(raw_inp)

        return embeddings
    
    def create_prediction_mask(self, batch):
        num_ctx = batch.xc.shape[1]
        num_tar = batch.xt.shape[1]
        num_all = num_ctx + num_tar
        mask = torch.zeros(num_all, num_all).fill_(float('-inf'))
        mask[:, :num_ctx] = 0.0
        mask.diagonal().fill_(0.0)
        return mask

    def predict(self, batch):
        xt = batch.xt
        yt = batch.yt
        device = next(self.encoder.parameters()).device

        embeddings = self.construct_prediction_input(batch)
        mask = self.create_prediction_mask(batch)
        mask = mask.to(embeddings.dtype)
        mask = mask.to(device)

        encoding = self.encoder(embeddings, mask=mask)
        prediction = encoding[:, -xt.shape[1]:, :]

        if self.uncertainty == 'normal':

            mean = self.mean_predictor(prediction)
            std = self.std_predictor(prediction)

            if self.bound_std:
                std = 0.05 + 0.95 * F.softplus(std)
            else:
                std = std = torch.exp(std)

            pred_dist = Normal(mean, std)
            sample = pred_dist.sample()

        elif self.uncertainty == 'riemann':
            logits = self.bucket_predictor(prediction)  # (batch_size, num_tar, num_buckets)
            sample = self.riemann_distribution.sample(logits).unsqueeze(-1)  # 
        else: 
            sample = self.mean_predictor(prediction)
        
        return sample
    