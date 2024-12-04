import torch
import torch.distributions as dist
from types import SimpleNamespace
import torch.nn as nn
import torch.optim as optim


class TS_machine:
    def __init__(self, uq_model, device, dim_llm_embedding, imagination_horizon, nl):
        self.uq_model = uq_model
        self.device = device
        self.dim_llm_embedding = dim_llm_embedding
        self.imagination_horizon = imagination_horizon
        self.nl = nl
        self.uq_joint_context_x = []
        self.uq_joint_context_y = []
        self.uq_joint_pred_loss = []
        self.uq_joint_actual_rewards = []

    def predict(self, x_t, arm):
        if self.nl:
            arm.to(self.device)
            x_t = x_t.to(self.device)
            uq_joint_predicted_rewards = arm(x_t)
            return uq_joint_predicted_rewards
        else:
            arm = arm.to(self.device)
            x_t = x_t.to(self.device)
            uq_joint_predicted_rewards = torch.matmul(x_t, arm.T)
            return uq_joint_predicted_rewards

    def update_batch(self, x_t, reward_t):
        self.uq_joint_context_x.append(x_t.unsqueeze(0))
        self.uq_joint_context_y.append(torch.tensor(reward_t, device=self.device).unsqueeze(0).unsqueeze(0))
    
    def set_batch(self):
        batch = SimpleNamespace(
        xc=torch.stack(self.uq_joint_context_x, dim=1).repeat(1, 1, 1).to(self.device) if self.uq_joint_context_x else torch.empty(1, 0, self.dim_llm_embedding, device=self.device),
        yc=torch.stack(self.uq_joint_context_y, dim=1).repeat(1, 1, 1).to(self.device) if self.uq_joint_context_y else torch.empty(1, 0, 1, device=self.device))

        return batch
    
    def ts_imagination_joint(self):
        inner_context_x = []
        inner_context_y = []
        uq_joint_batch = self.set_batch()
        dim_llm_embedding = self.dim_llm_embedding
        device = self.device
        uq_model = self.uq_model
        uq_context_x = uq_joint_batch.xc
        uq_context_y = uq_joint_batch.yc
        

        for i in range(self.imagination_horizon):

            # Generate random contexts
            context = torch.randn(1, 1, dim_llm_embedding, device=device)

            inner_xc = torch.cat(inner_context_x, dim=1) if inner_context_x else torch.empty(1, 0, dim_llm_embedding, device=device)
            full_xc = torch.cat([uq_context_x, inner_xc], dim=1) 
            inner_yc = torch.cat(inner_context_y, dim=1) if inner_context_y else torch.empty(1, 0, 1, device=device)
            full_yc = torch.cat([uq_context_y, inner_yc], dim=1) 
            full_xc = full_xc.to(device)
            full_yc = full_yc.to(device)
            batch = SimpleNamespace(
                xc=full_xc,
                yc=full_yc,
                xt=context,  # Shape: num_arms,1, dim_llm_embedding
                yt=torch.zeros(1, 1, 1, device=device)  # Shape: num_arms,1,1
            )

            predicted_rewards = uq_model.predict(batch)
            inner_context_x.append(context)
            inner_context_y.append(predicted_rewards)

        # Concatenate all inner context data
        X = torch.cat(inner_context_x, dim=1)  
        y = torch.cat(inner_context_y, dim=1)  
        X = torch.cat([uq_context_x, X], dim=1)
        y = torch.cat([uq_context_y, y], dim=1)

        if self.nl:
            return train_mlp(X, y, epochs=1000, lr=1e-2)
        else:
            return torch.linalg.lstsq(X.squeeze(0), y.squeeze(0)).solution

    def ts_imagination_marginal(self):
        inner_context_x = []
        inner_context_y = []
        uq_joint_batch = self.set_batch()
        dim_llm_embedding = self.dim_llm_embedding
        device = self.device
        uq_model = self.uq_model
        uq_context_x = uq_joint_batch.xc
        uq_context_y = uq_joint_batch.yc
        uq_context_x = uq_context_x.to(device)
        uq_context_y = uq_context_y.to(device)
        

        for i in range(self.imagination_horizon):

            # Generate random contexts
            context = torch.randn(1, 1, dim_llm_embedding, device=device)

            batch = SimpleNamespace(
                xc=uq_context_x,
                yc=uq_context_y,
                xt=context,  # Shape: num_arms,1, dim_llm_embedding
                yt=torch.zeros(1, 1, 1, device=device)  # Shape: num_arms,1,1
            )

            predicted_rewards = uq_model.predict(batch)
            inner_context_x.append(context)
            inner_context_y.append(predicted_rewards)

        # Concatenate all inner context data
        X = torch.cat(inner_context_x, dim=1)  
        y = torch.cat(inner_context_y, dim=1)  
        X = torch.cat([uq_context_x, X], dim=1)
        y = torch.cat([uq_context_y, y], dim=1)

        if self.nl:
            return train_mlp(X, y, epochs=1000, lr=1e-2)
        else:
            return torch.linalg.lstsq(X.squeeze(0), y.squeeze(0)).solution
    

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=1):
        super(MLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.network(x)

def train_mlp(X, y, epochs=1000, lr=1e-2):
    model = MLP(input_dim=X.shape[-1]).to(X.device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    for _ in range(epochs):
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

    return model.eval()