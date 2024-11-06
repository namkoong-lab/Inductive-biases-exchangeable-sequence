import torch
from types import SimpleNamespace



class joint_prediction_machine:
    def __init__(self, uq_model, device, B, D, T, batch_data):
        self.uq_model = uq_model
        self.device = device
        self.B = B
        self.D = D
        self.T = T
        self.uq_joint_batch = batch_data
        self.uq_joint_context_x = []
        self.uq_joint_context_y = []

    def ts_imagination(self):
        inner_context_x = []
        inner_context_y = []
        uq_joint_batch = self.uq_joint_batch
        device = self.device
        uq_model = self.uq_model
        batch_size = uq_joint_batch.xt.shape[0]
        uq_context_x = uq_joint_batch.xc
        uq_context_y = uq_joint_batch.yc

        with torch.no_grad():
            for i in range(self.T):

                # Generate random contexts
                context_x = uq_joint_batch.xt[:, i, :]

                inner_xc = torch.cat(inner_context_x, dim=1) if inner_context_x else torch.empty(batch_size, 0, self.D, device=device)
                full_xc = torch.cat([uq_context_x, inner_xc], dim=1) 
                inner_yc = torch.cat(inner_context_y, dim=1) if inner_context_y else torch.empty(batch_size, 0, 1, device=device)
                full_yc = torch.cat([uq_context_y, inner_yc], dim=1) 
                full_xc = full_xc.to(device)
                full_yc = full_yc.to(device)
                batch = SimpleNamespace(
                    xc=full_xc,
                    yc=full_yc,
                    xt=context_x, 
                    yt=torch.zeros(batch_size, 1, 1, device=device)  
                )


                predicted_rewards = uq_model.predict(batch)
                inner_context_x.append(context_x)
                inner_context_y.append(predicted_rewards)

        # Concatenate all inner context data
        X = torch.cat(inner_context_x, dim=1)  
        y = torch.cat(inner_context_y, dim=1)  
        X = torch.cat([uq_context_x, X], dim=1)
        y = torch.cat([uq_context_y, y], dim=1)

        return X, y