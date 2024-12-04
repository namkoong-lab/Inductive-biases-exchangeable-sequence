import torch
from collections import defaultdict

class LinUCBDisjoint:
    def __init__(self, alpha, d):
        """
        Initializes the LinUCB algorithm with disjoint linear models.

        Args:
            alpha (float): Exploration parameter.
            d (int): Dimension of feature vectors.
        """
        self.alpha = alpha
        self.d = d
        # Using defaultdict to handle new arms dynamically
        self.A = defaultdict(lambda: torch.eye(d))
        self.b = defaultdict(lambda: torch.zeros(d, 1))
        # To store the inverse of A for each arm to save computation
        self.A_inv = defaultdict(lambda: torch.eye(d))
    
    def select_arm(self, feature_matrix):
        """
        Selects the arm with the highest UCB score.

        Args:
            feature_matrix (torch.Tensor): Tensor of shape (num_arms, d) representing features of all arms.

        Returns:
            int: The index of the selected arm.
        """
        num_arms = feature_matrix.size(0)
        p = torch.zeros(num_arms)
        
        for a in range(num_arms):
            x_a = feature_matrix[a].unsqueeze(1)  # Shape: (d, 1)
            theta_a = torch.matmul(self.A_inv[a], self.b[a])  # Shape: (d, 1)
            mean = torch.matmul(theta_a.t(), x_a).item()
            uncertainty = self.alpha * torch.sqrt(torch.matmul(x_a.t(), torch.matmul(self.A_inv[a], x_a))).item()
            p[a] = mean + uncertainty
        
        # Select the arm with the highest p value
        a_t = torch.argmax(p).item()
        return a_t

    def update(self, a_t, x_t, r_t):
        """
        Updates the model parameters for the selected arm.

        Args:
            a_t (int): The index of the selected arm.
            x_t (torch.Tensor): Feature vector of the selected arm, shape (d,).
            r_t (float): Observed reward.
        """
        x_t = x_t.unsqueeze(1)  # Shape: (d, 1)
        self.A[a_t] += torch.matmul(x_t, x_t.t())
        self.b[a_t] += r_t * x_t
        # Update the inverse using the Sherman-Morrison formula for efficiency
        A_inv_a = self.A_inv[a_t]
        x = x_t
        A_inv_x = torch.matmul(A_inv_a, x)  # Shape: (d, 1)
        denominator = 1.0 + torch.matmul(x.t(), A_inv_x)  # Shape: (1, 1)
        numerator = torch.matmul(A_inv_x, A_inv_x.t())  # Shape: (d, d)
        self.A_inv[a_t] = A_inv_a - numerator / denominator
