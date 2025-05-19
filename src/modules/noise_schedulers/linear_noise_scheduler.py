import torch

"""
beta_start and beta_end are hyperparameters
"""
class LinearNoiseScheduler:
    def __init__(self, timesteps: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02):
        self.timesteps = timesteps

        # Start and end are the beta values for the linear noise schedule that we linearly interpolate between (hence linear scheduler)
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.betas = torch.linspace(beta_start, beta_end, timesteps)
        self.alphas = 1.0 - self.betas # Equation 1
        self.alpha_hat = torch.cumprod(self.alphas, dim=0)  # Equation 2

    def add_noise(self, x0: torch.Tensor, t: torch.Tensor):
        """
        Forward process: q(x_t | x_0)
        """
        epsilon = torch.randn_like(x0) # Input: x_0 - it returns the same size/shape as the input tensor (i.e. image)

        # gather alpha_bars for each sample in the batch
        alpha_bar_t = self.alpha_hat[t].view(-1, 1, 1, 1).to(x0.device)
        first_term = torch.sqrt(alpha_bar_t) * x0
        second_term = torch.sqrt(1 - alpha_bar_t) * epsilon
        return first_term + second_term # Equation 3