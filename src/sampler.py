import torch
import torch.nn as nn
import logging
from tqdm import tqdm

from linear_noise_scheduler import LinearNoiseScheduler

class Sampler:
    def __init__(self, img_size, device, t, noise_scheduler, unet):
        self.img_size = img_size
        self.device = device
        self.t = t
        self.noise_scheduler = noise_scheduler
        self.unet = unet
    
    def sample(self, num_of_samples: int, model, linear_noise_scheduler: LinearNoiseScheduler):
        logging.info(f"Sampling {num_of_samples}")

        # Check if model is in eval mode
        if model.training:
            raise ValueError("Model is in training mode. Please set it to eval mode before sampling ('model.eval()').")

        with torch.no_grad():
            x = torch.randn((num_of_samples, 3, self.img_size, self.img_size)).to(self.device)
            for i in tqdm(reversed(range(self.t + 1))):
                t = (torch.ones(num_of_samples) * i).long().to(self.device)
                pred_noise = model(x, t)

                if i > 1:
                    z = torch.randn_like(x)
                else:
                    z = torch.zeros_like(x)
                
                term_100 = ((1 - alpha_t) / (torch.sqrt(1 - alpha_hat)))
                term_2 = (x - term_100 * predicted_noise)
                term_1 = 1 / torch.sqrt(alpha_t)
                term_3 = torch.sqrt(linear_noise_scheduler.betas[t]) * z
                x_t_minus_1 = term_1 * term_2 + term_3

if __name__ == "__main__":
    # Example usage
    img_size = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t = 1000
    noise_scheduler = LinearNoiseScheduler()
    unet = nn.Module()  # Replace with actual UNet model

    sampler = Sampler(img_size, device, t, noise_scheduler, unet)
    num_of_samples = 10
    model = nn.Module()  # Replace with actual model

    model.eval()  # Set model to eval mode
    samples = sampler.sample(num_of_samples, model, noise_scheduler)