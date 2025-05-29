import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from modules.time_embedding.sinusodial_embedding import SinusoidalEmbedding

from modules.unet.unet import UNet

class Model(nn.Module):
    def __init__(self, noise_scheduler: str = "linear", input_channels: int = 1, num_classes: int = 3):
        super().__init__()

        # Noise scheduler selection
        if noise_scheduler == "linear":
            from modules.noise_schedulers.linear_noise_scheduler import LinearNoiseScheduler
            self.noise_scheduler = LinearNoiseScheduler()
        elif noise_scheduler == "cosine":
            from modules.noise_schedulers.cosine_noise_scheduler import CosineNoiseScheduler
            self.noise_scheduler = CosineNoiseScheduler()
        else:
            raise ValueError(f"Unknown noise scheduler: {noise_scheduler}")
        
        # U-Net selection
        self.unet = UNet(in_channels=input_channels, num_classes=num_classes) # TODO: Num of classes should not be here

        # Time embeddings
        self.time_embedding_dim = 256
        self.sinusoidalEmbedding = SinusoidalEmbedding(self.time_embedding_dim) # TODO: Do something with time embeddings


    def to(self, device):
        super().to(device)
        self.noise_scheduler.to(device)
        self.unet.to(device)
        self.sinusoidalEmbedding.to(device)
        return self

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        """
        x: Noisy images
        t: Timesteps tensor for the batch
        """
        # TODO: Do something with time embeddings
        time_embeddings = self.sinusoidalEmbedding(t)
        return self.unet(x)


if __name__ == "__main__":
    # Example usage
    model = Model()
    print(model)
    print("Num params: ", sum(p.numel() for p in model.parameters()))