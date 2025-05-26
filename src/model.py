import torch.nn as nn
import torch.nn.functional as F

from modules.unet.unet import UNet

class Model(nn.Module):
    def __init__(self, noise_scheduler: str = "linear"):
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
        self.unet = UNet(in_channels=3, num_classes=1) # TODO: Num of classes should not be here

    def get_loss(self, model, x_0, t):
        x_noisy, noise = self.noise_scheduler.add_noise(x_0, t)
        noise_pred = model(x_noisy, t)
        return F.l1_loss(noise, noise_pred)

    def forward(self):
        pass


if __name__ == "__main__":
    # Example usage
    model = Model()
    print(model)
    print("Num params: ", sum(p.numel() for p in model.parameters()))