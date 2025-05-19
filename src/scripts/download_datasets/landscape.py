import kagglehub
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# URL: https://www.kaggle.com/datasets/arnaud58/landscape-pictures

# Download latest version
path = kagglehub.dataset_download("arnaud58/landscape-pictures")

print("Path to dataset files:", path)

# C:\Users\Shlomi\.cache\kagglehub\datasets\arnaud58\landscape-pictures\versions\2
