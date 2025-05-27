from torch.optim import Adam
import torch
import torchvision
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import Model

model = Model()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model.to(device)

optimizer = Adam(model.parameters(), lr=0.001)
epochs = 1 # Try more!

IMG_SIZE = 32
BATCH_SIZE = 64
T = 300

data_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

data_transform = transforms.Compose([
    transforms.ToTensor(),
])

train = torchvision.datasets.FashionMNIST(root="./data", download=True, transform=data_transform, train=True)

test = torchvision.datasets.FashionMNIST(root="./data", download=True, transform=data_transform, train=False)


train_dataloader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)







for epoch in tqdm(range(epochs), desc="Training", unit="epoch"):
    for step, batch in enumerate(train_dataloader):
      optimizer.zero_grad()

      imgs, labels = batch
      imgs = imgs.to(device)
      labels = labels.to(device)

      t = torch.randint(0, T, (BATCH_SIZE,), device=device).long()
      imgs = torch.nn.functional.pad(imgs, (2, 2, 2, 2))  # Padding to make it compatible with the model input size
      loss = model.get_loss(model, imgs, t)
      loss.backward()
      optimizer.step()

    #   if epoch % 5 == 0 and step == 0:
    #     print(f"Epoch {epoch} | step {step:03d} Loss: {loss.item()} ")
    #     sample_plot_image()

# Save model
torch.save(model.state_dict(), "model.pth")
