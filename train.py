import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from albumentations.pytorch import ToTensorV2
from dataset import ImageDataset
from generator import UNetGenerator
import albumentations as A
from discriminator import PatchGANDiscriminator

# Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 16
num_epochs = 100
lr = 0.0002
beta1 = 0.5
beta2 = 0.999

train_dir = r'C:\Users\itskp\Downloads\GAN\pix2pix\data\train'
val_dir = r'C:\Users\itskp\Downloads\GAN\pix2pix\data\val'

transform = A.Compose([
    A.Resize(256, 256),
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2()
])

train_dataset = ImageDataset(root_dir=train_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

generator = UNetGenerator().to(device)
discriminator = PatchGANDiscriminator().to(device)

optimizer_g = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, beta2))
optimizer_d = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, beta2))

criterion = nn.BCEWithLogitsLoss()

for epoch in range(num_epochs):
    for gray_images, color_images in train_loader:
        gray_images = gray_images.to(device)
        color_images = color_images.to(device)
        batch_size = gray_images.size(0)
        labels = torch.full((batch_size, 1, 30, 15), 1.0, dtype=torch.float, device=device)
        fake_labels = torch.full((batch_size, 1, 30, 15), 0.0, dtype=torch.float, device=device)

        # Train Discriminator
        optimizer_d.zero_grad()

        real_images = torch.cat((gray_images, color_images), dim=1)
        output = discriminator(real_images)
        d_loss_real = criterion(output, labels)
        d_loss_real.backward()

        fake_images = generator(gray_images)
        fake_images_combined = torch.cat((gray_images, fake_images), dim=1)
        output = discriminator(fake_images_combined.detach())
        d_loss_fake = criterion(output, fake_labels)
        d_loss_fake.backward()
        optimizer_d.step()

        # Train Generator
        optimizer_g.zero_grad()

        output = discriminator(torch.cat((gray_images, fake_images), dim=1))
        g_loss = criterion(output, labels)
        g_loss.backward()
        optimizer_g.step()

    print(f'Epoch [{epoch+1}/{num_epochs}]  d_loss: {d_loss_real.item() + d_loss_fake.item():.4f}  g_loss: {g_loss.item():.4f}')
