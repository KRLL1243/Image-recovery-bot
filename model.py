import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np
import time
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torchvision.utils import save_image

# Parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
model_save_path = 'unet_model.pth'
num_epochs = 5
learning_rate = 0.001
data_size_limit = 100000


# 1. Model Definition (U-Net)
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        # Encoder
        self.enc1 = self.conv_block(3, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)

        # Decoder
        self.dec4 = self.deconv_block(512, 256)
        self.dec3 = self.deconv_block(256, 128)
        self.dec2 = self.deconv_block(128, 64)
        self.dec1 = self.deconv_block(64, 3)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(out_channels)
        )

    def deconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)

        # Decoder
        dec4 = self.dec4(enc4)
        dec3 = self.dec3(dec4 + enc3)
        dec2 = self.dec2(dec3 + enc2)
        dec1 = self.dec1(dec2 + enc1)

        return dec1


# 2. Data preparation
class ImagePairDataset(Dataset):
    def __init__(self, img_paths, transform=None):
        self.img_paths = img_paths
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = Image.open(img_path)

        # Divide the image into left (blurred) and right (perfect) parts
        img_width, img_height = img.size
        left_img = img.crop((0, 0, img_width // 2, img_height))
        right_img = img.crop((img_width // 2, 0, img_width, img_height))

        if self.transform:
            left_img = self.transform(left_img)
            right_img = self.transform(right_img)

        return left_img, right_img


# Transformations for data
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Image folder path
img_dir = "prepared_dataset"
img_files = [os.path.join(img_dir, fname) for fname in os.listdir(img_dir) if fname.endswith('.jpg')]

if data_size_limit < len(img_files):
    img_files = img_files[:data_size_limit]


# Split dataset function
def split_dataset(image_files, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, test_size=10):
    assert train_ratio + val_ratio + test_ratio == 1.0, "The ratios should add up to a total of 1.0"

    train_files, temp_files = train_test_split(image_files, test_size=(val_ratio + test_ratio))
    val_files, test_files = train_test_split(temp_files, test_size=test_ratio / (val_ratio + test_ratio))

    if test_size != 0:
        test_files = test_files[:test_size]

    return train_files, val_files, test_files


# Splitting a dataset
train_files, val_files, test_files = split_dataset(image_files=img_files)
print(f"Training data size: {len(train_files)}, Validation data size: {len(val_files)}, Test data size: {len(test_files)}")

# Creating DataLoader's for train, val and test
train_dataset = ImagePairDataset(train_files, transform=transform)
val_dataset = ImagePairDataset(val_files, transform=transform)
test_dataset = ImagePairDataset(test_files, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)


# 3. Model training
def train():
    model = UNet().to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Lists for storing loss values
    train_losses = []
    val_losses = []

    # Epochs
    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        train_loss = 0
        steps_per_epoch = len(train_loader)

        with tqdm(total=steps_per_epoch, desc=f"Epoch {epoch + 1}/{num_epochs}", ncols=100) as pbar:
            for step, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(device), targets.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)

                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

                pbar.set_postfix(train_loss=train_loss / (step + 1), refresh=True)
                pbar.update(1)

        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        epoch_time = time.time() - start_time
        remaining_time = epoch_time * (num_epochs - (epoch + 1)) / (epoch + 1)
        remaining_minutes = remaining_time // 60
        remaining_seconds = remaining_time % 60
        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, "
              f"Time: {epoch_time:.2f}s, Estimated Remaining Time: {remaining_minutes:.0f}m {remaining_seconds:.0f}s")
        time.sleep(0.1)

    # Visualization of losses
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.show()

    # Saving the model
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved: {model_save_path}")


# 4. Testing and visualization of results
def test():
    model = UNet().to(device)
    model.load_state_dict(torch.load(model_save_path, map_location=device, weights_only=True))
    model.eval()

    os.makedirs("Results", exist_ok=True)  # Results folder

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            # Saving the restored, blurred and target image
            save_image(outputs, f"Results/fake_{i + 1}.png")
            save_image(inputs, f"Results/input_{i + 1}.png")
            save_image(targets, f"Results/real_{i + 1}.png")

            # Transformation for visualization
            inputs = inputs.cpu().numpy().transpose(0, 2, 3, 1)
            targets = targets.cpu().numpy().transpose(0, 2, 3, 1)
            outputs = outputs.cpu().numpy().transpose(0, 2, 3, 1)

            # Transform the data into the range [0, 1] for visualization
            inputs = np.clip(inputs[0], 0, 1)
            targets = np.clip(targets[0], 0, 1)
            outputs = np.clip(outputs[0], 0, 1)

            # Visualization
            fig, ax = plt.subplots(1, 3, figsize=(15, 5))
            ax[0].imshow(inputs)
            ax[0].set_title("Input (Corrupted)")
            ax[1].imshow(targets)
            ax[1].set_title("Ground Truth (Ideal)")
            ax[2].imshow(outputs)
            ax[2].set_title("Predicted (Restored)")

            for a in ax:
                a.axis('off')

            plt.show()


# Starting training
train()

# Starting testing
test()
