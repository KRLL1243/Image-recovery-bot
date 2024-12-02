import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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


# Function for testing a model on a new image
def test_on_new_image(model, image_path, output_path, device='cuda'):
    # 1. Uploading the image
    input_image = Image.open(image_path).convert('RGB')  # Open as RGB if the image is in a different format
    print(f"Input image size: {input_image.size}")

    # 2. Convert the image to the desired format for the model
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Convert the image to 256x256 size
        transforms.ToTensor(),  # Convert the image into a tensor
    ])

    # Applying transformations
    input_tensor = transform(input_image).unsqueeze(0).to(device)

    # 3. Running the image through the model
    model.eval()  # Set the model to evaluation mode (not training)
    with torch.no_grad():  # Not calculating gradients
        output_tensor = model(input_tensor)

    # 4. Convert the result into an image for saving
    output_image = output_tensor.cpu().numpy().squeeze(0)
    output_image = np.transpose(output_image, (1, 2, 0))

    # Limit the pixel values in the range [0, 1]
    output_image = np.clip(output_image, 0, 1)

    # Convert to range [0, 255] and type uint8
    output_image = (output_image * 255).astype(np.uint8)

    # Save the result
    restored_image = Image.fromarray(output_image)
    restored_image.save(output_path)

    # 5. Visualize the results
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(input_image)
    plt.title("Input Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(restored_image)
    plt.title("Restored Image")
    plt.axis('off')

    plt.show()


# Image path
image_path = 'my_test_data/face_bl_10.jpg'

# The path where the restored image will be saved
output_path = 'restored_image.png'

# Loading the model
model = UNet().to(device)
model.load_state_dict(torch.load(model_save_path, map_location=device, weights_only=True))
model.eval()

# Calling the test function
test_on_new_image(model, image_path, output_path, device)
