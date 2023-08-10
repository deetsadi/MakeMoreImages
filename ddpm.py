from torchvision import transforms
from torch.nn import Module, ModuleList, Conv2d, BatchNorm2d, Linear, ReLU, ConvTranspose2d, Sequential
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import torchvision
import torch
import math
import matplotlib.pyplot as plt
import torch
import urllib
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path
import argparse
import os

# -------------------------
# Dataset Class
# -------------------------

class ImageDataset(Dataset):
    def __init__(self, images_dir, image_size):
        self.images = []
        
        for ext in [".png", ".jpg", ".jpeg"]:
            self.images = self.images + list(Path(images_dir).glob(f"*{ext}"))
        
        self.transforms = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)), # Normalizes to [-1, 1]
        ])
    
    def __len__(self): return len(self.images)
    
    def __getitem__(self, index): return self.transforms(Image.open(self.images[index]).convert('RGB'))

# -------------------------
# DDPM Components
# -------------------------

class Scheduler:
    def __init__(self, timesteps=1000, start_beta=0.001, end_beta=0.02, device='cuda'):
        """
        Initializing scheduling functions.
        
        beta_schedule : linearly spaced points from start_beta -> end_beta 
        alpha_schedule : 1 - betas
        alpha_hat : cumulative product of alpha_schedule
        sqrt_alpha_hat : sqrt(alpha_hat)
        sqrt_one_minus_alpha_hat : sqrt(1 - alpha_hat)
        
        ex. betas = [0, 0.1, 0.2, ...]
            alpha_schedule = [1, 0.9, 0.8, ...]
            alpha_hat = [1, 1 * 0.9, 1 * 0.9 * 0.8, ...]
            etc.

        Follows notation provided by Section 2, Equations 2 and 4 in the paper.
        """

        self.device = device

        self.start_beta = start_beta
        self.end_beta = end_beta
        self.timesteps = timesteps
        
        self.beta_schedule = torch.linspace(self.start_beta, self.end_beta, self.timesteps).to(self.device)
        self.alpha_schedule = 1 - self.beta_schedule
        self.alpha_hat = torch.cumprod(self.alpha_schedule, axis=0)
  
        self.sqrt_alpha_hat = torch.sqrt(self.alpha_hat)
        self.sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat)
    
    def forward(self, x_0, t):
        """
        Adds gaussian noise to image according to variance schedule.

        x_t = sqrt(alpha_hat_t) * x_0 + sqrt(1 - alpha_hat_t) * noise
        x_t = mean + variance

        Section 2, Equation 4
        """

        sqrt_alpha_hat_t = self.sqrt_alpha_hat[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_hat_t = self.sqrt_one_minus_alpha_hat[t].view(-1, 1, 1, 1)

        epsilon = torch.randn_like(x_0)
        epsilon.to(self.device)
        
        mean = sqrt_alpha_hat_t * x_0
        mean.to(self.device)
        variance = sqrt_one_minus_alpha_hat_t * epsilon
        variance.to(self.device)

        return mean + variance, epsilon
    
    def backward(self, model, i, x_t, t):
        """
        One step of the backward (denoising) process.

        e_theta is our model, which has learnt to predict noise from an image at timestep t. 
        x_(t-1) = 1 / sqrt(alpha_t) * (x_t - (beta_t / (sqrt(1 - alpha_hat_t))) * e_theta(x_t, t) ) + σ_t * noise

        Section 3.2, Algorithm 2 and Equation 11
        """

        sqrt_alpha_t = torch.sqrt(self.alpha_schedule)[t].view(-1, 1, 1, 1)
        beta_t = self.beta_schedule[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_hat_t = self.sqrt_one_minus_alpha_hat[t].view(-1, 1, 1, 1)
        sigma_t = torch.sqrt(self.beta_schedule)[t].view(-1, 1, 1, 1)

        if i > 1:
          random_noise = torch.randn_like(x_t)
        else: 
          random_noise = torch.zeros_like(x_t)

        mean = (1 / sqrt_alpha_t) * (x_t - ( (beta_t / sqrt_one_minus_alpha_hat_t) * model(x_t, t) )) 
        variance = sigma_t * random_noise

        return mean + variance

    def sample(self, model, num_samples, image_size, device):
        """
        Sample new images from the current model by starting with Gaussian noise and iteratively denoising.

        Section 3.2, Algorithm 2
        """
        model.eval()
        with torch.no_grad():
            x = torch.randn((num_samples, 3, image_size, image_size), device=device)

            for i in tqdm(reversed(range(1, self.timesteps))):
                t = torch.ones(num_samples, dtype=torch.long, device=device) * i

                x = self.backward(model, i, x, t)

        return x
    
class SinusoidalPositionEmbeddings(Module):
    """
    This class implementes positional embeddings, as introduced in the paper Attention is All You Need (Vaswani et. al). 
    These embeddings help our model learn denoising patterns based on the specific timestep in the diffusion process.
    Check out this link to learn more: https://machinelearningmastery.com/a-gentle-introduction-to-positional-encoding-in-transformer-models-part-1/
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        embeddings = math.log(10000) / (self.dim // 2 - 1)
        embeddings = torch.exp(torch.arange(self.dim // 2, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class Block(Module):
    def __init__(self, channels_in, channels_out, time_embedding_dims, num_filters = 3, downsample=True):
        super().__init__()

        # get the time embedding, pass it through a linear layer so it gets the correct number of channels
        self.time_embedding_dims = time_embedding_dims
        self.time_mlp = Sequential(*[
            SinusoidalPositionEmbeddings(time_embedding_dims),
            Linear(time_embedding_dims, channels_out),
            ReLU()
        ])
        
        # We can define 2 conv blocks that have a similar structure of Conv2d -> ReLU -> BatchNorm
        self.conv_block_1 = []
        self.conv_block_2 = []

        self.downsample = downsample

        if downsample:
            self.conv_block_1.append(Conv2d(channels_in, channels_out, num_filters, padding=1))
            self.final = Conv2d(channels_out, channels_out, 4, 2, 1)
        else:
            self.conv_block_1.append(Conv2d(2 * channels_in, channels_out, num_filters, padding=1))
            self.final = ConvTranspose2d(channels_out, channels_out, 4, 2, 1)
        
        self.conv_block_2.append(Conv2d(channels_out, channels_out, 3, padding=1))

        self.conv_block_1.append(ReLU())
        self.conv_block_2.append(ReLU())

        self.conv_block_1.append(BatchNorm2d(channels_out))
        self.conv_block_2.append(BatchNorm2d(channels_out))


        # Convert our lists of layers into a Sequential module
        self.conv_block_1 = Sequential(*self.conv_block_1)
        self.conv_block_2 = Sequential(*self.conv_block_2)

    def forward(self, x, t, **kwargs):
        o = self.conv_block_1(x)

        o_time = self.time_mlp(t)
        o = o + o_time[(..., ) + (None, ) * 2]

        o = self.conv_block_2(o)

        return self.final(o)

class UNet(Module):
    def __init__(self, img_channels = 3, time_embedding_dims = 128, sequence_channels = (64, 128, 256, 512, 1024)):
        super().__init__()
        self.time_embedding_dims = time_embedding_dims
        sequence_channels_rev = reversed(sequence_channels)

        # Conv layer to increase number of channels of input to the number expected by the first downsampling block
        self.conv1 = Conv2d(img_channels, sequence_channels[0], 3, padding=1)

        # 4 downsampling blocks in sequential order based on sequence_channels tuple
        self.downsampling = ModuleList([Block(channels_in, channels_out, time_embedding_dims) for channels_in, channels_out in zip(sequence_channels, sequence_channels[1:])])
        
        # 4 upsampling blocks in reverse order based on sequence_channels tuple
        self.upsampling = ModuleList([Block(channels_in, channels_out, time_embedding_dims, downsample=False) for channels_in, channels_out in zip(sequence_channels[::-1], sequence_channels[::-1][1:])])
        
        # Conv layer to decrease number of channels to the amount expected in the output image
        self.conv2 = Conv2d(sequence_channels[0], img_channels, 1)


    def forward(self, x, t, **kwargs):
        residuals = []
        o = self.conv1(x)
        for ds in self.downsampling:
            o = ds(o, t, **kwargs)
            residuals.append(o)
        for us, res in zip(self.upsampling, reversed(residuals)):
            o = us(torch.cat((o, res), dim=1), t, **kwargs)

        return self.conv2(o)

# -------------------------
# Helper Methods
# -------------------------
    
def tensor_2_image(x):
    reverse_transform = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2), # [0,1]
        transforms.Lambda(lambda t: t.permute(1, 2, 0)), # change to PIL format
        transforms.Lambda(lambda t: t * 255.), # [0.,255.]
        transforms.Lambda(lambda t: t.cpu().numpy().astype(np.uint8)), 
        transforms.ToPILImage(), 
    ])
    
    return reverse_transform(x)

def create_output_grid(samples, image_size, images_per_row, save_path, show=False):
    images = []
    for i in range(len(samples)):
        images.append(tensor_2_image(samples[i].cpu()))

    num_rows = len(images) // images_per_row + (1 if len(images) % images_per_row > 0 else 0)

    # Create a new figure and specify the number of rows and columns
    fig, axes = plt.subplots(num_rows, images_per_row, figsize=(15, 3*num_rows))

    # Loop through each image and corresponding subplot
    for i, (ax, image) in enumerate(zip(axes.flatten(), images)):
        # Resize the image to the specified size
        image = image.resize((image_size, image_size), Image.ANTIALIAS)
        
        # Display the image
        ax.imshow(image)
        ax.axis('off')

        if i >= len(images) - 1:
            break

    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    plt.savefig(save_path)

    if show: plt.show()
    
# -------------------------
# Train + Sampling
# -------------------------

def train(in_dir, out_dir, image_size, epochs, batch_size, resume, device='cuda', timesteps=1000, lr=1e-4):
    
    Path(out_dir).mkdir(parents=True, exist_ok=True)
  
    dataset = ImageDataset(in_dir, image_size)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    
    model = UNet()
    model.to(device)
    
    optimizer = Adam(model.parameters(), lr=lr)
    
    start_epoch = 0
    # Load the previously trained checkpoint if the path is provided
    if resume is not None:
        checkpoint = torch.load(resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        
    scheduler = Scheduler(timesteps=timesteps, device=device)
    
    print ("Starting training...")
    for epoch in range(start_epoch, epochs):
        epoch_losses = []
        for idx, batch in enumerate(tqdm(dataloader)):

            batch = batch.to('cuda')

            # Sample a random timestep to noise and denoise
            t = torch.randint(0, scheduler.timesteps, (len(batch),)).long().to(device)

            # Get the batch with Gaussian noise added, then predict the noise added using the UNet model
            batch_noisy, noise = scheduler.forward(batch, t)
            predicted_noise = model(batch_noisy, t)

            # Calculate the loss based on how accurately the UNet predicted the noise added to the original batch
            loss = F.mse_loss(noise, predicted_noise)

            epoch_losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f"Epoch: {epoch}, Loss: {np.mean(epoch_losses)}")
        if epoch % 20 == 0 and epoch > start_epoch:
            with torch.no_grad():
                # Sample images during training and save them so we get an idea of how our model improves over time
                samples = scheduler.sample(model, 10, image_size, device)
                model.train()
                create_output_grid(samples, image_size, 5, os.path.join(out_dir, f"epoch-{epoch}.png"), show=False)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, os.path.join(out_dir, f"epoch-{epoch}.pt"))

def sample(model_path, image_size, num_samples, out_dir, device='cuda', timesteps=1000, show=False,):
    model = UNet()
    model.to('cuda')
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    scheduler = Scheduler(timesteps=timesteps)
    samples = scheduler.sample(model, num_samples, image_size, device)

    create_output_grid(samples, image_size, 5, os.path.join(out_dir, f"samples.png"), show=False)

if __name__=="__main__":
    
    parser = argparse.ArgumentParser(
                    prog='MakeMoreImages',
                    description='Simple one script implementation of a DDPM from scratch.')
    
    parser.add_argument('--in_dir')
    parser.add_argument('--image_size')
    parser.add_argument('--out_dir', default="outputs") 
    parser.add_argument('--resume', default=None)
    parser.add_argument('--epochs', default=100)
    parser.add_argument('--batch_size', default=32)
    parser.add_argument('--sample', action='store_true')
    parser.add_argument('--model', default=None)
    parser.add_argument('--num_samples', default=10)
    parser.add_argument('--device', default='cuda')
    
    args = parser.parse_args()
    
    if args.sample:
        sample(args.model, int(args.image_size), int(args.num_samples), args.out_dir, device=args.device)
    else:
        train(args.in_dir, args.out_dir, int(args.image_size), int(args.epochs), int(args.batch_size), args.resume, device=args.device)
    
    
