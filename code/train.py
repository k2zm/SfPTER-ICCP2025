# %%
import os
import time
import csv
from datetime import datetime
import math
import glob
import numpy as np
from tqdm import tqdm
short_progress_bar="{l_bar}{bar:10}{r_bar}{bar:-10b}"

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from models.ours import Ours_Net
from models.oursmb import Ours_MB
from models.sfpw import SfPW_Net
from models.dsfp import DSfP_Net
from models.unet import UNet
from models.kondo import Kondo_Net

# Choose model and dataset
TRAIN_DATA_ROOT = "./data_train/ThermoPolSynth"
MODEL_NAME = "ours"  # "ours", "oursmb", "sfpw", "dsfp", "unet", "kondo"
WEIGHT_DIR = f"weights/{MODEL_NAME}"
os.makedirs(os.path.join(WEIGHT_DIR, "weights"), exist_ok=True)

MODEL_DICT = {
    "ours": {
        "model": Ours_Net,
    },
    "oursmb": {
        "model": Ours_MB,
    },
    "sfpw": {
        "model": SfPW_Net,
    },
    "dsfp": {
        "model": DSfP_Net,
    },
    "unet": {
        "model": UNet,
    },
    "kondo": {
        "model": Kondo_Net,
    },
}

# --- Hyperparameters ---
SAVE_INTERVAL = 10
NUM_WORKER = 8

BATCH_SIZE = 8
NUM_EPOCHS = 100
INIT_LR = 1e-4
LR_STEP = 10
LR_GAMMA = 0.5

CROP_SCALE = 0.7
IMAGE_SIZE = 256
DOLP_SCALER = 10
STOKES_BLEND_K_MIN = 0.60
STOKES_BLEND_K_MAX = 0.70
STOKES_SCALE_MEAN = 0.98
STOKES_SCALE_STD = 0.05
CROP_SCALE_MIN = 0.5
CROP_SCALE_MAX = 0.7
TRANSFORM_NOISE_MIN = -0.1
TRANSFORM_NOISE_MAX = 0.1
GAUSSIAN_NOISE_STD = 0.0013
ALPHA_THRESHOLD = 0.5
# --- End Hyperparameters ---

# %%
def main():
    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MODEL_DICT[MODEL_NAME]["model"](in_channels=8).to(device)

    # Loss function and optimizer
    cosine_similarity_fn = nn.CosineSimilarity(dim=1, eps=1e-8)
    def criterion(output, normal, mask):
        lossmap = 1 - cosine_similarity_fn(output, normal).unsqueeze(1)
        loss = (lossmap * mask).sum() / mask.sum()
        return loss
    optimizer = optim.Adam(model.parameters(), lr=INIT_LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=LR_STEP, gamma=LR_GAMMA)

    # Dataset and DataLoader
    train_dataset = TEPDataset(TRAIN_DATA_ROOT, mode="train")
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKER, shuffle=True)

    # Variables for recording
    train_losses = []
    log_file = os.path.join(WEIGHT_DIR, "training_log.csv")
    start_time = time.time()

    # tqdm bar
    with tqdm(total=NUM_EPOCHS, desc="Epoch: ", bar_format=short_progress_bar) as pbar:
        for epoch in range(NUM_EPOCHS):
            model.train()
            running_loss = 0.0
            epoch_start_time = time.time()
            batch_times = []
            samples_processed = 0
            
            # for batch_idx, (tensor, mask, normal) in enumerate(tqdm(train_loader, desc="Batch: ", bar_format=short_progress_bar, leave=False)):
            for batch_idx, (tensor, mask, normal) in enumerate(train_loader):
                batch_start_time = time.time()
                tensor, mask, normal = tensor.to(device), mask.to(device), normal.to(device)
                
                optimizer.zero_grad()
                output = model(tensor)
                output = F.normalize(output, p=2, dim=1)
                loss = criterion(output, normal, mask)
                loss.backward()
                optimizer.step()
                
                # Measuring batch processing time
                batch_end_time = time.time()
                batch_time = (batch_end_time - batch_start_time) * 1000  # Milliseconds
                batch_times.append(batch_time)
                
                # Counting the number of samples
                samples_processed += tensor.size(0)
                
                running_loss += loss.item()
            
            # Calculating various metrics
            epoch_loss = running_loss / len(train_loader)
            train_losses.append(epoch_loss)
            epoch_time = time.time() - epoch_start_time
            avg_batch_time = np.mean(batch_times)  # Average batch processing time (ms)
            samples_per_sec = samples_processed / epoch_time  # Number of samples processed per second
            
            # GPU memory usage
            if torch.cuda.is_available():
                gpu_mem_used = torch.cuda.memory_allocated() / 1024 / 1024  # MB
                gpu_mem_max = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
            else:
                gpu_mem_used = 0
                gpu_mem_max = 0
            
            # Cumulative time (minutes)
            elapsed_time = (time.time() - start_time) / 60
            
            # Logging metrics
            log_metrics(log_file, epoch+1, epoch_loss, avg_batch_time, samples_per_sec, 
                        gpu_mem_used, gpu_mem_max, elapsed_time)
            
            # Save model
            if (epoch + 1) % SAVE_INTERVAL == 0 or (epoch + 1) == NUM_EPOCHS:
                model_path = os.path.join(WEIGHT_DIR, "weights", f"model_{epoch+1}.pth")
                torch.save(model.state_dict(), model_path)

            # Update progress bar with more information
            pbar.set_postfix({
                "Loss": f"{epoch_loss:.4f}", 
                "Batch": f"{avg_batch_time:.1f}ms", 
                "Speed": f"{samples_per_sec:.1f}samp/s",
                "Mem": f"{gpu_mem_used:.0f}MB"
            })
            pbar.update(1)

            # Update learning rate
            scheduler.step()

    print("Training finished.")
    print(f"Results saved to: {WEIGHT_DIR}")
    print(f"Total training time: {elapsed_time:.2f} minutes")


# define dataset
def rotate_stokes(stokes, angle):
    # rotation matrix
    R = torch.tensor([[1, 0, 0],
                      [0, math.cos(2*angle), -math.sin(2*angle)],
                      [0, math.sin(2*angle), math.cos(2*angle)]])    
    stokes = torch.einsum('ij,jhw->ihw', R, stokes)
    return stokes

def rotate_normal(normal, angle):
    R = torch.tensor([[math.cos(angle), -math.sin(angle), 0],
                      [math.sin(angle), math.cos(angle), 0],
                      [0, 0, 1]])
    normal = torch.einsum('ij,jhw->ihw', R, normal)
    return normal

def get_transform_matrix(angle, crop_scale):
    """
    Returns an affine transformation matrix for rotation (angle: radians) and center cropping (extracting a region of the original image scale times).

    angle: Rotation angle (radians)
    crop_scale: The side length of the cropped region is scale times that of the original image.
    """
    cos_val = math.cos(angle)
    sin_val = math.sin(angle)
    
    transform_mat = torch.tensor([
         [ cos_val * crop_scale, -sin_val * crop_scale, np.random.uniform(TRANSFORM_NOISE_MIN, TRANSFORM_NOISE_MAX)],
         [ sin_val * crop_scale,  cos_val * crop_scale, np.random.uniform(TRANSFORM_NOISE_MIN, TRANSFORM_NOISE_MAX)]
    ])
    return transform_mat

def rotate_and_crop(image, transform_mat, output_size):
    """
    Applies the given affine transformation matrix to the input image and obtains an image of output size (output_size x output_size).

    image: Tensor, shape (C, H, W) or (N, C, H, W)
    transform_mat: Tensor, shape (2, 3)  Calculated by get_transform_matrix
    output_size: Height/width of the output image (square).
    """
    # Add batch dimension if it doesn't exist
    added_batch = False
    if image.dim() == 3:
        image = image.unsqueeze(0)  # (1, C, H, W)
        added_batch = True

    # Expand transform_mat to a shape with batch size 1
    transform_mat = transform_mat.unsqueeze(0)  # (1, 2, 3)
    
    # Create a grid of input coordinates corresponding to the output coordinates with affine_grid
    grid = F.affine_grid(transform_mat, size=(1, image.size(1), output_size, output_size), align_corners=True)
    
    # Apply transformation by grid_sample
    output_image = F.grid_sample(image, grid, align_corners=True, mode='bilinear')
    
    if added_batch:
        output_image = output_image.squeeze(0)
    
    return output_image    

def add_gaussian_noise(image, mean=0.0, std=0.1):
    noise = torch.randn_like(image) * std + mean
    noisy_image = image + noise
    return noisy_image

def blend_stokes(stokes_emit, stokes_ref, k):    
    stokes = stokes_emit + stokes_ref * k
    return stokes

class TEPDataset(Dataset):
    def __init__(self, root_dir, mode, num_variant=36):
        if mode not in ["train", "test"]:
            raise ValueError("mode must be 'train' or 'test'")
        
        self.root_dir = root_dir
        self.mode = mode
        self.objectlist = glob.glob(os.path.join(root_dir, "*"))
        self.num_variant = num_variant

    def __len__(self):
        if self.mode == "train":
            return len(self.objectlist) * int(self.num_variant)
        elif self.mode == "test":
            return len(self.objectlist)
    
    def __getitem__(self, idx):
        if self.mode == "train":
            idx_object = idx // self.num_variant
            idx_variant = idx % self.num_variant
        elif self.mode == "test":
            idx_object = idx
            idx_variant = 0

        # load npy files
        stokes_emit = np.load(file = os.path.join(self.objectlist[idx_object], "stokes_emit.npy"))[:,:,:3]
        stokes_ref = np.load(file = os.path.join(self.objectlist[idx_object], "stokes_ref.npy"))[:,:,:3]
        alpha = np.load(file = os.path.join(self.objectlist[idx_object], "alpha.npy")).astype(np.float32)
        normal = np.load(file = os.path.join(self.objectlist[idx_object], "normal.npy"))

        if self.mode == "train":
            # Blend the stokes vectors of emission and reflection
            stokes = blend_stokes(stokes_emit, stokes_ref, k=np.random.uniform(STOKES_BLEND_K_MIN, STOKES_BLEND_K_MAX)) # 0.6
            stokes = stokes * np.random.normal(loc=STOKES_SCALE_MEAN, scale=STOKES_SCALE_STD)

            # Convert ndarray to tensor
            stokes =  torch.from_numpy(stokes).permute(2, 0, 1)
            alpha = torch.from_numpy(alpha).unsqueeze(0)
            normal = torch.from_numpy(normal).permute(2, 0, 1)

            # Rotate the image and crop the center
            ## Calculate the transformation matrix
            rotate_angle = (2 * math.pi) * (idx_variant / self.num_variant)
            transform_mat = get_transform_matrix(rotate_angle, np.random.uniform(CROP_SCALE_MIN, CROP_SCALE_MAX)) # np.random.uniform(0.5, 0.7)
            ## Apply the transformation matrix
            stokes = rotate_and_crop(stokes, transform_mat, IMAGE_SIZE)
            alpha = rotate_and_crop(alpha, transform_mat, IMAGE_SIZE)
            normal = rotate_and_crop(normal, transform_mat, IMAGE_SIZE)
            ## Rotate the Stokes parameters and normal vector according to the image rotation
            stokes = rotate_stokes(stokes, rotate_angle)
            normal = rotate_normal(normal, rotate_angle)

            # Add noise
            stokes = add_gaussian_noise(stokes, std=GAUSSIAN_NOISE_STD)

        elif self.mode == "test":
            stokes = blend_stokes(stokes_emit, stokes_ref, k=0.6)

            # ndarray to tensor
            stokes = torch.from_numpy(stokes).permute(2, 0, 1)
            alpha = torch.from_numpy(alpha).unsqueeze(0)
            normal = torch.from_numpy(normal).permute(2, 0, 1)

            # no transform, just resize
            transform_mat = get_transform_matrix(angle=0, crop_scale=1.0)
            stokes = rotate_and_crop(stokes, transform_mat, IMAGE_SIZE)
            alpha = rotate_and_crop(alpha, transform_mat, IMAGE_SIZE)
            normal = rotate_and_crop(normal, transform_mat, IMAGE_SIZE)

        # make tensor
        pol1 = (stokes[0] + stokes[1]) / 2
        pol2 = (stokes[0] - stokes[1]) / 2
        pol3 = (stokes[0] + stokes[2]) / 2
        pol4 = (stokes[0] - stokes[2]) / 2
        intensity = stokes[0]
        dolp = torch.sqrt(stokes[1]**2 + stokes[2]**2) / (stokes[0] + 1e-6) * DOLP_SCALER
        aolp1 = stokes[1] / torch.sqrt(stokes[1]**2 + stokes[2]**2 + 1e-6)
        aolp2 = stokes[2] / torch.sqrt(stokes[1]**2 + stokes[2]**2 + 1e-6)
        tensor = torch.stack([pol1, pol2, pol3, pol4, intensity, dolp, aolp1, aolp2], dim=0)

        # apply mask
        mask = alpha > ALPHA_THRESHOLD
        tensor = torch.where(mask, tensor, torch.zeros_like(tensor))
        normal = torch.where(mask, normal, torch.zeros_like(normal))
        tensor = torch.nan_to_num(tensor, nan=0.0)
        normal = torch.nan_to_num(normal, nan=0.0)

        return tensor, mask, normal
    
def log_metrics(log_file, epoch, loss, batch_time, samples_per_sec, gpu_mem_used, gpu_mem_max, elapsed_time):
    """Function to log metrics to a file"""
    if not os.path.exists(log_file):
        with open(log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Epoch', 'Loss', 'Batch_Time_ms', 'Samples_per_Sec', 
                            'GPU_Mem_Used_MB', 'GPU_Mem_Max_MB', 'Elapsed_Time_min'])
    
    with open(log_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([epoch, f"{loss:.4f}", f"{batch_time:.2f}", f"{samples_per_sec:.2f}", 
                        f"{gpu_mem_used:.2f}", f"{gpu_mem_max:.2f}", f"{elapsed_time:.2f}"])


if __name__ == "__main__":
    main()


# %%
