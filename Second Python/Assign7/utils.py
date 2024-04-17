from glob import glob
from os import path
from typing import Optional

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

import torchvision.transforms as transforms
import torchvision.transforms.functional as functional

import torch

import torch.nn as nn

import os

from matplotlib import pyplot as plt

def prepare_image(image: np.ndarray, x: int, y: int, width: int, height: int, size: int) -> \
        tuple[np.ndarray, np.ndarray, np.ndarray]:
    if image.ndim < 2:
        raise ValueError("image must have shape (H, W)")

    if x < 0 or (x + width) > image.shape[-1]:
        raise ValueError(f"x={x} and width={width} do not fit into the image width={image.shape[-1]}")
    if y < 0 or (y + height) > image.shape[-2]:
        raise ValueError(f"y={y} and height={height} do not fit into the image height={image.shape[-2]}")
    
    area = (slice(y, y + height), slice(x, x + width))
    
    pixelated_image = pixelate(image, x, y, width, height, size)
    
    known_array = np.ones_like(image, dtype=bool)
    known_array[area] = False
    
    target_array = image.copy()
    
    return pixelated_image, known_array, target_array


def pixelate(image: np.ndarray, x: int, y: int, width: int, height: int, size: int) -> np.ndarray:
    image = image.copy() 
    curr_x = x
    while curr_x < x + width:
        curr_y = y
        while curr_y < y + height:
            block = (slice(curr_y, int(min(curr_y + size, y + height))),
                     slice(curr_x, int(min(curr_x + size, x + width))))
            if image[block].size > 0: 
                image[block] = int(np.mean(image[block]))
            curr_y += size
        curr_x += size
    return image




class RandomImagePixelationDataset(Dataset):
    
    def __init__(
            self,
            image_dir,
            width_range: tuple[int, int],
            height_range: tuple[int, int],
            size_range: tuple[int, int],
            dtype: Optional[type] = None
    ):
        self.image_files = sorted(path.abspath(f) for f in glob(path.join(image_dir, "**", "*.jpg"), recursive=True))
        self.width_range = width_range
        self.height_range = height_range
        self.size_range = size_range
        self.dtype = dtype
        self.transform_chain = transforms.Compose(
                    [transforms.Resize(size=64, interpolation=functional.InterpolationMode.BILINEAR),
                    transforms.CenterCrop(size=(64,64)),
                    transforms.Grayscale()]
                    )

    
    def __getitem__(self, index):
        with Image.open(self.image_files[index]) as im:
            #print("Shape of image before transform: ", np.array(im).shape)
            im = self.transform_chain(im)
            #print("Shape of image after transform: ", np.array(im).shape)
            image = np.array(im, dtype=self.dtype)
            #print(f"Image at index {index} shape: {image.shape}")

        image_width = image.shape[-1]
        image_height = image.shape[-2]
        rng = np.random.default_rng(seed=index)
        
        width = min(rng.integers(low=self.width_range[0], high=self.width_range[1], endpoint=True), image_width)
        height = min(rng.integers(low=self.height_range[0], high=self.height_range[1], endpoint=True), image_height)
        
        x = rng.integers(image_width - width, endpoint=True)
        y = rng.integers(image_height - height, endpoint=True)

        size = rng.integers(low=self.size_range[0], high=self.size_range[1], endpoint=True)
        
        pixelated_image, known_array, target_array = prepare_image(image, x, y, width, height, size)
        known_array = known_array.astype(float)
        return functional.to_tensor(pixelated_image), functional.to_tensor(known_array).float(), functional.to_tensor(target_array)

    def __len__(self):
        return len(self.image_files)




def stack_with_padding(batch_as_list: list):
    dtype_map = {
        torch.float32: np.float32,
        torch.float64: np.float64,
        torch.bool: np.bool

    }
    n = len(batch_as_list)
    pixelated_images_dtype = dtype_map[batch_as_list[0][0].dtype]
    known_arrays_dtype = dtype_map[batch_as_list[0][1].dtype]
    target_arrays_dtype = dtype_map[batch_as_list[0][2].dtype]
    shapes = []
    pixelated_images = []
    known_arrays = []
    target_arrays = []
    image_files = []

    for pixelated_image, known_array, target_array in batch_as_list:
        shapes.append(pixelated_image.shape)  
        pixelated_images.append(pixelated_image)
        known_arrays.append(known_array)
        target_arrays.append(target_array)


    max_shape = np.max(np.stack(shapes, axis=0), axis=0)
    stacked_pixelated_images = np.zeros(shape=(n, *max_shape), dtype=pixelated_images_dtype)
    stacked_known_arrays = np.ones(shape=(n, *max_shape), dtype=known_arrays_dtype)
    stacked_target_arrays = np.zeros(shape=(n, *max_shape), dtype=target_arrays_dtype)

    for i in range(n):
        channels, height, width = pixelated_images[i].shape
        stacked_pixelated_images[i, :channels, :height, :width] = pixelated_images[i]
        stacked_known_arrays[i, :channels, :height, :width] = known_arrays[i]
        channels, height, width = target_arrays[i].shape
        stacked_target_arrays[i, :channels, :height, :width] = target_arrays[i]
    return torch.from_numpy(stacked_pixelated_images), torch.from_numpy(stacked_known_arrays), torch.from_numpy(stacked_target_arrays)

def stack_with_padding_for_test(batch_as_list: list):

    dtype_map = {
        torch.float32: np.float32,
        torch.float64: np.float64,
        torch.bool: np.bool

    }
    n = len(batch_as_list)
    pixelated_images_dtype = dtype_map[batch_as_list[0][0].dtype]
    known_arrays_dtype = dtype_map[batch_as_list[0][1].dtype]
    shapes = []
    pixelated_images = []
    known_arrays = []


    for pixelated_image, known_array, target_array in batch_as_list:
        shapes.append(pixelated_image.shape)  
        pixelated_images.append(pixelated_image)
        known_arrays.append(known_array)


    max_shape = np.max(np.stack(shapes, axis=0), axis=0)
    stacked_pixelated_images = np.zeros(shape=(n, *max_shape), dtype=pixelated_images_dtype)
    stacked_known_arrays = np.ones(shape=(n, *max_shape), dtype=known_arrays_dtype)

    for i in range(n):
        channels, height, width = pixelated_images[i].shape
        stacked_pixelated_images[i, :channels, :height, :width] = pixelated_images[i]
        stacked_known_arrays[i, :channels, :height, :width] = known_arrays[i]
    return torch.from_numpy(stacked_pixelated_images), torch.from_numpy(stacked_known_arrays)
 


def plot_preds(inputs, path):

    os.makedirs(path, exist_ok=True)
    
    for i, input in enumerate(inputs):
        fig, ax = plt.subplots()
        ax.imshow(input, cmap="gray", interpolation="none")
        ax.set_axis_off()
        fig.savefig(os.path.join(path, f"{i:07d}.png"), dpi=100)
        plt.close(fig)



def plot(inputs, targets, predictions, path, update):

    os.makedirs(path, exist_ok=True)
    fig, axes = plt.subplots(ncols=3, figsize=(15, 5))
    
    for i in range(len(inputs)):
        for ax, data, title in zip(axes, [inputs, targets, predictions], ["Input", "Target", "Prediction"]):
            ax.clear()
            ax.set_title(title)
            ax.imshow(data[i, 0], cmap="gray", interpolation="none")
            ax.set_axis_off()
        fig.savefig(os.path.join(path, f"{update:07d}_{i:02d}.png"), dpi=100)
    
    plt.close(fig)



class DualInputCNN(nn.Module):
    def __init__(
            self,
            input_channels: int,
            hidden_channels: int,
            num_hidden_layers: int,
            use_batch_normalization: bool,
            num_classes: int,
            kernel_size: int = 3,
            activation_function: nn.Module = nn.ReLU()
    ):

        super().__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.num_hidden_layers = num_hidden_layers
        self.use_batch_normalization = use_batch_normalization
        self.num_classes = num_classes
        self.kernel_size = kernel_size
        self.activation_function = activation_function
        
        self.conv_layers1 = nn.ModuleList()
        self.conv_layers2 = nn.ModuleList()
        for i in range(num_hidden_layers):
            if i == 0: 
                in_channels = input_channels
            else: 
                in_channels = hidden_channels
            self.conv_layers1.append(nn.Conv2d(
                in_channels,
                hidden_channels,
                kernel_size,
                padding="same",
                padding_mode="zeros",
                bias=not use_batch_normalization
            ))
            self.conv_layers2.append(nn.Conv2d(
                in_channels,
                hidden_channels,
                kernel_size,
                padding="same",
                padding_mode="zeros",
                bias=not use_batch_normalization
            ))

        if self.use_batch_normalization:
            self.batch_norm_layers1 = nn.ModuleList()
            self.batch_norm_layers2 = nn.ModuleList()
            for _ in range(num_hidden_layers):
                self.batch_norm_layers1.append(nn.BatchNorm2d(hidden_channels))
                self.batch_norm_layers2.append(nn.BatchNorm2d(hidden_channels))
        self.output_layer = nn.Sequential(
    nn.ConvTranspose2d(hidden_channels * 2, hidden_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
    nn.ReLU(),
    nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=2, padding=1),  # Additional conv layer to reduce size
    nn.ReLU(),
    nn.Conv2d(hidden_channels, 1, kernel_size=3, padding=1),
)
    
    def forward(self, input_images1: torch.Tensor, input_images2: torch.Tensor) -> torch.Tensor:

        for i in range(self.num_hidden_layers):
            input_images1 = self.conv_layers1[i](input_images1)
            input_images2 = self.conv_layers2[i](input_images2)
            
            if self.use_batch_normalization:
                input_images1 = self.batch_norm_layers1[i](input_images1)
                input_images2 = self.batch_norm_layers2[i](input_images2)

            input_images1 = self.activation_function(input_images1)
            input_images2 = self.activation_function(input_images2)

        input_images = torch.cat([input_images1, input_images2], dim=1)
        input_images = self.output_layer(input_images)



        return input_images


class SimpleCNN(nn.Module):
    def __init__(
            self,
            input_channels: int,
            hidden_channels: int,
            num_hidden_layers: int,
            use_batch_normalization: bool,
            num_classes: int,
            kernel_size: int = 3,
            activation_function: nn.Module = nn.ReLU()
    ):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.num_hidden_layers = num_hidden_layers
        self.use_batch_normalization = use_batch_normalization
        self.num_classes = num_classes
        self.kernel_size = kernel_size
        self.activation_function = activation_function
        
        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(nn.Conv2d(
            input_channels,
            hidden_channels,
            kernel_size,
            padding="same",
            padding_mode="zeros",
            bias=not use_batch_normalization
        ))
        for i in range(1, num_hidden_layers):
            self.conv_layers.append(nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size,
                padding="same",
                padding_mode="zeros",
                bias=not use_batch_normalization
            ))
        if self.use_batch_normalization:
            self.batch_norm_layers = nn.ModuleList()
            for i in range(num_hidden_layers):
                self.batch_norm_layers.append(nn.BatchNorm2d(hidden_channels))
        self.output_layer = nn.Linear(self.hidden_channels * 64 * 64, self.num_classes)
    
    def forward(self, input_images: torch.Tensor) -> torch.Tensor:
        for i in range(self.num_hidden_layers):
            input_images = self.conv_layers[i](input_images)
            if self.use_batch_normalization:
                input_images = self.batch_norm_layers[i](input_images)
            input_images = self.activation_function(input_images)
        input_images = input_images.view(-1, self.hidden_channels * 64 * 64)
        input_images = self.output_layer(input_images)
        return input_images


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, pixelated_data, known_data, transform=None):
        self.pixelated_data = pixelated_data
        self.known_data = known_data
        self.transform = transform

    def __len__(self):
        return len(self.pixelated_data)

    def __getitem__(self, idx):
        pixelated_img = self.pixelated_data[idx]
        known_img = self.known_data[idx]
        if self.transform:
            pixelated_img = self.transform(pixelated_img)
            known_img = self.transform(known_img)
        return pixelated_img, known_img



def to_grayscale(pil_image: np.ndarray) -> np.ndarray:
    if pil_image.ndim == 2:
        return pil_image.copy()[None]
    if pil_image.ndim != 3:
        raise ValueError("image must have either shape (H, W) or (H, W, 3)")
    if pil_image.shape[2] != 3:
        raise ValueError(f"image has shape (H, W, {pil_image.shape[2]}), but it should have (H, W, 3)")
    
    rgb = pil_image / 255
    rgb_linear = np.where(
        rgb < 0.04045,
        rgb / 12.92,
        ((rgb + 0.055) / 1.055) ** 2.4
    )
    grayscale_linear = 0.2126 * rgb_linear[..., 0] + 0.7152 * rgb_linear[..., 1] + 0.0722 * rgb_linear[..., 2]
    
    grayscale = np.where(
        grayscale_linear < 0.0031308,
        12.92 * grayscale_linear,
        1.055 * grayscale_linear ** (1 / 2.4) - 0.055
    )
    grayscale = grayscale * 255

    if np.issubdtype(pil_image.dtype, np.integer):
        grayscale = np.round(grayscale)
    return grayscale.astype(pil_image.dtype)[None]
def stack_with_padding_for_test(batch):
    pixelated_images, known_arrays = zip(*batch)
    max_X = max(img.shape[-2] for img in pixelated_images)
    max_Y = max(img.shape[-1] for img in pixelated_images)
    batched_pixelated_images = torch.zeros(len(batch), max_X, max_Y)
    batched_known_arrays = torch.zeros(len(batch), max_X, max_Y)
    for i in range(len(batch)):
        X, Y = pixelated_images[i].shape[-2], pixelated_images[i].shape[-1]
        batched_pixelated_images[i, :X, :Y] = torch.from_numpy(pixelated_images[i])
        batched_known_arrays[i, :X, :Y] = torch.from_numpy(known_arrays[i])
    
    return batched_pixelated_images.unsqueeze(1), batched_known_arrays.unsqueeze(1)

def stack_images(batch_as_list: list):
    pixelated_images = [item[0] for item in batch_as_list]
    known_arrays = [item[1] for item in batch_as_list]
    target_arrays = [item[2] for item in batch_as_list]
    stacked_pixelated_images = torch.stack(pixelated_images, dim=0)
    stacked_known_arrays = torch.stack(known_arrays, dim=0)
    stacked_target_arrays = torch.stack(target_arrays, dim=0)

    return stacked_pixelated_images, stacked_known_arrays, stacked_target_arrays

class SimpleCNN(nn.Module):
    
    def __init__(self, n_in_channels: int = 1, n_hidden_layers: int = 3, n_kernels: int = 32, kernel_size: int = 7):
        super().__init__()
        
        cnn = []
        for i in range(n_hidden_layers):
            cnn.append(nn.Conv2d(
                in_channels=n_in_channels,
                out_channels=n_kernels,
                kernel_size=kernel_size,
                padding=kernel_size // 2
            ))
            cnn.append(nn.ReLU())
            n_in_channels = n_kernels
        self.hidden_layers = nn.Sequential(*cnn)
        
        self.output_layer = nn.Conv2d(
            in_channels=n_in_channels,
            out_channels=1,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )
    
    def forward(self, x):
        cnn_out = self.hidden_layers(x)
        predictions = self.output_layer(cnn_out)
        return predictions

