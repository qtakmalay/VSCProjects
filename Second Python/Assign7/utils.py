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
        # This is actually more general than the assignment specification
        raise ValueError("image must have shape (H, W)")
    #print(width, height, size)

    if x < 0 or (x + width) > image.shape[-1]:
        raise ValueError(f"x={x} and width={width} do not fit into the image width={image.shape[-1]}")
    if y < 0 or (y + height) > image.shape[-2]:
        raise ValueError(f"y={y} and height={height} do not fit into the image height={image.shape[-2]}")
    
    # The (height, width) slices to extract the area that should be pixelated. Since we
    # need this multiple times, specify the slices explicitly instead of using [:] notation
    area = (slice(y, y + height), slice(x, x + width))
    
    # This returns already a copy, so we are independent of "image"
    pixelated_image = pixelate(image, x, y, width, height, size)
    
    known_array = np.ones_like(image, dtype=bool)
    known_array[area] = False
    
    # Create a copy to avoid that "target_array" and "image" point to the same array
    target_array = image[area].copy()
    
    return pixelated_image, known_array, target_array


def pixelate(image: np.ndarray, x: int, y: int, width: int, height: int, size: int) -> np.ndarray:
    image = image.copy() # Need a copy since we overwrite data directly
    curr_x = x
    while curr_x < x + width:
        curr_y = y
        while curr_y < y + height:
            block = (slice(curr_y, int(min(curr_y + size, y + height))),
                     slice(curr_x, int(min(curr_x + size, x + width))))
            if image[block].size > 0: # Check if the array is not empty
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
        
        # Create RNG in each __getitem__ call to ensure reproducibility even in
        # environments with multiple threads and/or processes
        rng = np.random.default_rng(seed=index)
        
        # Both width and height can be arbitrary, but they must not exceed the
        # actual image width and height
        width = min(rng.integers(low=self.width_range[0], high=self.width_range[1], endpoint=True), image_width)
        height = min(rng.integers(low=self.height_range[0], high=self.height_range[1], endpoint=True), image_height)
        
        # Ensure that x and y always fit with the randomly chosen width and
        # height (and not throw an error in "prepare_image")
        x = rng.integers(image_width - width, endpoint=True)
        y = rng.integers(image_height - height, endpoint=True)
        # Block size can be arbitrary again
        size = rng.integers(low=self.size_range[0], high=self.size_range[1], endpoint=True)
        
        pixelated_image, known_array, target_array = prepare_image(image, x, y, width, height, size)
        known_array = known_array.astype(float)
        #print("get item pixelated_images.shape:", pixelated_image.shape)
        #print("get item known_arrays.shape:", known_array.shape)
        #print("get item target_arrays.shape:", target_array.shape)
        #print("get item concatenate pixelated_images and known arrays:", np.concatenate([pixelated_image, known_array]).shape)
        #print("get item concatenate target_arrays.shape and target arrays:", np.concatenate([target_array, target_array]).shape)
        return functional.to_tensor(pixelated_image), functional.to_tensor(known_array).float(), functional.to_tensor(target_array)

        return (np.concatenate([pixelated_image, known_array]), target_array) #, functional.to_tensor(target_array), self.image_files[index]
    
    def __len__(self):
        return len(self.image_files)




def stack_with_padding(batch_as_list: list):
    # Expected list elements are 4-tuples:
    # (pixelated_image, known_array, target_array, image_file)
    dtype_map = {
        torch.float32: np.float32,
        torch.float64: np.float64,
        torch.bool: np.bool
        # Add other dtype mappings as needed
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
        shapes.append(pixelated_image.shape)  # Equal to known_array.shape
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
    #print("stacked_pixelated_images.shape:", stacked_pixelated_images.shape)
    #print("stacked_known_arrays.shape:", stacked_known_arrays.shape)
    #print("stacked_target_arrays.shape:", stacked_target_arrays.shape)
    return torch.from_numpy(stacked_pixelated_images), torch.from_numpy(stacked_known_arrays), torch.from_numpy(stacked_target_arrays)
    # return (torch.from_numpy(stacked_pixelated_images),
    #         torch.from_numpy(stacked_known_arrays),
    #         torch.from_numpy(stacked_target_arrays))






def plot(inputs, targets, predictions, path, update):
    """Plotting the inputs, targets and predictions to file ``path``."""
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
            if i == 0:  # For the first layer, use input_channels
                in_channels = input_channels
            else:  # For subsequent layers, use hidden_channels
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
        #print(f"Input shapes: {input_images1.shape}, {input_images2.shape}")  # Check input shapes

        for i in range(self.num_hidden_layers):
            input_images1 = self.conv_layers1[i](input_images1)
            input_images2 = self.conv_layers2[i](input_images2)
            
            #print(f"Output shape after conv layer {i}: {input_images1.shape}, {input_images2.shape}")  # Check output shapes after each conv layer

            if self.use_batch_normalization:
                input_images1 = self.batch_norm_layers1[i](input_images1)
                input_images2 = self.batch_norm_layers2[i](input_images2)

            input_images1 = self.activation_function(input_images1)
            input_images2 = self.activation_function(input_images2)

        # Concatenate along the channel dimension
        input_images = torch.cat([input_images1, input_images2], dim=1)
        input_images = self.output_layer(input_images)

        #print(f"Final output shape: {input_images.shape}")  # Check final output shape

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
        # We already added one conv layer, so start the range from 1 instead of 0
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

