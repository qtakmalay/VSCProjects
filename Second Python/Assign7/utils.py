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
    target_array = image.copy()
    
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




def plot_preds(inputs, path):
    """Plotting the inputs to file ``path``."""
    os.makedirs(path, exist_ok=True)
    
    for i, input in enumerate(inputs):
        fig, ax = plt.subplots()
        #input = np.squeeze(input)  # remove single-dimensional entries from the shape of the array
        ax.imshow(input, cmap="gray", interpolation="none")
        ax.set_axis_off()
        fig.savefig(os.path.join(path, f"{i:07d}.png"), dpi=100)
        plt.close(fig)



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

# def stack_with_padding(batch_as_list: list):
#     # Expected list elements are 4-tuples:
#     # (pixelated_image, known_array, target_array, image_file)
#     n = len(batch_as_list)
#     pixelated_images_dtype = batch_as_list[0][0].dtype  # Same for every sample
#     known_arrays_dtype = batch_as_list[0][1].dtype
#     shapes = []
#     pixelated_images = []
#     known_arrays = []
#     target_arrays = []
#     image_files = []
    
#     for pixelated_image, known_array, target_array, image_file in batch_as_list:
#         shapes.append(pixelated_image.shape)  # Equal to known_array.shape
#         pixelated_images.append(pixelated_image)
#         known_arrays.append(known_array)
#         target_arrays.append(torch.from_numpy(target_array))
#         image_files.append(image_file)
    
#     max_shape = np.max(np.stack(shapes, axis=0), axis=0)
#     stacked_pixelated_images = np.zeros(shape=(n, *max_shape), dtype=pixelated_images_dtype)
#     stacked_known_arrays = np.ones(shape=(n, *max_shape), dtype=known_arrays_dtype)
    
#     for i in range(n):
#         channels, height, width = pixelated_images[i].shape
#         stacked_pixelated_images[i, :channels, :height, :width] = pixelated_images[i]
#         stacked_known_arrays[i, :channels, :height, :width] = known_arrays[i]
    
#     return torch.from_numpy(stacked_pixelated_images), torch.from_numpy(
#         stacked_known_arrays), target_arrays, image_files

# img_data = np.where(known_img_np[0], 255, 0).astype(np.uint8)

# pix_img_pil = [Image.fromarray(np.squeeze(i).astype(np.uint8)) for i in pix_img_np]
# trans_pix_img_pil = [transform_chain(i) for i in pix_img_pil]
# fig, axes = plt.subplots(1, 2)
# axes[0].imshow(trans_pix_img_pil[0])
# axes[1].imshow(trans_pix_img_pil[1])
# # axes[2].imshow(trans_pix_img_pil[2])
# # axes[3].imshow(trans_pix_img_pil[3])
# # axes[4].imshow(trans_pix_img_pil[4])
# fig.tight_layout()
# plt.show()

# img = Image.fromarray(np.squeeze(img_data).astype(np.uint8))

# # Display the image
# img.show()

# class SimpleNetwork(nn.Module):
    
#     def __init__(
#             self,
#             input_neurons: int,
#             hidden_neurons: int,
#             output_neurons: int,
#             activation_function: nn.Module = nn.ReLU()
#     ):
#         super().__init__()
#         self.input_neurons = input_neurons
#         self.hidden_neurons = hidden_neurons
#         self.output_neurons = output_neurons
#         self.activation_function = activation_function
        
#         self.input_layer = nn.Linear(self.input_neurons, self.hidden_neurons)
#         self.hidden_layer_1 = nn.Linear(self.hidden_neurons, self.hidden_neurons)
#         self.hidden_layer_2 = nn.Linear(self.hidden_neurons, self.hidden_neurons)
#         self.output_layer = nn.Linear(self.hidden_neurons, self.output_neurons)
    
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.activation_function(self.input_layer(x))
#         x = self.activation_function(self.hidden_layer_1(x))
#         x = self.activation_function(self.hidden_layer_2(x))
#         x = self.output_layer(x)
#         return x



# class TrainDataset(torch.utils.data.Dataset):
#    def __init__(self, image_dir):
#       self.image_dir = image_dir
#       self.list_files = glob.glob(os.path.join(image_dir, "**", "*.jpg"), recursive=True)
#    def __getitem__(self, index: int):
#       return (Image.open(self.list_files[index]), index)
#    def __len__(self):
#       return len(self.list_files)
   
# class TestDataset(torch.utils.data.Dataset):
#     def __init__(self, pixelated_data, known_data, transform=None):
#         self.pixelated_data = pixelated_data
#         self.known_data = known_data
#         self.transform = transform

#     def __len__(self):
#         return len(self.pixelated_data)

#     def __getitem__(self, idx):
#         pixelated_img = self.pixelated_data[idx]
#         known_img = self.known_data[idx]
#         if self.transform:
#             pixelated_img = self.transform(pixelated_img)
#             known_img = self.transform(known_img)
#         return pixelated_img, known_img

# # Training loop
# for epoch in range(epochs):
#     model.train()
#     for pixelated_images, known_arrays, target_arrays, _ in train_loader:
#         # Combine pixelated_images and known_arrays as channels and move to device
#         inputs = torch.cat((pixelated_images.unsqueeze(1), known_arrays.unsqueeze(1)), dim=1).to(device)
#         targets = target_arrays.to(device)
#         mask = (~known_arrays).to(device)

#         # Forward pass
#         outputs = model(inputs.squeeze(2))
#         masked_outputs = outputs * mask
#         masked_targets = targets * mask
#         loss = criterion(masked_outputs, masked_targets)

#         # Backward pass and optimization
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#     print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")
#     model.eval()
#     with torch.no_grad():
#         total_loss = 0
#         for pixelated_images, known_arrays, target_arrays, _ in valid_loader:
#             # Combine pixelated_images and known_arrays as channels and move to device
#             inputs = torch.cat((pixelated_images.unsqueeze(1), known_arrays.unsqueeze(1)), dim=1).to(device)
#             targets = target_arrays.to(device)
#             mask = (~known_arrays).to(device)

#             # Forward pass and loss computation
#             outputs = model(inputs.squeeze(2))
#             masked_outputs = outputs * mask
#             masked_targets = targets * mask
#             loss = criterion(masked_outputs, masked_targets)

#             total_loss += loss.item()

#         print(f"Validation Loss: {total_loss / len(valid_loader)}")

# Training loop
# for epoch in range(epochs):
#     model.train()
#     for pixelated_images, known_arrays, target_arrays, _ in train_loader:
#         # Combine pixelated_images and known_arrays as channels and move to device
#         inputs = torch.cat((pixelated_images.unsqueeze(1), known_arrays.unsqueeze(1)), dim=1).to(device)
#         targets = target_arrays.to(device)

#         # Forward pass
#         outputs = model(inputs.squeeze(2))
#         loss = criterion(outputs, targets)

#         # Backward pass and optimization
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#     print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")
#     model.eval()
#     with torch.no_grad():
#         total_loss = 0
#         for pixelated_images, known_arrays, target_arrays, _ in valid_loader:
#             # Combine pixelated_images and known_arrays as channels and move to device
#             inputs = torch.cat((pixelated_images.unsqueeze(1), known_arrays.unsqueeze(1)), dim=1).to(device)
#             targets = target_arrays.to(device)

#             # Forward pass and loss computation
#             outputs = model(inputs.squeeze(2))
#             loss = criterion(outputs, targets)

#             total_loss += loss.item()

#         print(f"Validation Loss: {total_loss / len(valid_loader)}")

# class DualInputCNN(nn.Module):
#     def __init__(self, n_in_channels: int = 1, n_hidden_layers: int = 3, n_kernels: int = 32, kernel_size: int = 7):
#         super(DualInputCNN, self).__init__()

#         self.cnn1 = self._build_cnn(n_in_channels, n_hidden_layers, n_kernels, kernel_size)
#         self.cnn2 = self._build_cnn(n_in_channels, n_hidden_layers, n_kernels, kernel_size)

#         self.fc = nn.Linear(n_kernels*2, 1)

#     def _build_cnn(self, n_in_channels, n_hidden_layers, n_kernels, kernel_size):
#         cnn = []
#         for i in range(n_hidden_layers):
#             cnn.append(nn.Conv2d(in_channels=n_in_channels, out_channels=n_kernels, kernel_size=kernel_size, padding=kernel_size // 2))
#             cnn.append(nn.ReLU())
#             n_in_channels = n_kernels
#         cnn.append(nn.Conv2d(in_channels=n_in_channels, out_channels=1, kernel_size=kernel_size, padding=kernel_size // 2))
#         return nn.Sequential(*cnn)

#     def forward(self, x1, x2):
#         x1 = self.cnn1(x1)
#         x2 = self.cnn2(x2)

#         print(f"Shape of x1 after cnn1: {x1.shape}")
#         print(f"Shape of x2 after cnn2: {x2.shape}")

#         x = torch.cat((x1, x2), dim=1)
#         x = x.view(x.size(0), -1)

#         print(f"Shape of x before fc: {x.shape}")

#         out = self.fc(x)
#         return out

# class DualInputCNN(nn.Module):
#     def __init__(self, n_in_channels: int = 1, n_hidden_layers: int = 3, n_kernels: int = 32, kernel_size: int = 7):
#         super(DualInputCNN, self).__init__()

#         self.cnn1 = self._build_cnn(n_in_channels, n_hidden_layers, n_kernels, kernel_size)
#         self.cnn2 = self._build_cnn(n_in_channels, n_hidden_layers, n_kernels, kernel_size)

#         self.fc = nn.Linear(n_kernels*2, 1)

#     def _build_cnn(self, n_in_channels, n_hidden_layers, n_kernels, kernel_size):
#         cnn = []
#         for i in range(n_hidden_layers):
#             cnn.append(nn.Conv2d(in_channels=n_in_channels, out_channels=n_kernels, kernel_size=kernel_size, padding=kernel_size // 2))
#             cnn.append(nn.ReLU())
#             n_in_channels = n_kernels
#         cnn.append(nn.Conv2d(in_channels=n_in_channels, out_channels=1, kernel_size=kernel_size, padding=kernel_size // 2))
#         cnn.append(nn.AvgPool2d(kernel_size=64))  # Added pooling layer
#         return nn.Sequential(*cnn)

#     def forward(self, x1, x2):
#         x1 = self.cnn1(x1)
#         x2 = self.cnn2(x2)

#         print(f"Shape of x1 after cnn1: {x1.shape}")
#         print(f"Shape of x2 after cnn2: {x2.shape}")

#         x = torch.cat((x1, x2), dim=1)
#         x = x.view(x.size(0), -1)

#         print(f"Shape of x before fc: {x.shape}")

#         out = self.fc(x)
#         return out


# class DualInputCNN(nn.Module):
#     def __init__(self, n_in_channels: int = 1, n_hidden_layers: int = 3, n_kernels: int = 32, kernel_size: int = 7):
#         super(DualInputCNN, self).__init__()

#         self.cnn1 = self._build_cnn(n_in_channels, n_hidden_layers, n_kernels, kernel_size)
#         self.cnn2 = self._build_cnn(n_in_channels, n_hidden_layers, n_kernels, kernel_size)

#         self.fc = nn.Linear(n_kernels*2, 1)

#     def _build_cnn(self, n_in_channels, n_hidden_layers, n_kernels, kernel_size):
#         cnn = []
#         for i in range(n_hidden_layers):
#             cnn.append(nn.Conv2d(in_channels=n_in_channels, out_channels=n_kernels, kernel_size=kernel_size, padding=kernel_size // 2))
#             cnn.append(nn.ReLU())
#             n_in_channels = n_kernels
#         cnn.append(nn.Conv2d(in_channels=n_in_channels, out_channels=1, kernel_size=kernel_size, padding=kernel_size // 2))
#         cnn.append(nn.AdaptiveAvgPool2d((1,1)))  # Added adaptive pooling layer
#         return nn.Sequential(*cnn)

#     def forward(self, x1, x2):
#         x1 = self.cnn1(x1)
#         x2 = self.cnn2(x2)

#         x = torch.cat((x1, x2), dim=1)
#         x = x.view(x.size(0), -1)

#         out = self.fc(x)
#         return out

# def stack_with_padding_for_test(batch):
#     # batch is a list of data points, each data point being a tuple of 
#     # (pixelated_image, known_array)

#     # unzip the batch to get separate lists of pixelated_images and known_arrays
#     pixelated_images, known_arrays = zip(*batch)
    
#     # compute maximum X and Y dimensions
#     max_X = max(img.shape[-2] for img in pixelated_images)
#     max_Y = max(img.shape[-1] for img in pixelated_images)
    
#     # create zero tensors
#     batched_pixelated_images = torch.zeros(len(batch), max_X, max_Y)
#     batched_known_arrays = torch.zeros(len(batch), max_X, max_Y)
    
#     # copy data into zero tensors
#     for i in range(len(batch)):
#         X, Y = pixelated_images[i].shape[-2], pixelated_images[i].shape[-1]
#         batched_pixelated_images[i, :X, :Y] = pixelated_images[i]
#         batched_known_arrays[i, :X, :Y] = known_arrays[i]
    
#     return batched_pixelated_images.unsqueeze(1), batched_known_arrays.unsqueeze(1)

# def sequence_collate_fn(batch):
#     max_x = max([sample[0].shape[0] for sample in batch])
#     max_y = max([sample[0].shape[1] for sample in batch])
#     n_samples = len(batch)
    
#     stacked_images = torch.zeros((n_samples, max_x, max_y))
#     targets = []
    
#     max_target_h = max([sample[1].shape[0] for sample in batch])
#     max_target_w = max([sample[1].shape[1] for sample in batch])
    
#     for i, sample in enumerate(batch):
#         image, target = sample
#         h, w = image.shape[0], image.shape[1]
#         stacked_images[i, :h, :w] = torch.from_numpy(image)
        
#         target_h, target_w = target.shape
#         pad_h = max_target_h - target_h
#         pad_w = max_target_w - target_w
#         padded_target = functional.pad(img=torch.tensor(target), padding=(0, pad_w, 0, pad_h), padding_mode="constant", fill=0)
#         targets.append(padded_target)

#     targets = torch.stack(targets, dim=0)

#     return stacked_images, targets

# def sequence_collate_fn(batch: list):
#     """
#     This function will be used to pad the images to the max size in the batch and combine them into a single tensor.
#     """
#     print("Batch shapes:", [sample[0].shape for sample in batch])
#     max_x = max([sample[0].shape[1] for sample in batch])
#     max_y = max([sample[0].shape[2] for sample in batch])
#     n_channels = batch[0][0].shape[0]
    
#     n_samples = len(batch)

#     # creating a tensor filled with zeros, big enough to hold all images
#     stacked_images = torch.zeros((n_samples, n_channels, max_x, max_y))
#     targets = []

#     for i, sample in enumerate(batch):
#         image, target = sample
#         h, w = image.shape[1], image.shape[2]
#         stacked_images[i, :, :h, :w] = torch.from_numpy(image)
#         targets.append(torch.from_numpy(target))

#     targets = torch.stack(targets)

#     return stacked_images, targets

# def sequence_collate_fn(batch_as_list: list):
#     #
#     # Handle sequences
#     #
#     # Get sequence entries, which are at index 0 in each sample tuple
#     sequences = [sample[0] for sample in batch_as_list]
#     # Get the maximum sequence length in the current minibatch
#     max_seq_len = np.max([seq.shape[0] for seq in sequences])
#     # Allocate a tensor that can fit all padded sequences
#     n_seq_features = sequences[0].shape[1]  # Could be hard-coded to 3
#     stacked_sequences = torch.zeros(size=(len(sequences), max_seq_len, n_seq_features), dtype=torch.float32)
#     # Write the sequences into the tensor stacked_sequences
#     for i, sequence in enumerate(sequences):
#         stacked_sequences[i, :len(sequence), :] = torch.from_numpy(sequence)
    
#     #
#     # Handle labels
#     #
#     # Get label entries, which are at index 1 in each sample tuple
#     labels = [sample[1] for sample in batch_as_list]
#     # Convert them to tensors and stack them
#     stacked_labels = torch.stack([torch.tensor(label, dtype=torch.float32) for label in labels], dim=0)
    
#     return stacked_sequences, stacked_labels


class TestDataset(Dataset):
    def __init__(self, pixelated_images, known_arrays):
        self.pixelated_images = pixelated_images
        self.known_arrays = known_arrays

    def __len__(self):
        return len(self.pixelated_images)

    def __getitem__(self, idx):
        return functional.to_tensor(self.pixelated_images[idx]), functional.to_tensor(self.known_arrays[idx])


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
    # batch is a list of data points, each data point being a tuple of 
    # (pixelated_image, known_array)

    # unzip the batch to get separate lists of pixelated_images and known_arrays
    pixelated_images, known_arrays = zip(*batch)
    
    # compute maximum X and Y dimensions
    max_X = max(img.shape[-2] for img in pixelated_images)
    max_Y = max(img.shape[-1] for img in pixelated_images)
    
    # create zero tensors
    batched_pixelated_images = torch.zeros(len(batch), max_X, max_Y)
    batched_known_arrays = torch.zeros(len(batch), max_X, max_Y)
    
    # copy data into zero tensors
    for i in range(len(batch)):
        X, Y = pixelated_images[i].shape[-2], pixelated_images[i].shape[-1]
        batched_pixelated_images[i, :X, :Y] = torch.from_numpy(pixelated_images[i])
        batched_known_arrays[i, :X, :Y] = torch.from_numpy(known_arrays[i])
    
    return batched_pixelated_images.unsqueeze(1), batched_known_arrays.unsqueeze(1)

def stack_images(batch_as_list: list):
    # Expected list elements are 4-tuples:
    # (pixelated_image, known_array, target_array, image_file)
    pixelated_images = [item[0] for item in batch_as_list]
    known_arrays = [item[1] for item in batch_as_list]
    target_arrays = [item[2] for item in batch_as_list]
    #image_files = [item[3] for item in batch_as_list]
    
    # Directly stack all images, arrays, and targets since they have the same size
    stacked_pixelated_images = torch.stack(pixelated_images, dim=0)
    stacked_known_arrays = torch.stack(known_arrays, dim=0)
    stacked_target_arrays = torch.stack(target_arrays, dim=0)

    return stacked_pixelated_images, stacked_known_arrays, stacked_target_arrays   #, image_files

class SimpleCNN(nn.Module):
    
    def __init__(self, n_in_channels: int = 1, n_hidden_layers: int = 3, n_kernels: int = 32, kernel_size: int = 7):
        """Simple CNN with ``n_hidden_layers``, ``n_kernels`` and
        ``kernel_size`` as hyperparameters."""
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
        """Apply CNN to input ``x`` of shape ``(N, n_channels, X, Y)``, where
        ``N=n_samples`` and ``X``, ``Y`` are spatial dimensions."""
        # Apply hidden layers: (N, n_in_channels, X, Y) -> (N, n_kernels, X, Y)
        cnn_out = self.hidden_layers(x)
        # Apply output layer: (N, n_kernels, X, Y) -> (N, 1, X, Y)
        predictions = self.output_layer(cnn_out)
        return predictions

