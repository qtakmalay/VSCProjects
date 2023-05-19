# -*- coding: utf-8 -*-
"""
Author -- Michael Widrich, Andreas Schörgenhumer

################################################################################

The following copyright statement applies to all code within this file.

Copyright statement:
This material, no matter whether in printed or electronic form, may be used for
personal and non-commercial educational use only. Any reproduction of this
manuscript, no matter whether as a whole or in parts, no matter whether in
printed or in electronic form, requires explicit prior acceptance of the
authors.

################################################################################

Datasets file of example project.
"""

import numpy as np
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset


def rgb2gray(rgb_array: np.ndarray, r: float = 0.2989, g: float = 0.5870, b: float = 0.1140):
    """Convert numpy array with 3 color channels of shape (..., 3) to
    grayscale."""
    grayscale_array = (rgb_array[..., 0] * r +
                       rgb_array[..., 1] * g +
                       rgb_array[..., 2] * b)
    grayscale_array = np.round(grayscale_array)
    grayscale_array = np.asarray(grayscale_array, dtype=np.uint8)
    return grayscale_array


class CIFAR10(Dataset):
    
    def __init__(self, data_folder: str = "cifar10"):
        """Dataset providing CIFAR10 grayscale images as inputs."""
        # Load or download CIFAR10 data set
        cifar10 = torchvision.datasets.CIFAR10(data_folder, train=True, download=True)
        # Get images and convert them to grayscale
        self.data = rgb2gray(cifar10.data)
    
    def __getitem__(self, idx):
        image_data = self.data[idx]
        return image_data, idx
    
    def __len__(self):
        return len(self.data)


class RotatedImages(Dataset):
    
    def __init__(self, dataset: Dataset, rotation_angle: float = 45, transform_chain: transforms.Compose = None):
        """Provides images from ``dataset`` as inputs and images rotated by
        ``rotation_angle`` as targets."""
        self.dataset = dataset
        self.rotation_angle = rotation_angle
        self.transform_chain = transform_chain
    
    def __getitem__(self, idx):
        image_data, idx = self.dataset[idx]
        image_data = TF.to_pil_image(image_data)
        if self.transform_chain is not None:
            image_data = self.transform_chain(image_data)
        
        # Create rotated target (this will introduce unknown parts of the image)
        rotated_image_data = TF.rotate(image_data, angle=self.rotation_angle,
                                       interpolation=TF.InterpolationMode.BILINEAR)
        # Crop and resize to get rid of unknown image parts
        image_data = TF.resized_crop(image_data, 8, 8, 16, 16, size=32)
        rotated_image_data = TF.resized_crop(rotated_image_data, 8, 8, 16, 16, size=32)
        
        # Convert to float32
        image_data = np.asarray(image_data, dtype=np.float32)
        rotated_image_data = np.asarray(rotated_image_data, dtype=np.float32)
        # Perform normalization based on input values of individual sample
        mean = image_data.mean()
        std = image_data.std()
        image_data[:] -= mean
        image_data[:] /= std
        rotated_image_data[:] -= mean
        rotated_image_data[:] /= std
        
        # Add information about relative position in image to inputs. When
        # merely setting "full_inputs = image_data", we would not be feeding
        # information about the position in the image, and this would be bad for
        # our CNN
        full_inputs = np.zeros(shape=(*image_data.shape, 3), dtype=image_data.dtype)
        full_inputs[..., 0] = image_data
        full_inputs[..., 1] = np.linspace(start=-1, stop=1, num=full_inputs.shape[1])
        full_inputs[..., 2] = np.transpose(full_inputs[:, :, 1])
        
        # Convert numpy arrays to tensors
        full_inputs = TF.to_tensor(full_inputs)
        rotated_image_data = TF.to_tensor(rotated_image_data)
        
        return full_inputs, rotated_image_data, idx
    
    def __len__(self):
        return len(self.dataset)
