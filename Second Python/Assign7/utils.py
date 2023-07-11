from glob import glob
from os import path
from typing import Optional

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

import torchvision.transforms as transforms
import torchvision.transforms.functional as functional

def prepare_image(image: np.ndarray, x: int, y: int, width: int, height: int, size: int) -> \
        tuple[np.ndarray, np.ndarray, np.ndarray]:
    if image.ndim < 2:
        # This is actually more general than the assignment specification
        raise ValueError("image must have shape (H, W)")
    if width < 2 or height < 2 or size < 2:
        raise ValueError("width/height/size must be >= 2")
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

class TestDataset(Dataset):
    def __init__(self, pixelated_images, known_arrays):
        self.pixelated_images = pixelated_images
        self.known_arrays = known_arrays

    def __len__(self):
        return len(self.pixelated_images)

    def __getitem__(self, idx):
        return self.pixelated_images[idx], self.known_arrays[idx]

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
            im = self.transform_chain(im)
            image = np.array(im, dtype=self.dtype)

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
        return pixelated_image, known_array, target_array, self.image_files[index]
    
    def __len__(self):
        return len(self.image_files)
    

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
