
import torch
from typing import Sequence, Union, Tuple
import os, glob
from PIL import Image
import numpy as np
import torchvision
from a6_ex1 import random_augmented_image
class ImageDataset(torch.utils.data.Dataset):
   def __init__(self, image_dir):
      self.list_files = sorted(glob.glob(os.path.join(image_dir, "**", "*.jpg"), recursive=True))
   def __getitem__(self, index: int):
      return (Image.open(self.list_files[index]), index)
   def __len__(self):
      return len(self.images)
   
class TransformedImageDataset(torch.utils.data.Dataset):
   def __init__(self, dataset: ImageDataset, image_size: Union[int, Tuple[int]]):
      self.dataset = dataset
      self.image_size = image_size
   def __getitem__(self, index: int):
      image_tensor, _ = self.dataset[index]
      tensor_image = random_augmented_image(image_tensor, 
                                           self.image_size, index)
      return (tensor_image, index)
         
   def __len__(self):
      return len(self.dataset)
   
if __name__ == "__main__":
    from matplotlib import pyplot as plt
    imgs = ImageDataset(image_dir="C:\\Users\\azatv\\VSCode\\VSCProjects\\Second Python\\Assign6")
    transformed_imgs = TransformedImageDataset(imgs, image_size=300)
    for (original_img, index), (transformed_img, _) in zip(imgs, transformed_imgs):
        fig, axes = plt.subplots(1, 2)
        axes[0].imshow(original_img)
        axes[0].set_title("Original image")
        axes[1].imshow(torchvision.transforms.functional.to_pil_image(transformed_img))
        axes[1].set_title("Transformed image")
        fig.suptitle(f"Image {index}")
        fig.tight_layout()
        plt.show()