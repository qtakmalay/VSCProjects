
import numpy as np
import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
import torchvision.transforms as transforms
import torchvision.transforms.functional as functional
from typing import Sequence, Union
import os
import random

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def random_augmented_image(
    image: Image,
    image_size: Union[int, Sequence[int]],
    seed: int
    ) -> torch.Tensor:
    torch.random.manual_seed(seed)
    #transfr_inx = np.random.integers(size=2, low=0, high=4, dtype=np.uint8)
    choosen_transforms =[torchvision.transforms.RandomRotation(60),
        torchvision.transforms.RandomVerticalFlip(),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ColorJitter((0.8, 1.2))]
    transform_chain = transforms.Compose([
        transforms.Resize(size=image_size),  # Resize image to minimum edge = 100 pixels
        choosen_transforms[random.randint(0, 3)],
        choosen_transforms[random.randint(0, 3)],
        transforms.ToTensor(),
        torch.nn.Dropout(0.1)
    ])
    return transform_chain(image)


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    with Image.open("C:\\Users\\azatv\\VSCode\\VSCProjects\\Second Python\\Assign6\\08_example_image.jpg") as image:
        transformed_image = random_augmented_image(image, image_size=300, seed=3)
        fig, axes = plt.subplots(1, 2)
        axes[0].imshow(image)
        axes[0].set_title("Original image")
        axes[1].imshow(functional.to_pil_image(transformed_image))
        axes[1].set_title("Transformed image")
        fig.tight_layout()
        plt.show() 