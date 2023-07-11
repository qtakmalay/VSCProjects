import pickle as pkl
from matplotlib import pyplot as plt
import torchvision.transforms as transforms
import torchvision.transforms.functional as functional
from PIL import Image
import numpy as np
import os, glob, torch
import torch.nn as nn
from torch.utils.data import DataLoader
import utils
from torch.utils.data import random_split

rng = np.random.default_rng()
width = rng.integers(0, 32, size=1)
height = rng.integers(0, 32, size=1)

batch_size = 64

os.environ['KMP_DUPLICATE_LIB_OK']='True'

train_image_path = r'C:\Users\azatv\VSCode\VSCProjects\Second Python\Assign7\training'




with open(r'C:\Users\azatv\VSCode\VSCProjects\Second Python\Assign7\test_set_ones.pkl', 'rb') as f:
    data = pkl.load(f)
pix_img_np = np.array([i for i in data["pixelated_images"]])
known_img_np = np.array([i for i in data["known_arrays"]])



train_dataset = utils.RandomImagePixelationDataset(train_image_path, (0, int(64-width)), (0, int(64-width)), (4, 16), dtype=np.uint8)
train_size = int(0.8 * len(train_dataset))
valid_size = len(train_dataset) - train_size
train_dataset, valid_dataset = random_split(train_dataset, [train_size, valid_size])

#test loader
test_dataset = utils.TestDataset(pix_img_np, known_img_np)
test_loader = DataLoader(test_dataset, 
                         batch_size=batch_size, 
                         shuffle=False, 
                         num_workers=4)

#train loader
train_loader = DataLoader(train_dataset, 
                          batch_size=batch_size, 
                          shuffle=True)

#validation loader
valid_loader = DataLoader(valid_dataset, 
                          batch_size=batch_size, 
                          shuffle=False)


# pixelated_image, known_array, target_array, image_file = train_dataset[0]
# pixelated_image = Image.fromarray(np.squeeze(pixelated_image).astype(np.uint8))
# #known_array = np.where(known_img_np[0], 255, 0).astype(np.uint8)
# target_array = Image.fromarray(np.squeeze(pixelated_image).astype(np.uint8))
# known_array = Image.fromarray(np.squeeze(known_array).astype(np.uint8))

# fig, axes = plt.subplots(6, 3, figsize=(15, 30))  # Increase the figure size for visibility

# for i in range(6):  # Adjust this to the number of images you want
#     pixelated_image, known_array, target_array, _ = train_dataset[i]
#     axes[i, 0].imshow(pixelated_image, cmap='gray')
#     axes[i, 0].set_title(f'Pixelated Image {i+1}')
#     axes[i, 1].imshow(known_array, cmap='gray')
#     axes[i, 1].set_title(f'Known Array {i+1}')
#     axes[i, 2].imshow(target_array, cmap='gray')
#     axes[i, 2].set_title(f'Target Array {i+1}')

# # Remove the axis labels for clean plots
# for ax in axes.ravel():
#     ax.axis('off')

# fig.tight_layout()
# plt.show()

