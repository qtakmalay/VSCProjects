
import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse, glob, os, math
import matplotlib.pyplot as plt

def prepare_image(image: np.ndarray, x: int, y: int, width: int, height: int, size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if image.shape[2] != 1:
        raise ValueError(f"The shape of the image has to be one (H, W, 1), but got {image.shape}.")
    elif (width and height and size) < 2:
        raise ValueError(f"Values of width, height, and size must be larger than 1, but got {width}, {height}, {size}.")
    elif x < 0 or x + width > image.shape[0]:
        raise ValueError(f"x is smaller than 0 or x + width is larger than the width of the input image")
    elif y < 0 or y + height > image.shape[1]:
        raise ValueError(f"y is smaller than 0 or y + height is larger than the height of the input image")
    else:
        print("Hello world!")
    pass
def to_grayscale(pil_image: np.ndarray) -> np.ndarray:
    if(len(pil_image.shape)<3):
      result = pil_image[np.newaxis, :, :].copy()
    elif len(pil_image.shape)==3:
        if pil_image.shape[2] != 3:
            raise ValueError
        pil_image_n = pil_image/255
        
        pil_image_cl = np.where(pil_image_n <= 0.04045, pil_image_n / 12.92, np.power((pil_image_n + 0.055) / 1.055, 2.4))
        pil_image_yl = pil_image_cl[:,:,0] * 0.2126 + pil_image_cl[:,:,1] * 0.7152 + pil_image_cl[:,:,2] * 0.0722 
        pil_image_y = np.where(pil_image_yl <= 0.0031308, pil_image_yl * 12.92, (1.055 * np.power(pil_image_yl, 1/2.4)) - 0.055)
        pil_image_y = pil_image_y[:, :, np.newaxis]
        result = (pil_image_y * 255).clip(0, 255)
            
    else:
        raise ValueError

    if np.issubdtype(pil_image.dtype, np.integer):
        result = np.round(result).astype(pil_image.dtype)

    return result

input_path = "C:\\Users\\azatv\\VSCProjects\\Second Python\\U3-5\\04_images"
image_files = sorted(glob.glob(os.path.join(input_path, "**", "*.jpg"), recursive=True))
print("Filename: ",os.path.basename(image_files[0]))
with Image.open(image_files[0]) as im:  # This returns a PIL image "Second Python\\U3-5\\04_images\\000\\train_101.jpg"
    image = np.array(im)  # We can convert it to a numpy array
print(f"mode: {im.mode}; shape: {image.shape}; min: {image.min()}; max: {image.max()}; dtype: {image.dtype}")


grayscaled_img = to_grayscale(image)
print(grayscaled_img.shape[0])
prepared_img = prepare_image(grayscaled_img, 10, 10, 30, 30, 3)
plt.imshow(grayscaled_img)
plt.show()