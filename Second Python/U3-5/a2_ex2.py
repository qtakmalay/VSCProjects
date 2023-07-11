
import numpy as np
from PIL import Image
from tqdm import tqdm
import glob, os
import matplotlib.pyplot as plt


def round_array(arr: np.ndarray) -> np.ndarray:
    return np.round(np.mean(arr)).astype('int32')
def process_array(image: np.ndarray, cut_height: int, cut_width: int, by: int, ey: int, bx: int, ex: int)-> np.ndarray:
    image[by:ey-cut_height, bx:ex-cut_width] = round_array(image[by:ey-cut_height, bx:ex-cut_width])
    return image
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
        pixelated_image, known_array, target_array = image.copy(), np.ones_like(image, dtype=bool), image[y:y+height, x:x+width]
        for i in range(round(height/size)):
            for j in range(round(width/size)):
                #pixelated_image = process_array(pixelated_image, height%size, width%size, y+size*i, y+size*i+size, x+size*j, x+size*j+size)
                if height%size > 0 and i == round(height/size)-1:
                    pixelated_image[y+size*i:y+size*i+size-height%size, x+size*j:x+size*j+size] = round_array(pixelated_image[y+size*i: y+i*size+size-height%size, x+size*j: x+j*size+size])
                    known_array[y+size*i:y+size*i+size-height%size, x+size*j:x+size*j+size] = False
                elif width%size > 0 and i == round(width/size)-1:
                    pixelated_image[y+size*i:y+size*i+size, x+size*j:x+size*j+size-width%size] = round_array(pixelated_image[y+size*i: y+i*size+size, x+size*j: x+j*size+size-width%size])
                    known_array[y+size*i:y+size*i+size, x+size*j:x+size*j+size-width%size] = False
                else:
                    pixelated_image[y+size*i:y+size*i+size, x+size*j:x+size*j+size] = round_array(pixelated_image[y+size*i: y+i*size+size, x+size*j: x+j*size+size])
                    known_array[y+size*i:y+size*i+size, x+size*j:x+size*j+size] = False
        
        return np.round(pixelated_image).astype(image.dtype), known_array, target_array

