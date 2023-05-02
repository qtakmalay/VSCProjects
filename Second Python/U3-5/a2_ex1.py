import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse, glob, os, math
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

        result = (pil_image_y[:, :, np.newaxis] * 255).clip(0, 255)
            
    else:
        raise ValueError

    if np.issubdtype(pil_image.dtype, np.integer):
        result = np.round(result).astype(pil_image.dtype)

    return result


