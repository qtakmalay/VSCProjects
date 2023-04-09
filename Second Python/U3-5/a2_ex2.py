
import numpy as np
from PIL import Image
from tqdm import tqdm
import glob, os
import matplotlib.pyplot as plt

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


def plot_images(original_image, pixelated_image, known_array, target_array):
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].imshow(original_image.squeeze(), cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    axes[1].imshow(pixelated_image.squeeze(), cmap='gray')
    axes[1].set_title('Pixelated Image')
    axes[1].axis('off')

    axes[2].imshow(known_array.squeeze(), cmap='gray')
    axes[2].set_title('Known Array')
    axes[2].axis('off')

    axes[3].imshow(target_array.squeeze(), cmap='gray')
    axes[3].set_title('Target Array')
    axes[3].axis('off')

    plt.show()

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

#     pixelated_image = image.copy()
#     known_array = np.ones_like(image, dtype=bool)
#     target_array = image.copy()
    
#     for i in range(y, y + height, size):
#         for j in range(x, x + width, size):
#             block_height = min(size, y + height - i)
#             block_width = min(size, x + width - j)
            
#             block = image[i:i + block_height, j:j + block_width, 0]
#             avg_value = np.mean(block)
            
#             pixelated_image[i:i + block_height, j:j + block_width, 0] = avg_value
#             known_array[i:i + block_height, j:j + block_width, 0] = False
            
#     return pixelated_image, known_array, target_array
    
input_path = "C:\\Users\\azatv\\VSCProjects\\Second Python\\U3-5\\04_images"
image_files = sorted(glob.glob(os.path.join(input_path, "**", "*.jpg"), recursive=True))
print("Filename: ",os.path.basename(image_files[0]))
with Image.open(image_files[0]) as im:  # This returns a PIL image "Second Python\\U3-5\\04_images\\000\\train_101.jpg"
    image = np.array(im)  # We can convert it to a numpy array

grayscaled_img = to_grayscale(image)
prepared_img = prepare_image(grayscaled_img, 0, 0, 64, 64, 10)
plot_images(grayscaled_img, prepared_img[0], prepared_img[1], prepared_img[2])