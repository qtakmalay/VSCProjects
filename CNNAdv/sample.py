import cv2
import torch


arr = torch.ones(4,4)

arr[1,:2] = 0
print(f"torch arr: {arr}")
con_arr = torch.cat([arr,arr], dim=1)
print(f"cat arr: {con_arr}")
'''
def extract_imgs_and_labels(path:str) -> Tensor, Tensor{

}
'''