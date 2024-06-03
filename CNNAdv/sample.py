import cv2, torch, numpy as np, os





def extract_imgs_and_labels(path):
    if(len(os.listdir()) == 0): return None
    labels_map = dict()
    for folder in os.listdir(path):
        if not labels_map.__contains__(folder):
            labels_map.update({str(folder):None})
        folder_path = os.path.join(path, folder)
        for file in os.listdir(folder_path):
            labels_map[folder].append(str(file))
    
    print(labels_map)

    

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.chdir()


extract_imgs_and_labels(os.getcwd())
        



