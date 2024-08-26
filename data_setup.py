
import os
import cv2
import numpy as np

def load_images(img_dir, H=256, W=256):
    file_paths = []
    
    for root, dirs, files in os.walk(img_dir):
        for file in files:
            file_paths.append(os.path.join(root, file))
                              
    images = []
    for path in file_paths:
        try:
            img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (H, W), interpolation=cv2.INTER_AREA)
            images.append(img)
        except:
            pass

    return np.array(images)
