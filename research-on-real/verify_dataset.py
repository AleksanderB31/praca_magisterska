import cv2
import torch
import numpy as np
from utils.data_loader import MTFLDatasetRefined
import os

def verify_dataset():
    DATASET_ROOT = "/home/aleksander/workspace/praca_magisterska/datasets/MTFL"
    TRAIN_TXT = os.path.join(DATASET_ROOT, "training.txt")
    
    if not os.path.exists(TRAIN_TXT):
        print("Dataset not found.")
        return

    dataset = MTFLDatasetRefined(root_dir=DATASET_ROOT, txt_file=TRAIN_TXT)
    
    os.makedirs("research-on-real/debug_data", exist_ok=True)
    
    for i in range(5):
        sample = dataset[i]
        image = sample['image'].numpy()[0] * 255
        image = image.astype(np.uint8)
        landmarks = sample['landmarks'].numpy()
        
        # Draw landmarks
        vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        for j in range(0, 10, 2):
            x = int(landmarks[j])
            y = int(landmarks[j+1])
            cv2.circle(vis, (x, y), 1, (0, 0, 255), -1)
            
        cv2.imwrite(f"research-on-real/debug_data/sample_{i}.png", vis)
        print(f"Saved sample_{i}.png. Landmarks: {landmarks}")

if __name__ == "__main__":
    verify_dataset()
