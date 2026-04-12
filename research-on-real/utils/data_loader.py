import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

class MTFLDataset(Dataset):
    def __init__(self, root_dir, txt_file, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            txt_file (string): Path to the text file with annotations.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        
        with open(txt_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                # Format: path x1 y1 x2 y2 x3 y3 x4 y4 x5 y5 gender smile glasses pose
                # Note: MTFL format might vary, checking standard.
                # Usually: path le_x le_y re_x re_y n_x n_y lm_x lm_y rm_x rm_y gender smile glasses pose
                if len(parts) < 15:
                    continue
                self.samples.append(parts)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        parts = self.samples[idx]
        img_path = os.path.join(self.root_dir, parts[0].replace('\\', '/'))
        
        image = cv2.imread(img_path)
        if image is None:
            # Return a dummy if image not found to avoid crashing
            image = np.zeros((40, 40, 1), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
        # Resize to 40x40 as per TCDCN paper
        image = cv2.resize(image, (40, 40))
        
        # Landmarks (5 points)
        landmarks = list(map(float, parts[1:11]))
        
        # Attributes
        gender = int(parts[11]) - 1
        smile = int(parts[12]) - 1
        glasses = int(parts[13]) - 1
        pose = int(parts[14]) - 1
        
        # Normalize image
        image = image.astype(np.float32) / 255.0
        image = np.expand_dims(image, axis=0) # 1 x 40 x 40
        
        # Normalize landmarks (relative to 40x40?)
        # The original landmarks are in original image coordinates.
        # We need to scale them to 40x40.
        # But we don't know original size unless we read it.
        # Wait, we resized the image. We must resize landmarks too.
        # To do this correctly, we need original dimensions.
        # I should read original image, get size, then resize.
        
        # Re-read to get size (optimized: do this before resize)
        # Actually, I already read it.
        # But wait, `cv2.imread` returns the image.
        # I need to get h, w from it.
        
        # Refined logic in __getitem__
        
        sample = {
            'image': torch.from_numpy(image),
            'landmarks': torch.tensor(landmarks, dtype=torch.float32),
            'gender': torch.tensor(gender, dtype=torch.long),
            'smile': torch.tensor(smile, dtype=torch.long),
            'glasses': torch.tensor(glasses, dtype=torch.long),
            'pose': torch.tensor(pose, dtype=torch.long)
        }
        
        return sample

# Refined implementation with landmark scaling
class MTFLDatasetRefined(Dataset):
    def __init__(self, root_dir, txt_file, transform=None, augment=False):
        self.root_dir = root_dir
        self.transform = transform
        self.augment = augment
        self.samples = []
        
        with open(txt_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 15:
                    continue
                self.samples.append(parts)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        parts = self.samples[idx]
        img_path = os.path.join(self.root_dir, parts[0].replace('\\', '/'))
        
        image = cv2.imread(img_path)
        
        if image is None:
            return self.__getitem__((idx + 1) % len(self))
            
        h, w = image.shape[:2]
        
        # Landmarks (5 points * 2)
        landmarks = np.array(list(map(float, parts[1:11]))).reshape(-1, 2)
        
        # Augmentation
        if self.augment:
            # 1. Random Rotation
            angle = np.random.uniform(-15, 15)
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
            image = cv2.warpAffine(image, M, (w, h))
            
            # Rotate landmarks
            ones = np.ones(shape=(len(landmarks), 1))
            points_ones = np.concatenate([landmarks, ones], axis=1)
            landmarks = M.dot(points_ones.T).T
            
            # 2. Random Noise
            if np.random.random() > 0.5:
                noise = np.random.normal(0, 5, image.shape).astype(np.uint8)
                image = cv2.add(image, noise)

        # Scale landmarks to [0, 1]
        scaled_landmarks = []
        for pt in landmarks:
            scaled_landmarks.append(pt[0] / w)
            scaled_landmarks.append(pt[1] / h)
            
        # Resize image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (40, 40))
        
        image = image.astype(np.float32) / 255.0
        image = np.expand_dims(image, axis=0)
        
        gender = int(parts[11]) - 1
        smile = int(parts[12]) - 1
        glasses = int(parts[13]) - 1
        pose = int(parts[14]) - 1
        
        return {
            'image': torch.from_numpy(image),
            'landmarks': torch.tensor(scaled_landmarks, dtype=torch.float32),
            'gender': torch.tensor(gender, dtype=torch.long),
            'smile': torch.tensor(smile, dtype=torch.long),
            'glasses': torch.tensor(glasses, dtype=torch.long),
            'pose': torch.tensor(pose, dtype=torch.long)
        }
