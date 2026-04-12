import os
from utils.data_loader import MTFLDatasetRefined
from collections import Counter

def debug_dataset():
    DATASET_ROOT = "/home/aleksander/workspace/praca_magisterska/datasets/MTFL"
    TRAIN_TXT = os.path.join(DATASET_ROOT, "training.txt")
    
    dataset = MTFLDatasetRefined(root_dir=DATASET_ROOT, txt_file=TRAIN_TXT)
    
    genders = []
    smiles = []
    glasses = []
    poses = []
    
    print(f"Checking {len(dataset)} samples...")
    
    for i in range(len(dataset)):
        sample = dataset[i]
        genders.append(sample['gender'].item())
        smiles.append(sample['smile'].item())
        glasses.append(sample['glasses'].item())
        poses.append(sample['pose'].item())
        
        if i % 1000 == 0:
            print(f"Processed {i} samples")
            
    print("\n--- Statistics ---")
    print(f"Gender values: {Counter(genders)}")
    print(f"Smile values: {Counter(smiles)}")
    print(f"Glasses values: {Counter(glasses)}")
    print(f"Pose values: {Counter(poses)}")

if __name__ == "__main__":
    debug_dataset()
