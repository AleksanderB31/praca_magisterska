import cv2
import dlib
import numpy as np
import os
import glob
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from core.fatigue_system import FatigueDetector
from models.tcdcn import TCDCN
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import functools
import torch.multiprocessing as mp

def get_ground_truth(filename):
    basename = os.path.basename(filename)
    name_without_ext = os.path.splitext(basename)[0]
    try:
        label = int(name_without_ext)
        if label in [0, 5, 10]:
            return label
    except ValueError:
        pass
    return None

def map_fatigue_level_to_class(level):
    if level == "Normal":
        return 0
    elif level == "Mild Fatigue":
        return 5
    elif level == "Moderate Fatigue":
        return 5
    elif level == "Severe Fatigue":
        return 10
    return 0 

def process_single_video(video_path, predictor_path, tcdcn_weights, device_str):
    gt = get_ground_truth(video_path)
    if gt is None:
        return None
        
    device = torch.device(device_str)
    
    # Initialize components inside the process
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    
    # Fatigue system
    fatigue_system = FatigueDetector()
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
        
    frame_predictions = []
    
    # Tracking Parameters
    DETECTION_INTERVAL = 30 # Detect face every 30 frames (1 second at 30fps)
    last_rect = None
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize for faster processing (standardizing to 480p height)
        h, w = frame.shape[:2]
        scale = 480.0 / h
        small_frame = cv2.resize(frame, (int(w * scale), 480))
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        
        rect = None
        # Detection logic
        if frame_idx % DETECTION_INTERVAL == 0 or last_rect is None:
            rects = detector(gray, 0)
            if len(rects) > 0:
                rect = max(rects, key=lambda r: r.width() * r.height())
                last_rect = rect
            else:
                last_rect = None
        else:
            rect = last_rect
            
        if rect is not None:
            try:
                # Landmark prediction is fast if we have the rect
                shape = predictor(gray, rect)
                landmarks = np.array([[p.x, p.y] for p in shape.parts()])
                
                # Update fatigue system (processes every frame now)
                status = fatigue_system.update(landmarks)
                pred_class = map_fatigue_level_to_class(status['fatigue_level'])
                frame_predictions.append(pred_class)
            except:
                last_rect = None # Reset on error
        
        frame_idx += 1
            
    cap.release()
    
    if not frame_predictions:
        return (gt, 0)
    
    # Global Majority Vote for the entire video
    counts = {0: 0, 5: 0, 10: 0}
    for p in frame_predictions:
        counts[p] += 1
    
    # Final prediction is the state that occurred most often
    final_pred = max(counts, key=counts.get)
    
    return (gt, final_pred)

def test_uta():
    DATASET_DIR = "datasets/UTA" 
    PREDICTOR_PATH = "research-on-real/models/shape_predictor_68_face_landmarks.dat"
    # We will use the latest weights after training finishes
    # For now, let's assume we use the best one available or wait.
    # I'll use a placeholder that we can update.
    TCDCN_WEIGHTS = "models/weights/tcdcn_epoch_50.pth"
    
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Main process device: {device_str}")

    if not os.path.exists(PREDICTOR_PATH):
        print(f"Error: Predictor not found at {PREDICTOR_PATH}")
        return

    video_files = []
    for ext in ('**/*.mov', '**/*.MOV', '**/*.mp4'):
        video_files.extend(glob.glob(os.path.join(DATASET_DIR, ext), recursive=True))
        
    print(f"Found {len(video_files)} videos. Starting redesigned evaluation (No frame skip, full voting)...")
    
    # Use fewer workers because we are processing EVERY frame now (memory/CPU intensive)
    # 8 workers should be safe for 16 cores
    with ProcessPoolExecutor(max_workers=8) as executor:
        process_func = functools.partial(
            process_single_video, 
            predictor_path=PREDICTOR_PATH, 
            tcdcn_weights=TCDCN_WEIGHTS, 
            device_str=device_str
        )
        
        results = list(tqdm(executor.map(process_func, video_files), total=len(video_files), desc="Evaluating Videos"))

    results = [r for r in results if r is not None]
    y_true = [r[0] for r in results]
    y_pred = [r[1] for r in results]
    
    print("\n--- Redesigned Evaluation Results ---")
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['Normal (0)', 'Mild/Mod (5)', 'Severe (10)'], labels=[0, 5, 10]))
    
    cm = confusion_matrix(y_true, y_pred, labels=[0, 5, 10])
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
                xticklabels=['Normal (0)', 'Mild/Mod (5)', 'Severe (10)'],
                yticklabels=['Normal (0)', 'Mild/Mod (5)', 'Severe (10)'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix - UTA Dataset (Full Video Analysis)')
    plt.savefig('research-on-real/confusion_matrix_redesigned.png')
    print("Confusion matrix saved to research-on-real/confusion_matrix_redesigned.png")

if __name__ == "__main__":
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    test_uta()
