import cv2
import dlib
import numpy as np
import os
import torch
import matplotlib.pyplot as plt
from core.fatigue_system import FatigueDetector
from models.tcdcn import TCDCN

def debug_video(video_path, output_path):
    # Paths
    PREDICTOR_PATH = "research-on-real/models/shape_predictor_68_face_landmarks.dat"
    TCDCN_WEIGHTS = "research-on-real/models/weights/tcdcn_best.pth" # Using best model
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not os.path.exists(PREDICTOR_PATH):
        print(f"Error: Predictor not found at {PREDICTOR_PATH}")
        return

    # Setup Dlib
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(PREDICTOR_PATH)
    
    # Setup TCDCN
    tcdcn = TCDCN(num_landmarks=5).to(device)
    has_tcdcn = False
    if os.path.exists(TCDCN_WEIGHTS):
        try:
            tcdcn.load_state_dict(torch.load(TCDCN_WEIGHTS, map_location=device))
            tcdcn.eval()
            has_tcdcn = True
            print("TCDCN loaded.")
        except Exception as e:
            print(f"Failed to load TCDCN: {e}")

    # Initialize Fatigue System
    fatigue_system = FatigueDetector()
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not open {video_path}")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"Processing video: {video_path}")
    
    history = {
        'ear': [],
        'mar': [],
        'perclos': [],
        'm_index': [],
        'n_count': []
    }

    DETECTION_INTERVAL = 10
    last_rect = None
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        rect = None
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
            # Landmarks
            shape = predictor(gray, rect)
            landmarks = np.array([[p.x, p.y] for p in shape.parts()])
            
            # Update Fatigue System
            status = fatigue_system.update(landmarks)
            
            # Collect History
            history['ear'].append(status['ear'])
            history['mar'].append(status['mar'])
            history['perclos'].append(status['perclos'])
            history['m_index'].append(status['m_index'])
            history['n_count'].append(status['n_count'])
            
            # Draw Face Box
            x, y, w, h = rect.left(), rect.top(), rect.width(), rect.height()
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Draw Landmarks
            for (lx, ly) in landmarks:
                cv2.circle(frame, (lx, ly), 1, (0, 0, 255), -1)
                
            # Display Status Overlays
            y_offset = 30
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2
            
            info = [
                f"EAR: {status['ear']:.2f}",
                f"MAR: {status['mar']:.2f}",
                f"PERCLOS: {status['perclos']:.2f}",
                f"M-Index: {status['m_index']:.2f}",
                f"N-Count: {status['n_count']}",
                f"Level: {status['fatigue_level']}"
            ]
            
            for line in info:
                color = (0, 255, 0)
                if "Severe" in line: color = (0, 0, 255)
                elif "Moderate" in line: color = (0, 165, 255)
                elif "Mild" in line: color = (0, 255, 255)
                
                cv2.putText(frame, line, (10, y_offset), font, font_scale, color, thickness)
                y_offset += 30
        else:
            # If no face, append last known or default
            for key in history:
                if len(history[key]) > 0:
                    history[key].append(history[key][-1])
                else:
                    history[key].append(0.0)

        out.write(frame)
        frame_idx += 1
        
        if frame_idx % 100 == 0:
            print(f"Processed {frame_idx} frames...")

    cap.release()
    out.release()
    
    # Generate Plots (Figure 11 style)
    print("Generating plots...")
    fig, axes = plt.subplots(4, 1, figsize=(12, 16))
    
    frames = np.arange(len(history['ear']))
    
    # 1. EAR
    axes[0].plot(frames, history['ear'], label='EAR', color='blue')
    axes[0].axhline(y=0.2, color='red', linestyle='--', label='Threshold (0.2)')
    axes[0].set_title('Eye Aspect Ratio (EAR)')
    axes[0].set_ylabel('Value')
    axes[0].legend()
    axes[0].grid(True)
    
    # 2. MAR
    axes[1].plot(frames, history['mar'], label='MAR', color='green')
    axes[1].axhline(y=0.6, color='red', linestyle='--', label='Threshold (0.6)')
    axes[1].set_title('Mouth Aspect Ratio (MAR)')
    axes[1].set_ylabel('Value')
    axes[1].legend()
    axes[1].grid(True)
    
    # 3. PERCLOS
    axes[2].plot(frames, history['perclos'], label='PERCLOS', color='orange')
    axes[2].set_title('PERCLOS (History Window)')
    axes[2].set_ylabel('Value')
    axes[2].grid(True)
    
    # 4. M-Index and N-Count
    ax4_2 = axes[3].twinx()
    axes[3].plot(frames, history['m_index'], label='M-Index', color='purple')
    axes[3].axhline(y=0.605, color='red', linestyle='--', label='M-Threshold')
    ax4_2.plot(frames, history['n_count'], label='N-Count', color='black', alpha=0.3)
    
    axes[3].set_title('Comprehensive Index M and N-Count')
    axes[3].set_xlabel('Frame Index')
    axes[3].set_ylabel('M-Index')
    ax4_2.set_ylabel('N-Count (Frames > M-Thresh)')
    axes[3].legend(loc='upper left')
    ax4_2.legend(loc='upper right')
    axes[3].grid(True)
    
    plt.tight_layout()
    plot_path = output_path.replace('.mp4', '_metrics.png')
    plt.savefig(plot_path)
    print(f"Plots saved to: {plot_path}")
    print("Finished.")

if __name__ == "__main__":
    VIDEO_PATH = "datasets/UTA/Fold1_part2/Fold1_part2/10/10.MOV"
    OUTPUT_PATH = "research-on-real/debug_severe_10.mp4"
    debug_video(VIDEO_PATH, OUTPUT_PATH)
