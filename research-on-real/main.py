import cv2
import dlib
import torch
import numpy as np
from models.tcdcn import TCDCN
from core.fatigue_system import FatigueDetector
import os

def main():
    # Load Dlib detectors
    detector = dlib.get_frontal_face_detector()
    predictor_path = "research-on-real/models/shape_predictor_68_face_landmarks.dat"
    
    # Check if predictor exists (it might be bzipped)
    if not os.path.exists(predictor_path):
        if os.path.exists(predictor_path + ".bz2"):
            print("Extracting shape_predictor_68_face_landmarks.dat.bz2...")
            os.system(f"bzip2 -d {predictor_path}.bz2")
        else:
            print("Please download shape_predictor_68_face_landmarks.dat to models/")
            return
            
    predictor = dlib.shape_predictor(predictor_path)
    
    # Load TCDCN (Optional)
    # Note: TCDCN trained on MTFL has 5 landmarks.
    tcdcn = TCDCN(num_landmarks=5)
    tcdcn_weights = "research-on-real/models/weights/tcdcn_epoch_10.pth"
    
    has_tcdcn = False
    if os.path.exists(tcdcn_weights):
        try:
            tcdcn.load_state_dict(torch.load(tcdcn_weights, map_location='cpu'))
            tcdcn.eval()
            has_tcdcn = True
            print("TCDCN loaded.")
        except Exception as e:
            print(f"Failed to load TCDCN: {e}")
    else:
        print("TCDCN weights not found, running without TCDCN attributes.")

    # Fatigue Detector
    fatigue_system = FatigueDetector()
    
    # Open Webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return
    
    print("Starting Driver Fatigue Detection. Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)
        
        for rect in rects:
            # 1. Dlib Landmarks (for Fatigue)
            shape = predictor(gray, rect)
            landmarks = np.array([[p.x, p.y] for p in shape.parts()])
            
            # Update Fatigue System
            status = fatigue_system.update(landmarks)
            
            # Draw Face Box
            x, y, w, h = rect.left(), rect.top(), rect.width(), rect.height()
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Draw Landmarks (Optional, maybe just eyes/mouth)
            for (lx, ly) in landmarks:
                cv2.circle(frame, (lx, ly), 1, (0, 0, 255), -1)
                
            # Display Status
            cv2.putText(frame, f"EAR: {status['ear']:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"MAR: {status['mar']:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"PERCLOS: {status['perclos']:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"M-Index: {status['m_index']:.2f}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Fatigue Level Color
            color = (0, 255, 0)
            if status['fatigue_level'] == "Severe Fatigue":
                color = (0, 0, 255)
            elif status['fatigue_level'] == "Moderate Fatigue":
                color = (0, 165, 255) # Orange
            elif status['fatigue_level'] == "Mild Fatigue":
                color = (0, 255, 255) # Yellow
                
            cv2.putText(frame, f"Level: {status['fatigue_level']}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # 2. TCDCN Attributes (Optional)
            if has_tcdcn:
                # Align face using dlib.get_face_chip
                try:
                    # shape is already available from line 63
                    face_chip = dlib.get_face_chip(gray, shape, size=40, padding=0.25)
                except Exception as e:
                    print(f"Alignment failed: {e}")
                    continue
                
                if face_chip is not None and face_chip.size > 0:
                    # Show aligned face in a new window
                    face_viz = cv2.resize(face_chip, (150, 150))
                    cv2.imshow("Aligned Face (Input to TCDCN)", face_viz)

                    # face_chip is (40, 40) uint8 (grayscale since input was gray)
                    face_img = face_chip.astype(np.float32) / 255.0
                    face_img = torch.from_numpy(face_img).unsqueeze(0).unsqueeze(0)
                   
                    
                    with torch.no_grad():
                        _, gender, glasses, pose, _ = tcdcn(face_img)
                        
                    # Decode attributes
                    gender_label = "Female" if torch.argmax(gender) == 0 else "Male"
                    glasses_label = "Glasses" if torch.argmax(glasses) == 1 else "No Glasses"
                    
                    cv2.putText(frame, f"{gender_label}, {glasses_label}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        cv2.imshow("Driver Fatigue Detection", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
