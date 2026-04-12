from collections import deque
import numpy as np
from utils.metrics import calculate_ear, calculate_mar, calculate_perclos

class FatigueDetector:
    def __init__(self, ear_threshold=0.2, mar_threshold=0.6, perclos_threshold=0.8, 
                 history_length=300, detection_cycle=90):
        self.ear_threshold = ear_threshold
        self.mar_threshold = mar_threshold
        self.perclos_threshold = perclos_threshold
        self.history_length = history_length
        self.detection_cycle = detection_cycle
        
        self.ear_history = deque(maxlen=history_length)
        self.m_history = deque(maxlen=detection_cycle)
        
        # Weights
        self.w_ear = 0.2
        self.w_perclos = 0.8
        self.w_mar = 0.1
        
        self.m_threshold = 0.605
        
    def update(self, landmarks):
        """
        Update the fatigue state with new landmarks.
        
        Args:
            landmarks (np.array): 68 facial landmarks.
            
        Returns:
            dict: Current status including EAR, MAR, PERCLOS, M, and Fatigue Level.
        """
        # Extract points
        # Left Eye: 36-41
        left_eye = landmarks[36:42]
        # Right Eye: 42-47
        right_eye = landmarks[42:48]
        # Mouth: 48-68 (We need specific points for MAR)
        # [48, 50, 52, 54, 56, 58] -> Indices relative to 0-67
        mouth_indices = [48, 50, 52, 54, 56, 58]
        mouth = landmarks[mouth_indices]
        
        # Calculate Metrics
        ear_left = calculate_ear(left_eye)
        ear_right = calculate_ear(right_eye)
        ear = (ear_left + ear_right) / 2.0
        
        mar = calculate_mar(mouth)
        
        # Update History
        self.ear_history.append(ear)
        
        # Calculate PERCLOS
        perclos = calculate_perclos(self.ear_history, self.ear_threshold)
        
        # Calculate M-index
        # Interpretation: V_EAR is 1 if closed, 0 if open
        v_ear = 1.0 if ear < self.ear_threshold else 0.0
        
        # Interpretation: V_MAR is 1 if yawning, 0 if not (or raw MAR?)
        # Paper says "Mar value is the auxiliary value". 
        # If we use raw MAR (e.g. 0.7), it adds 0.07.
        # If we use binary (1.0), it adds 0.1.
        # Given the low weight, binary makes sense to give it a fixed "boost".
        v_mar = 1.0 if mar > self.mar_threshold else 0.0
        
        m_index = (self.w_ear * v_ear) + (self.w_perclos * perclos) + (self.w_mar * v_mar)
        
        self.m_history.append(m_index)
        
        # Determine Fatigue Level (N)
        # N = number of frames where M > 0.605 in the last 90 frames
        n_count = sum(1 for m in self.m_history if m > self.m_threshold)
        
        fatigue_level = "Normal"
        if n_count >= 50:
            fatigue_level = "Severe Fatigue"
        elif n_count > 20:
            fatigue_level = "Moderate Fatigue"
        elif n_count > 10:
            fatigue_level = "Mild Fatigue"
            
        return {
            "ear": ear,
            "mar": mar,
            "perclos": perclos,
            "m_index": m_index,
            "n_count": n_count,
            "fatigue_level": fatigue_level,
            "is_yawning": mar > self.mar_threshold,
            "is_eyes_closed": ear < self.ear_threshold
        }
