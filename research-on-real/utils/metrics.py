import numpy as np
from scipy.spatial import distance as dist

def calculate_ear(eye_points):
    """
    Calculate Eye Aspect Ratio (EAR).
    
    Args:
        eye_points (list or np.array): 6 points (x, y) for a single eye.
                                       Order: P1(left), P2(top-left), P3(top-right), 
                                              P4(right), P5(bottom-right), P6(bottom-left).
    Returns:
        float: EAR value.
    """
    # Vertical distances
    A = dist.euclidean(eye_points[1], eye_points[5]) # P2 - P6
    B = dist.euclidean(eye_points[2], eye_points[4]) # P3 - P5
    
    # Horizontal distance
    C = dist.euclidean(eye_points[0], eye_points[3]) # P1 - P4
    
    if C == 0:
        return 0.0
        
    ear = (A + B) / (2.0 * C)
    return ear

def calculate_mar(mouth_points):
    """
    Calculate Mouth Aspect Ratio (MAR).
    
    Args:
        mouth_points (list or np.array): Points for the mouth.
                                         Expected indices from 68-point model:
                                         [48, 50, 52, 54, 56, 58] (0-indexed relative to full face)
                                         mapped to 0-5 here.
                                         0: Left Corner (48)
                                         1: Top Left (50)
                                         2: Top Right (52)
                                         3: Right Corner (54)
                                         4: Bottom Right (56)
                                         5: Bottom Left (58)
    Returns:
        float: MAR value.
    """
    # Vertical distances
    A = dist.euclidean(mouth_points[1], mouth_points[5]) # 50 - 58
    B = dist.euclidean(mouth_points[2], mouth_points[4]) # 52 - 56
    
    # Horizontal distance
    C = dist.euclidean(mouth_points[0], mouth_points[3]) # 48 - 54
    
    if C == 0:
        return 0.0
        
    mar = (A + B) / (2.0 * C)
    return mar

def calculate_perclos(ear_history, threshold=0.2):
    """
    Calculate PERCLOS (Percentage of Eye Closure).
    
    Args:
        ear_history (list): List of EAR values for the past N frames.
        threshold (float): Threshold below which eyes are considered closed.
        
    Returns:
        float: PERCLOS value (0.0 to 1.0).
    """
    if not ear_history:
        return 0.0
        
    closed_frames = sum(1 for ear in ear_history if ear < threshold)
    perclos = closed_frames / len(ear_history)
    return perclos
