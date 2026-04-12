import torch
import torch.nn as nn
import torch.nn.functional as F

class TCDCN(nn.Module):
    def __init__(self, num_landmarks=68):
        super(TCDCN, self).__init__()
        
        # Shared layers
        # Input: 1 x 40 x 40
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # -> 16 x 20 x 20
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # -> 32 x 10 x 10
        
        self.conv3 = nn.Conv2d(32, 48, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(48)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) # -> 48 x 5 x 5
        
        self.conv4 = nn.Conv2d(48, 64, kernel_size=2, stride=1, padding=0) # -> 64 x 4 x 4
        self.bn4 = nn.BatchNorm2d(64)
        
        # Flatten: 64 * 4 * 4 = 1024
        self.fc = nn.Linear(64 * 4 * 4, 256)
        self.dropout = nn.Dropout(0.5)
        
        # Task-specific heads
        self.fc_landmarks = nn.Linear(256, num_landmarks * 2)
        
        # Deeper heads for attributes to handle imbalance/complexity
        self.gender_head = nn.Sequential(nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 2))
        self.glasses_head = nn.Sequential(nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 2))
        self.pose_head = nn.Sequential(nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 5))
        self.smiling_head = nn.Sequential(nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 2))

    def forward(self, x):
        # Shared layers
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        
        x = F.relu(self.bn4(self.conv4(x)))
        
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        x = self.dropout(x)
        
        # Heads
        landmarks = self.fc_landmarks(x)
        gender = self.gender_head(x)
        glasses = self.glasses_head(x)
        pose = self.pose_head(x)
        smiling = self.smiling_head(x)
        
        return landmarks, gender, glasses, pose, smiling
