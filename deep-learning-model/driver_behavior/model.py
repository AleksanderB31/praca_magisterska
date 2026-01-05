import torch
import torch.nn as nn

class TinyVGG(nn.Module):
    def __init__(self, input_shape, hidden_units, output_shape):
        """
        Args:
            input_shape (int): Number of input channels (e.g., 3 for RGB).
            hidden_units (int): Number of filters in convolutional layers.
            output_shape (int): Number of output classes.
        """
        super().__init__()
        
        # Block 1
        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, 
                      out_channels=hidden_units, 
                      kernel_size=3, 
                      stride=1, 
                      padding=1), 
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, 
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) 
        )
        
        # Block 2
        self.block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units, 
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, 
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) 
        )
        
        # Block 3
        self.block_3 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units, 
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, 
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU()
            # Note: Paper analysis suggests no MaxPool here, or it was omitted.
            # With 64x64 input:
            # Block 1 -> 32x32
            # Block 2 -> 16x16
            # Block 3 -> 16x16
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            # 16*16*hidden_units
            nn.Linear(in_features=hidden_units * 16 * 16, 
                      out_features=output_shape)
        )
    
    def forward(self, x):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.classifier(x)
        return x
