import torch
import torch.nn as nn
import torch.nn.functional as F

class RGC2SCNet(nn.Module):
    def __init__(self, input_shape=(2, 7, 7)):
        super(RGC2SCNet, self).__init__()
        
        # Convolutional paths for each channel
        # Path for channel 1
        self.conv1a = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.conv1b = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        
        # Path for channel 2
        self.conv2a = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, padding=2)
        self.conv2b = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, padding=2)
        
        # Calculate the size for the first fully connected layer
        self.fc1_input_size = self._get_conv_output(input_shape)
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.fc1_input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        
        # Output layers
        self.fc_x = nn.Linear(64, 1)   # For x-coordinate
        self.fc_y = nn.Linear(64, 1)   # For y-coordinate
        self.fc_exist = nn.Linear(64, 1)  # For existence (0 or 1)
        
    def _get_conv_output(self, shape):
        """Helper function to calculate the output size after convolutions."""
        x = torch.zeros(1, *shape)  # Create a dummy input tensor with two channels
        # Split the dummy input
        x1, x2 = x[:, 0:1, :, :], x[:, 1:2, :, :]
        
        # Pass through each path
        x11 = F.relu(self.conv1a(x1))
        x12 = F.relu(self.conv1b(x1))
        
        x21 = F.relu(self.conv2a(x2))
        x22 = F.relu(self.conv2b(x2))

        # Concatenate along the channel dimension
        x_concat = torch.cat((x11, x21, x21, x22), dim=1)
        return int(x_concat.numel() / x_concat.size(0))  # Total features per batch element
    
    def forward(self, x):
        # Split the input along the channel dimension
        x1, x2 = x[:, 0:1, :, :], x[:, 1:2, :, :]
        
        # Pass through each path
        x11 = F.relu(self.conv1a(x1))
        x12 = F.relu(self.conv1b(x1))
        
        x21 = F.relu(self.conv2a(x2))
        x22 = F.relu(self.conv2b(x2))
        
        # Concatenate along the channel dimension
        x_concat = torch.cat((x11, x21, x21, x22), dim=1)
        
        # Flatten and pass through fully connected layers
        x_flat = x_concat.view(x_concat.size(0), -1)
        x = F.relu(self.fc1(x_flat))
        features = F.relu(self.fc2(x))
        
        # Output branches
        x_out = self.fc_x(features)
        y_out = self.fc_y(features)
        exist_out = torch.sigmoid(self.fc_exist(features))  # Sigmoid for binary classification
        
        return x_out, y_out, exist_out
