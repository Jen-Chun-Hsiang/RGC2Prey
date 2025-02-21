import torch
import torch.nn as nn
import torch.nn.functional as F


import torch
import torch.nn as nn

class ParallelCNNFeatureExtractor(nn.Module):
    def __init__(self, input_height=24, input_width=32, conv_out_channels=16, fc_out_features=128, num_input_channel=1):
        super(ParallelCNNFeatureExtractor, self).__init__()
        
        # Define parallel convolution layers with different kernel sizes
        self.conv1 = nn.Conv2d(num_input_channel, conv_out_channels, kernel_size=4, stride=2, padding=0)
        self.bn1 = nn.BatchNorm2d(conv_out_channels)
        
        self.conv2 = nn.Conv2d(num_input_channel, conv_out_channels, kernel_size=4, dilation=4, stride=8, padding=4)
        self.bn2 = nn.BatchNorm2d(conv_out_channels)
        
        self.conv3 = nn.Conv2d(num_input_channel, conv_out_channels, kernel_size=4, dilation=8, stride=16, padding=8)
        self.bn3 = nn.BatchNorm2d(conv_out_channels)
        
        # Pooling layer to reduce spatial dimensions
        self.pool = nn.MaxPool2d(2, 2)
        
        # Determine the flattened feature size after the convolution and pooling layers
        with torch.no_grad():
            mock_input = torch.zeros(1, num_input_channel, input_height, input_width)
            mock_output = self._forward_conv_layers(mock_input)
            self.flat_feature_size = mock_output.size(1)  # Total flattened size after concatenation

        # Fully connected layer based on the flattened feature size
        self.fc = nn.Linear(self.flat_feature_size, fc_out_features)

    def _forward_conv_layers(self, x):
        """Forward pass through each convolutional layer, flatten, and concatenate their outputs."""
        x1 = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x2 = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x3 = self.pool(torch.relu(self.bn3(self.conv3(x))))
        
        # Flatten each output
        x1_flat = x1.view(x1.size(0), -1)
        x2_flat = x2.view(x2.size(0), -1)
        x3_flat = x3.view(x3.size(0), -1)
        
        # Concatenate flattened features
        x = torch.cat((x1_flat, x2_flat, x3_flat), dim=1)
        return x

    def forward(self, x):
        x = self._forward_conv_layers(x)
        x = self.fc(x)  # Project to feature vector of specified size
        return x
    

class ParallelCNNFeatureExtractor2(nn.Module):
    def __init__(self, input_height=24, input_width=32, conv_out_channels=16, fc_out_features=128, num_input_channel=1):
        super(ParallelCNNFeatureExtractor2, self).__init__()
        
        # Define parallel convolution layers with different kernel sizes (Series A)
        conv_out_channels_a = int(conv_out_channels*0.5)
        self.conv1_a = nn.Conv2d(num_input_channel, conv_out_channels_a, kernel_size=4, stride=2, padding=0)
        self.bn1_a = nn.BatchNorm2d(conv_out_channels_a)
        
        self.conv2_a = nn.Conv2d(num_input_channel, conv_out_channels_a, kernel_size=4, dilation=4, stride=8, padding=4)
        self.bn2_a = nn.BatchNorm2d(conv_out_channels_a)
        
        self.conv3_a = nn.Conv2d(num_input_channel, conv_out_channels_a, kernel_size=4, dilation=8, stride=16, padding=8)
        self.bn3_a = nn.BatchNorm2d(conv_out_channels_a)
        
        # Define additional parallel convolution layers (Series B)
        conv_out_channels_b = conv_out_channels
        self.conv1_b = nn.Conv2d(conv_out_channels_a, conv_out_channels_b, kernel_size=4, stride=1, padding=0)
        self.bn1_b = nn.BatchNorm2d(conv_out_channels_b)
        
        self.conv2_b = nn.Conv2d(conv_out_channels_a, conv_out_channels_b, kernel_size=3, stride=1, padding=0)
        self.bn2_b = nn.BatchNorm2d(conv_out_channels_b)
        
        self.conv3_b = nn.Conv2d(conv_out_channels_a, conv_out_channels_b, kernel_size=2, stride=1, padding=0)
        self.bn3_b = nn.BatchNorm2d(conv_out_channels_b)
        
        # Determine the flattened feature size after the convolution and pooling layers
        with torch.no_grad():
            mock_input = torch.zeros(1, num_input_channel, input_height, input_width)
            mock_output = self._forward_conv_layers(mock_input)
            self.flat_feature_size = mock_output.size(1)  # Total flattened size after concatenation

        # Fully connected layer based on the flattened feature size
        self.fc = nn.Linear(self.flat_feature_size, fc_out_features)

    def _forward_conv_layers(self, x):
        """Forward pass through each convolutional layer, flatten, and concatenate their outputs."""
        # Series A
        x1 = torch.relu(self.bn1_a(self.conv1_a(x)))
        x2 = torch.relu(self.bn2_a(self.conv2_a(x)))
        x3 = torch.relu(self.bn3_a(self.conv3_a(x)))
        
        # Series B
        x1 = torch.relu(self.bn1_b(self.conv1_b(x1)))
        x2 = torch.relu(self.bn2_b(self.conv2_b(x2)))
        x3 = torch.relu(self.bn3_b(self.conv3_b(x3)))
        
        x1_flat = x1.view(x1.size(0), -1)
        x2_flat = x2.view(x2.size(0), -1)
        x3_flat = x3.view(x3.size(0), -1)
        
        # Concatenate flattened features from both series
        x = torch.cat((x1_flat, x2_flat, x3_flat), dim=1)
        return x

    def forward(self, x):
        x = self._forward_conv_layers(x)
        x = self.fc(x)  # Project to feature vector of specified size
        return x
    

class ParallelCNNFeatureExtractor3(nn.Module):
    def __init__(self, input_height=24, input_width=32, conv_out_channels=16, fc_out_features=128, num_input_channel=1):
        super(ParallelCNNFeatureExtractor3, self).__init__()
        
        # Define parallel convolution layers with different kernel sizes (Series A)
        conv_out_channels_a = int(conv_out_channels*0.5)
        self.conv1_a = nn.Conv2d(num_input_channel, conv_out_channels_a, kernel_size=4, stride=2, padding=0)
        self.bn1_a = nn.BatchNorm2d(conv_out_channels_a)
        
        self.conv2_a = nn.Conv2d(num_input_channel, conv_out_channels_a, kernel_size=4, dilation=4, stride=8, padding=4)
        self.bn2_a = nn.BatchNorm2d(conv_out_channels_a)
        
        self.conv3_a = nn.Conv2d(num_input_channel, conv_out_channels_a, kernel_size=4, dilation=8, stride=16, padding=8)
        self.bn3_a = nn.BatchNorm2d(conv_out_channels_a)
        
        # Define additional parallel convolution layers (Series B)
        conv_out_channels_b = conv_out_channels
        self.conv1_b = nn.Conv2d(conv_out_channels_a, conv_out_channels_b, kernel_size=4, stride=1, padding=0)
        self.bn1_b = nn.BatchNorm2d(conv_out_channels_b)
        
        self.conv2_b = nn.Conv2d(conv_out_channels_a, conv_out_channels_b, kernel_size=3, stride=1, padding=0)
        self.bn2_b = nn.BatchNorm2d(conv_out_channels_b)
        
        self.conv3_b = nn.Conv2d(conv_out_channels_a, conv_out_channels_b, kernel_size=2, stride=1, padding=0)
        self.bn3_b = nn.BatchNorm2d(conv_out_channels_b)

        # Pooling layer to reduce spatial dimensions
        self.pool = nn.MaxPool2d(2, 2)
        
        # Determine the flattened feature size after the convolution and pooling layers
        with torch.no_grad():
            mock_input = torch.zeros(1, num_input_channel, input_height, input_width)
            mock_output = self._forward_conv_layers(mock_input)
            self.flat_feature_size = mock_output.size(1)  # Total flattened size after concatenation

        # Fully connected layer based on the flattened feature size
        self.fc = nn.Linear(self.flat_feature_size, fc_out_features)

    def _forward_conv_layers(self, x):
        """Forward pass through each convolutional layer, flatten, and concatenate their outputs."""
        # Series A
        x1_a = torch.relu(self.bn1_a(self.conv1_a(x)))
        x2_a = torch.relu(self.bn2_a(self.conv2_a(x)))
        x3_a = torch.relu(self.bn3_a(self.conv3_a(x)))
        
        # Series B
        x1_b = torch.relu(self.bn1_b(self.conv1_b(x1_a)))
        x2_b = torch.relu(self.bn2_b(self.conv2_b(x2_a)))
        x3_b = torch.relu(self.bn3_b(self.conv3_b(x3_a)))

        x1_a = self.pool(x1_a)
        x2_a = self.pool(x2_a)
        x3_a = self.pool(x3_a)
        x1_a_flat = x1_a.view(x1_a.size(0), -1)
        x2_a_flat = x1_a.view(x2_a.size(0), -1)
        x3_a_flat = x1_a.view(x3_a.size(0), -1)
        
        x1_b_flat = x1_b.view(x1_b.size(0), -1)
        x2_b_flat = x2_b.view(x2_b.size(0), -1)
        x3_b_flat = x3_b.view(x3_b.size(0), -1)
        
        # Concatenate flattened features from both series
        x = torch.cat((x1_a_flat, x2_a_flat, x3_a_flat, x1_b_flat, x2_b_flat, x3_b_flat), dim=1)
        return x

    def forward(self, x):
        x = self._forward_conv_layers(x)
        x = self.fc(x)  # Project to feature vector of specified size
        return x
    

class ParallelCNNFeatureExtractor4(nn.Module):
    def __init__(self, input_height=24, input_width=32, conv_out_channels=16, fc_out_features=128, num_input_channel=1):
        super(ParallelCNNFeatureExtractor4, self).__init__()
        
        # Define parallel convolution layers with different kernel sizes (Series A)
        conv_out_channels_a = int(conv_out_channels*0.5)
        self.conv1_a = nn.Conv2d(num_input_channel, conv_out_channels_a, kernel_size=4, stride=2, padding=0)
        self.bn1_a = nn.BatchNorm2d(conv_out_channels_a)
        
        self.conv2_a = nn.Conv2d(num_input_channel, conv_out_channels_a, kernel_size=4, dilation=4, stride=8, padding=4)
        self.bn2_a = nn.BatchNorm2d(conv_out_channels_a)
        
        self.conv3_a = nn.Conv2d(num_input_channel, conv_out_channels_a, kernel_size=4, dilation=8, stride=16, padding=8)
        self.bn3_a = nn.BatchNorm2d(conv_out_channels_a)
        
        # Define additional parallel convolution layers (Series B)
        conv_out_channels_b = conv_out_channels
        self.conv1_b = nn.Conv2d(conv_out_channels_a, conv_out_channels_b, kernel_size=4, stride=1, padding=0)
        self.bn1_b = nn.BatchNorm2d(conv_out_channels_b)
        
        self.conv2_b = nn.Conv2d(conv_out_channels_a, conv_out_channels_b, kernel_size=3, stride=1, padding=0)
        self.bn2_b = nn.BatchNorm2d(conv_out_channels_b)
        
        self.conv3_b = nn.Conv2d(conv_out_channels_a, conv_out_channels_b, kernel_size=2, stride=1, padding=0)
        self.bn3_b = nn.BatchNorm2d(conv_out_channels_b)

        # Define additional parallel convolution layers (Series C)      
        self.conv1_c = nn.Conv2d(conv_out_channels_b, conv_out_channels_b, kernel_size=3, stride=1, padding=0)
        self.bn1_c = nn.BatchNorm2d(conv_out_channels_b)
        
        self.conv2_c = nn.Conv2d(conv_out_channels_b, conv_out_channels_b, kernel_size=3, stride=1, padding=0)
        self.bn2_c = nn.BatchNorm2d(conv_out_channels_b)
        
        self.conv3_c = nn.Conv2d(conv_out_channels_b, conv_out_channels_b, kernel_size=3, stride=1, padding=0)
        self.bn3_c = nn.BatchNorm2d(conv_out_channels_b)
        
        # Determine the flattened feature size after the convolution and pooling layers
        with torch.no_grad():
            mock_input = torch.zeros(1, num_input_channel, input_height, input_width)
            mock_output = self._forward_conv_layers(mock_input)
            self.flat_feature_size = mock_output.size(1)  # Total flattened size after concatenation

        # Fully connected layer based on the flattened feature size
        self.fc = nn.Linear(self.flat_feature_size, fc_out_features)

    def _forward_conv_layers(self, x):
        """Forward pass through each convolutional layer, flatten, and concatenate their outputs."""
        # Series A
        x1 = torch.relu(self.bn1_a(self.conv1_a(x)))
        x2 = torch.relu(self.bn2_a(self.conv2_a(x)))
        x3 = torch.relu(self.bn3_a(self.conv3_a(x)))
        
        # Series B
        x1 = torch.relu(self.bn1_b(self.conv1_b(x1)))
        x2 = torch.relu(self.bn2_b(self.conv2_b(x2)))
        x3 = torch.relu(self.bn3_b(self.conv3_b(x3)))

        # Series C
        x1 = torch.relu(self.bn1_c(self.conv1_c(x1)))
        x2 = torch.relu(self.bn2_c(self.conv2_c(x2)))
        x3 = torch.relu(self.bn3_c(self.conv3_c(x3)))
        
        x1_flat = x1.view(x1.size(0), -1)
        x2_flat = x2.view(x2.size(0), -1)
        x3_flat = x3.view(x3.size(0), -1)
        
        # Concatenate flattened features from both series
        x = torch.cat((x1_flat, x2_flat, x3_flat), dim=1)
        return x

    def forward(self, x):
        x = self._forward_conv_layers(x)
        x = self.fc(x)  # Project to feature vector of specified size
        return x
    

class ParallelCNNFeatureExtractor4s(nn.Module):
    def __init__(self, input_height=24, input_width=32, conv_out_channels=16, fc_out_features=128, num_input_channel=1):
        super(ParallelCNNFeatureExtractor4s, self).__init__()
        
        # Define parallel convolution layers with different kernel sizes (Series A)
        conv_out_channels_a = int(conv_out_channels*0.5)
        self.conv1_a = nn.Conv2d(num_input_channel, conv_out_channels_a, kernel_size=4, stride=2, padding=0)
        self.bn1_a = nn.BatchNorm2d(conv_out_channels_a)
        
        self.conv2_a = nn.Conv2d(num_input_channel, conv_out_channels_a, kernel_size=4, dilation=4, stride=8, padding=4)
        self.bn2_a = nn.BatchNorm2d(conv_out_channels_a)
        
        self.conv3_a = nn.Conv2d(num_input_channel, conv_out_channels_a, kernel_size=4, dilation=8, stride=16, padding=8)
        self.bn3_a = nn.BatchNorm2d(conv_out_channels_a)
        
        # Define additional parallel convolution layers (Series B)
        conv_out_channels_b = conv_out_channels
        self.conv1_b = nn.Conv2d(conv_out_channels_a, conv_out_channels_b, kernel_size=4, stride=1, padding=0)
        self.bn1_b = nn.BatchNorm2d(conv_out_channels_b)
        
        self.conv2_b = nn.Conv2d(conv_out_channels_a, conv_out_channels_b, kernel_size=3, stride=1, padding=0)
        self.bn2_b = nn.BatchNorm2d(conv_out_channels_b)
        
        self.conv3_b = nn.Conv2d(conv_out_channels_a, conv_out_channels_b, kernel_size=2, stride=1, padding=0)
        self.bn3_b = nn.BatchNorm2d(conv_out_channels_b)

        # Define additional parallel convolution layers (Series C)      
        self.conv1_c = nn.Conv2d(conv_out_channels_b, conv_out_channels_b, kernel_size=3, stride=1, padding=0)
        self.bn1_c = nn.BatchNorm2d(conv_out_channels_b)
        
        self.conv2_c = nn.Conv2d(conv_out_channels_b, conv_out_channels_b, kernel_size=3, stride=1, padding=0)
        self.bn2_c = nn.BatchNorm2d(conv_out_channels_b)
        
        self.conv3_c = nn.Conv2d(conv_out_channels_b, conv_out_channels_b, kernel_size=3, stride=1, padding=0)
        self.bn3_c = nn.BatchNorm2d(conv_out_channels_b)
        
        # Determine the flattened feature size after the convolution and pooling layers
        with torch.no_grad():
            mock_input = torch.zeros(1, num_input_channel, input_height, input_width)
            mock_output = self._forward_conv_layers(mock_input)
            self.flat_feature_size = mock_output.size(1)  # Total flattened size after concatenation

        # Fully connected layer based on the flattened feature size
        self.fc = nn.Linear(self.flat_feature_size, fc_out_features)

    def _forward_conv_layers(self, x):
        """Forward pass through each convolutional layer, flatten, and concatenate their outputs."""
        # Series A
        x1 = torch.relu(self.bn1_a(self.conv1_a(x)))
        x2 = torch.relu(self.bn2_a(self.conv2_a(x)))
        x3 = torch.relu(self.bn3_a(self.conv3_a(x)))
        
        # Series B
        x1 = torch.relu(self.bn1_b(self.conv1_b(x1)))
        x2 = torch.relu(self.bn2_b(self.conv2_b(x2)))
        x3 = torch.relu(self.bn3_b(self.conv3_b(x3)))

        # Series C
        x1 = torch.relu(self.bn1_c(self.conv1_c(x1)))
        x2 = torch.relu(self.bn2_c(self.conv2_c(x2)))
        x3 = torch.relu(self.bn3_c(self.conv3_c(x3)))
        
        x1_flat = x1.view(x1.size(0), -1)
        x2_flat = x2.view(x2.size(0), -1)
        x3_flat = x3.view(x3.size(0), -1)
        
        # Concatenate flattened features from both series
        x = torch.cat((x1_flat, x2_flat, x3_flat), dim=1)
        return x

    def forward(self, x):
        x = self._forward_conv_layers(x)
        x = self.fc(x)  # Project to feature vector of specified size
        return x
    
    
class ParallelCNNFeatureExtractor41(nn.Module):
    def __init__(self, input_height=24, input_width=32, conv_out_channels=16, fc_out_features=128, num_input_channel=1):
        super(ParallelCNNFeatureExtractor41, self).__init__()
        
        # Define parallel convolution layers with different kernel sizes (Series A)
        conv_out_channels_a = int(conv_out_channels*0.5)
        self.conv1_a = nn.Conv2d(num_input_channel, conv_out_channels_a, kernel_size=3, stride=1, padding=0)
        self.bn1_a = nn.BatchNorm2d(conv_out_channels_a)
        
        self.conv2_a = nn.Conv2d(num_input_channel, conv_out_channels_a, kernel_size=4, dilation=4, stride=8, padding=4)
        self.bn2_a = nn.BatchNorm2d(conv_out_channels_a)
        
        self.conv3_a = nn.Conv2d(num_input_channel, conv_out_channels_a, kernel_size=4, dilation=8, stride=16, padding=8)
        self.bn3_a = nn.BatchNorm2d(conv_out_channels_a)
        
        # Define additional parallel convolution layers (Series B)
        conv_out_channels_b = conv_out_channels
        self.conv1_b = nn.Conv2d(conv_out_channels_a, conv_out_channels_b, kernel_size=4, stride=1, padding=0)
        self.bn1_b = nn.BatchNorm2d(conv_out_channels_b)
        
        self.conv2_b = nn.Conv2d(conv_out_channels_a, conv_out_channels_b, kernel_size=3, stride=1, padding=0)
        self.bn2_b = nn.BatchNorm2d(conv_out_channels_b)
        
        self.conv3_b = nn.Conv2d(conv_out_channels_a, conv_out_channels_b, kernel_size=2, stride=1, padding=0)
        self.bn3_b = nn.BatchNorm2d(conv_out_channels_b)

        # Define additional parallel convolution layers (Series C)      
        self.conv1_c = nn.Conv2d(conv_out_channels_b, conv_out_channels_b, kernel_size=3, stride=1, padding=0)
        self.bn1_c = nn.BatchNorm2d(conv_out_channels_b)
        
        self.conv2_c = nn.Conv2d(conv_out_channels_b, conv_out_channels_b, kernel_size=3, stride=1, padding=0)
        self.bn2_c = nn.BatchNorm2d(conv_out_channels_b)
        
        self.conv3_c = nn.Conv2d(conv_out_channels_b, conv_out_channels_b, kernel_size=3, stride=1, padding=0)
        self.bn3_c = nn.BatchNorm2d(conv_out_channels_b)
        
        # Determine the flattened feature size after the convolution and pooling layers
        with torch.no_grad():
            mock_input = torch.zeros(1, num_input_channel, input_height, input_width)
            mock_output = self._forward_conv_layers(mock_input)
            self.flat_feature_size = mock_output.size(1)  # Total flattened size after concatenation

        # Fully connected layer based on the flattened feature size
        self.fc = nn.Linear(self.flat_feature_size, fc_out_features)

    def _forward_conv_layers(self, x):
        """Forward pass through each convolutional layer, flatten, and concatenate their outputs."""
        # Series A
        x1 = torch.relu(self.bn1_a(self.conv1_a(x)))
        x2 = torch.relu(self.bn2_a(self.conv2_a(x)))
        x3 = torch.relu(self.bn3_a(self.conv3_a(x)))
        
        # Series B
        x1 = torch.relu(self.bn1_b(self.conv1_b(x1)))
        x2 = torch.relu(self.bn2_b(self.conv2_b(x2)))
        x3 = torch.relu(self.bn3_b(self.conv3_b(x3)))

        # Series C
        x1 = torch.relu(self.bn1_c(self.conv1_c(x1)))
        x2 = torch.relu(self.bn2_c(self.conv2_c(x2)))
        x3 = torch.relu(self.bn3_c(self.conv3_c(x3)))
        
        x1_flat = x1.view(x1.size(0), -1)
        x2_flat = x2.view(x2.size(0), -1)
        x3_flat = x3.view(x3.size(0), -1)
        
        # Concatenate flattened features from both series
        x = torch.cat((x1_flat, x2_flat, x3_flat), dim=1)
        return x

    def forward(self, x):
        x = self._forward_conv_layers(x)
        x = self.fc(x)  # Project to feature vector of specified size
        return x
    

class ParallelCNNFeatureExtractor42(nn.Module):
    def __init__(self, input_height=24, input_width=32, conv_out_channels=16, fc_out_features=128, num_input_channel=1):
        super(ParallelCNNFeatureExtractor42, self).__init__()
        
        # Define parallel convolution layers with different kernel sizes (Series A)
        conv_out_channels_a = int(conv_out_channels*0.5)
        self.conv1_a = nn.Conv2d(num_input_channel, conv_out_channels_a, kernel_size=3, stride=1, padding=0)
        self.bn1_a = nn.BatchNorm2d(conv_out_channels_a)
        
        self.conv2_a = nn.Conv2d(num_input_channel, conv_out_channels_a, kernel_size=4, dilation=4, stride=8, padding=4)
        self.bn2_a = nn.BatchNorm2d(conv_out_channels_a)
        
        self.conv3_a = nn.Conv2d(num_input_channel, conv_out_channels_a, kernel_size=4, dilation=8, stride=16, padding=8)
        self.bn3_a = nn.BatchNorm2d(conv_out_channels_a)
        
        # Define additional parallel convolution layers (Series B)
        conv_out_channels_b = conv_out_channels
        self.conv1_b = nn.Conv2d(conv_out_channels_a, conv_out_channels_b, kernel_size=3, stride=1, padding=0)
        self.bn1_b = nn.BatchNorm2d(conv_out_channels_b)
        
        self.conv2_b = nn.Conv2d(conv_out_channels_a, conv_out_channels_b, kernel_size=3, stride=1, padding=0)
        self.bn2_b = nn.BatchNorm2d(conv_out_channels_b)
        
        self.conv3_b = nn.Conv2d(conv_out_channels_a, conv_out_channels_b, kernel_size=2, stride=1, padding=0)
        self.bn3_b = nn.BatchNorm2d(conv_out_channels_b)

        # Define additional parallel convolution layers (Series C)      
        self.conv1_c = nn.Conv2d(conv_out_channels_b, conv_out_channels_b, kernel_size=3, stride=1, padding=0)
        self.bn1_c = nn.BatchNorm2d(conv_out_channels_b)
        
        self.conv2_c = nn.Conv2d(conv_out_channels_b, conv_out_channels_b, kernel_size=3, stride=1, padding=0)
        self.bn2_c = nn.BatchNorm2d(conv_out_channels_b)
        
        self.conv3_c = nn.Conv2d(conv_out_channels_b, conv_out_channels_b, kernel_size=3, stride=1, padding=0)
        self.bn3_c = nn.BatchNorm2d(conv_out_channels_b)
        
        # Determine the flattened feature size after the convolution and pooling layers
        with torch.no_grad():
            mock_input = torch.zeros(1, num_input_channel, input_height, input_width)
            mock_output = self._forward_conv_layers(mock_input)
            self.flat_feature_size = mock_output.size(1)  # Total flattened size after concatenation

        # Fully connected layer based on the flattened feature size
        self.fc = nn.Linear(self.flat_feature_size, fc_out_features)

    def _forward_conv_layers(self, x):
        """Forward pass through each convolutional layer, flatten, and concatenate their outputs."""
        # Series A
        x1 = torch.relu(self.bn1_a(self.conv1_a(x)))
        x2 = torch.relu(self.bn2_a(self.conv2_a(x)))
        x3 = torch.relu(self.bn3_a(self.conv3_a(x)))
        
        # Series B
        x1 = torch.relu(self.bn1_b(self.conv1_b(x1)))
        x2 = torch.relu(self.bn2_b(self.conv2_b(x2)))
        x3 = torch.relu(self.bn3_b(self.conv3_b(x3)))

        # Series C
        x1 = torch.relu(self.bn1_c(self.conv1_c(x1)))
        x2 = torch.relu(self.bn2_c(self.conv2_c(x2)))
        x3 = torch.relu(self.bn3_c(self.conv3_c(x3)))
        
        x1_flat = x1.view(x1.size(0), -1)
        x2_flat = x2.view(x2.size(0), -1)
        x3_flat = x3.view(x3.size(0), -1)
        
        # Concatenate flattened features from both series
        x = torch.cat((x1_flat, x2_flat, x3_flat), dim=1)
        return x

    def forward(self, x):
        x = self._forward_conv_layers(x)
        x = self.fc(x)  # Project to feature vector of specified size
        return x
    

class ParallelCNNFeatureExtractor5(nn.Module):
    def __init__(self, input_height=24, input_width=32, conv_out_channels=16, fc_out_features=128, num_input_channel=1):
        super(ParallelCNNFeatureExtractor5, self).__init__()
        
        # Define parallel convolution layers with different kernel sizes (Series A)
        conv_out_channels_a = int(conv_out_channels*0.5)
        self.conv1_a = nn.Conv2d(num_input_channel, conv_out_channels_a, kernel_size=4, stride=2, padding=0)
        self.bn1_a = nn.BatchNorm2d(conv_out_channels_a)
        
        self.conv2_a = nn.Conv2d(num_input_channel, conv_out_channels_a, kernel_size=4, dilation=4, stride=8, padding=4)
        self.bn2_a = nn.BatchNorm2d(conv_out_channels_a)
        
        self.conv3_a = nn.Conv2d(num_input_channel, conv_out_channels_a, kernel_size=4, dilation=8, stride=16, padding=8)
        self.bn3_a = nn.BatchNorm2d(conv_out_channels_a)
        
        # Define additional parallel convolution layers (Series B)
        conv_out_channels_b = conv_out_channels
        self.conv1_b = nn.Conv2d(conv_out_channels_a, conv_out_channels_b, kernel_size=4, stride=1, padding=0)
        self.bn1_b = nn.BatchNorm2d(conv_out_channels_b)
        
        self.conv2_b = nn.Conv2d(conv_out_channels_a, conv_out_channels_b, kernel_size=3, stride=1, padding=0)
        self.bn2_b = nn.BatchNorm2d(conv_out_channels_b)
        
        self.conv3_b = nn.Conv2d(conv_out_channels_a, conv_out_channels_b, kernel_size=2, stride=1, padding=0)
        self.bn3_b = nn.BatchNorm2d(conv_out_channels_b)

        # Define additional parallel convolution layers (Series C)      
        self.conv1_c = nn.Conv2d(conv_out_channels_b, conv_out_channels_b, kernel_size=3, stride=1, padding=0)
        self.bn1_c = nn.BatchNorm2d(conv_out_channels_b)
        
        self.conv2_c = nn.Conv2d(conv_out_channels_b, conv_out_channels_b, kernel_size=3, stride=1, padding=0)
        self.bn2_c = nn.BatchNorm2d(conv_out_channels_b)
        
        self.conv3_c = nn.Conv2d(conv_out_channels_b, conv_out_channels_b, kernel_size=3, stride=1, padding=0)
        self.bn3_c = nn.BatchNorm2d(conv_out_channels_b)
        
        # Determine the flattened feature size after the convolution and pooling layers
        with torch.no_grad():
            mock_input = torch.zeros(1, num_input_channel, input_height, input_width)
            mock_output = self._forward_conv_layers(mock_input)
            self.flat_feature_size = mock_output.size(1)  # Total flattened size after concatenation

        # Fully connected layer based on the flattened feature size
        self.fc = nn.Linear(self.flat_feature_size, fc_out_features)

    def _forward_conv_layers(self, x):
        """Forward pass through each convolutional layer, flatten, and concatenate their outputs."""
        # Series A
        x1 = torch.relu(self.bn1_a(self.conv1_a(x)))
        x2 = torch.relu(self.bn2_a(self.conv2_a(x)))
        x3 = torch.relu(self.bn3_a(self.conv3_a(x)))
        
        # Series B
        x1_b = torch.relu(self.bn1_b(self.conv1_b(x1)))
        x2_b = torch.relu(self.bn2_b(self.conv2_b(x2)))
        x3_b = torch.relu(self.bn3_b(self.conv3_b(x3)))

        # Series C
        x1_c = torch.relu(self.bn1_c(self.conv1_c(x1_b)))
        x2_c = torch.relu(self.bn2_c(self.conv2_c(x2_b)))
        x3_c = torch.relu(self.bn3_c(self.conv3_c(x3_b)))
        
        x1_b_flat = x1_b.view(x1_b.size(0), -1)
        x2_b_flat = x2_b.view(x2_b.size(0), -1)
        x3_b_flat = x3_b.view(x3_b.size(0), -1)

        x1_c_flat = x1_c.view(x1_c.size(0), -1)
        x2_c_flat = x2_c.view(x2_c.size(0), -1)
        x3_c_flat = x3_c.view(x3_c.size(0), -1)
        
        # Concatenate flattened features from both series
        x = torch.cat((x1_b_flat, x2_b_flat, x3_b_flat, x1_c_flat, x2_c_flat, x3_c_flat), dim=1)
        return x

    def forward(self, x):
        x = self._forward_conv_layers(x)
        x = self.fc(x)  # Project to feature vector of specified size
        return x
    

class ParallelCNNFeatureExtractor6(nn.Module):
    def __init__(self, input_height=24, input_width=32, conv_out_channels=16, fc_out_features=128, num_input_channel=1):
        super(ParallelCNNFeatureExtractor6, self).__init__()
        
        # Define parallel convolution layers with different kernel sizes (Series A)
        conv_out_channels_a = int(conv_out_channels*0.1)
        self.conv1_a = nn.Conv2d(num_input_channel, conv_out_channels_a, kernel_size=4, stride=2, padding=0)
        self.bn1_a = nn.BatchNorm2d(conv_out_channels_a)
        
        self.conv2_a = nn.Conv2d(num_input_channel, conv_out_channels_a, kernel_size=4, dilation=4, stride=8, padding=4)
        self.bn2_a = nn.BatchNorm2d(conv_out_channels_a)
        
        self.conv3_a = nn.Conv2d(num_input_channel, conv_out_channels_a, kernel_size=4, dilation=8, stride=16, padding=8)
        self.bn3_a = nn.BatchNorm2d(conv_out_channels_a)
        
        # Define additional parallel convolution layers (Series B)
        conv_out_channels_b = int(conv_out_channels*0.25)
        self.conv1_b = nn.Conv2d(conv_out_channels_a, conv_out_channels_b, kernel_size=4, stride=1, padding=0)
        self.bn1_b = nn.BatchNorm2d(conv_out_channels_b)
        
        self.conv2_b = nn.Conv2d(conv_out_channels_a, conv_out_channels_b, kernel_size=3, stride=1, padding=0)
        self.bn2_b = nn.BatchNorm2d(conv_out_channels_b)
        
        self.conv3_b = nn.Conv2d(conv_out_channels_a, conv_out_channels_b, kernel_size=2, stride=1, padding=0)
        self.bn3_b = nn.BatchNorm2d(conv_out_channels_b)

        # Define additional parallel convolution layers (Series C)   
        conv_out_channels_c = int(conv_out_channels*0.5)   
        self.conv1_c = nn.Conv2d(conv_out_channels_b, conv_out_channels_c, kernel_size=3, stride=1, padding=0)
        self.bn1_c = nn.BatchNorm2d(conv_out_channels_c)
        
        self.conv2_c = nn.Conv2d(conv_out_channels_b, conv_out_channels_c, kernel_size=3, stride=1, padding=0)
        self.bn2_c = nn.BatchNorm2d(conv_out_channels_c)
        
        self.conv3_c = nn.Conv2d(conv_out_channels_b, conv_out_channels_c, kernel_size=3, stride=1, padding=0)
        self.bn3_c = nn.BatchNorm2d(conv_out_channels_c)

        # Define additional parallel convolution layers (Series D)  
        conv_out_channels_d = conv_out_channels      
        self.conv1_d = nn.Conv2d(conv_out_channels_c, conv_out_channels_d, kernel_size=3, stride=1, padding=0)
        self.bn1_d = nn.BatchNorm2d(conv_out_channels_d)
        
        self.conv2_d = nn.Conv2d(conv_out_channels_c, conv_out_channels_d, kernel_size=3, stride=1, padding=0)
        self.bn2_d = nn.BatchNorm2d(conv_out_channels_d)

        # Define additional parallel convolution layers (Series E) 
        conv_out_channels_e = int(conv_out_channels*2)       
        self.conv1_e = nn.Conv2d(conv_out_channels_d, conv_out_channels_e, kernel_size=3, stride=1, padding=0)
        self.bn1_e = nn.BatchNorm2d(conv_out_channels_e)
        
        # Determine the flattened feature size after the convolution and pooling layers
        with torch.no_grad():
            mock_input = torch.zeros(1, num_input_channel, input_height, input_width)
            mock_output = self._forward_conv_layers(mock_input)
            self.flat_feature_size = mock_output.size(1)  # Total flattened size after concatenation

        # Fully connected layer based on the flattened feature size
        self.fc = nn.Linear(self.flat_feature_size, fc_out_features)

    def _forward_conv_layers(self, x):
        """Forward pass through each convolutional layer, flatten, and concatenate their outputs."""
        # Series A
        x1 = torch.relu(self.bn1_a(self.conv1_a(x)))
        x2 = torch.relu(self.bn2_a(self.conv2_a(x)))
        x3 = torch.relu(self.bn3_a(self.conv3_a(x)))
        
        # Series B
        x1 = torch.relu(self.bn1_b(self.conv1_b(x1)))
        x2 = torch.relu(self.bn2_b(self.conv2_b(x2)))
        x3 = torch.relu(self.bn3_b(self.conv3_b(x3)))

        # Series C
        x1 = torch.relu(self.bn1_c(self.conv1_c(x1)))
        x2 = torch.relu(self.bn2_c(self.conv2_c(x2)))
        x3 = torch.relu(self.bn3_c(self.conv3_c(x3)))

        # Series D
        x1 = torch.relu(self.bn1_d(self.conv1_d(x1)))
        x2 = torch.relu(self.bn2_d(self.conv2_d(x2)))

        # Series E
        x1 = torch.relu(self.bn1_e(self.conv1_e(x1)))
        
        x1_flat = x1.view(x1.size(0), -1)
        x2_flat = x2.view(x2.size(0), -1)
        x3_flat = x3.view(x3.size(0), -1)
        
        # Concatenate flattened features from both series
        x = torch.cat((x1_flat, x2_flat, x3_flat), dim=1)
        return x

    def forward(self, x):
        x = self._forward_conv_layers(x)
        x = self.fc(x)  # Project to feature vector of specified size
        return x
    

class ParallelCNNFeatureExtractor61(nn.Module):
    def __init__(self, input_height=24, input_width=32, conv_out_channels=16, fc_out_features=128, num_input_channel=1):
        super(ParallelCNNFeatureExtractor61, self).__init__()
        
        # Define parallel convolution layers with different kernel sizes (Series A)
        conv_out_channels_a = int(conv_out_channels*0.25)
        self.conv1_a = nn.Conv2d(num_input_channel, conv_out_channels_a, kernel_size=4, stride=2, padding=0)
        self.bn1_a = nn.BatchNorm2d(conv_out_channels_a)
        
        self.conv2_a = nn.Conv2d(num_input_channel, conv_out_channels_a, kernel_size=4, dilation=4, stride=8, padding=4)
        self.bn2_a = nn.BatchNorm2d(conv_out_channels_a)
        
        self.conv3_a = nn.Conv2d(num_input_channel, conv_out_channels_a, kernel_size=4, dilation=8, stride=16, padding=8)
        self.bn3_a = nn.BatchNorm2d(conv_out_channels_a)
        
        # Define additional parallel convolution layers (Series B)
        conv_out_channels_b = int(conv_out_channels*0.25)
        self.conv1_b = nn.Conv2d(conv_out_channels_a, conv_out_channels_b, kernel_size=4, stride=1, padding=0)
        self.bn1_b = nn.BatchNorm2d(conv_out_channels_b)
        
        self.conv2_b = nn.Conv2d(conv_out_channels_a, conv_out_channels_b, kernel_size=3, stride=1, padding=0)
        self.bn2_b = nn.BatchNorm2d(conv_out_channels_b)
        
        self.conv3_b = nn.Conv2d(conv_out_channels_a, conv_out_channels_b, kernel_size=2, stride=1, padding=0)
        self.bn3_b = nn.BatchNorm2d(conv_out_channels_b)

        # Define additional parallel convolution layers (Series C)   
        conv_out_channels_c = int(conv_out_channels*0.5)   
        self.conv1_c = nn.Conv2d(conv_out_channels_b, conv_out_channels_c, kernel_size=3, stride=1, padding=0)
        self.bn1_c = nn.BatchNorm2d(conv_out_channels_c)
        
        self.conv2_c = nn.Conv2d(conv_out_channels_b, conv_out_channels_c, kernel_size=3, stride=1, padding=0)
        self.bn2_c = nn.BatchNorm2d(conv_out_channels_c)
        
        self.conv3_c = nn.Conv2d(conv_out_channels_b, conv_out_channels_c, kernel_size=3, stride=1, padding=0)
        self.bn3_c = nn.BatchNorm2d(conv_out_channels_c)

        # Define additional parallel convolution layers (Series D)  
        conv_out_channels_d = conv_out_channels      
        self.conv1_d = nn.Conv2d(conv_out_channels_c, conv_out_channels_d, kernel_size=3, stride=1, padding=0)
        self.bn1_d = nn.BatchNorm2d(conv_out_channels_d)
        
        self.conv2_d = nn.Conv2d(conv_out_channels_c, conv_out_channels_d, kernel_size=3, stride=1, padding=0)
        self.bn2_d = nn.BatchNorm2d(conv_out_channels_d)

        # Define additional parallel convolution layers (Series E) 
        conv_out_channels_e = int(conv_out_channels*2)       
        self.conv1_e = nn.Conv2d(conv_out_channels_d, conv_out_channels_e, kernel_size=3, stride=1, padding=0)
        self.bn1_e = nn.BatchNorm2d(conv_out_channels_e)
        
        # Determine the flattened feature size after the convolution and pooling layers
        with torch.no_grad():
            mock_input = torch.zeros(1, num_input_channel, input_height, input_width)
            mock_output = self._forward_conv_layers(mock_input)
            self.flat_feature_size = mock_output.size(1)  # Total flattened size after concatenation

        # Fully connected layer based on the flattened feature size
        self.fc = nn.Linear(self.flat_feature_size, fc_out_features)

    def _forward_conv_layers(self, x):
        """Forward pass through each convolutional layer, flatten, and concatenate their outputs."""
        # Series A
        x1 = torch.relu(self.bn1_a(self.conv1_a(x)))
        x2 = torch.relu(self.bn2_a(self.conv2_a(x)))
        x3 = torch.relu(self.bn3_a(self.conv3_a(x)))
        
        # Series B
        x1 = torch.relu(self.bn1_b(self.conv1_b(x1)))
        x2 = torch.relu(self.bn2_b(self.conv2_b(x2)))
        x3 = torch.relu(self.bn3_b(self.conv3_b(x3)))

        # Series C
        x1 = torch.relu(self.bn1_c(self.conv1_c(x1)))
        x2 = torch.relu(self.bn2_c(self.conv2_c(x2)))
        x3 = torch.relu(self.bn3_c(self.conv3_c(x3)))

        # Series D
        x1 = torch.relu(self.bn1_d(self.conv1_d(x1)))
        x2 = torch.relu(self.bn2_d(self.conv2_d(x2)))

        # Series E
        x1 = torch.relu(self.bn1_e(self.conv1_e(x1)))
        
        x1_flat = x1.view(x1.size(0), -1)
        x2_flat = x2.view(x2.size(0), -1)
        x3_flat = x3.view(x3.size(0), -1)
        
        # Concatenate flattened features from both series
        x = torch.cat((x1_flat, x2_flat, x3_flat), dim=1)
        return x

    def forward(self, x):
        x = self._forward_conv_layers(x)
        x = self.fc(x)  # Project to feature vector of specified size
        return x
    

class ParallelCNNFeatureExtractor62(nn.Module):
    def __init__(self, input_height=24, input_width=32, conv_out_channels=16, fc_out_features=128, num_input_channel=1):
        super(ParallelCNNFeatureExtractor62, self).__init__()
        
        # Define parallel convolution layers with different kernel sizes (Series A)
        conv_out_channels_a = int(conv_out_channels*0.5)
        self.conv1_a = nn.Conv2d(num_input_channel, conv_out_channels_a, kernel_size=4, stride=2, padding=0)
        self.bn1_a = nn.BatchNorm2d(conv_out_channels_a)
        
        self.conv2_a = nn.Conv2d(num_input_channel, conv_out_channels_a, kernel_size=4, dilation=4, stride=8, padding=4)
        self.bn2_a = nn.BatchNorm2d(conv_out_channels_a)
        
        self.conv3_a = nn.Conv2d(num_input_channel, conv_out_channels_a, kernel_size=4, dilation=8, stride=16, padding=8)
        self.bn3_a = nn.BatchNorm2d(conv_out_channels_a)
        
        # Define additional parallel convolution layers (Series B)
        conv_out_channels_b = int(conv_out_channels*0.5)
        self.conv1_b = nn.Conv2d(conv_out_channels_a, conv_out_channels_b, kernel_size=4, stride=1, padding=0)
        self.bn1_b = nn.BatchNorm2d(conv_out_channels_b)
        
        self.conv2_b = nn.Conv2d(conv_out_channels_a, conv_out_channels_b, kernel_size=3, stride=1, padding=0)
        self.bn2_b = nn.BatchNorm2d(conv_out_channels_b)
        
        self.conv3_b = nn.Conv2d(conv_out_channels_a, conv_out_channels_b, kernel_size=2, stride=1, padding=0)
        self.bn3_b = nn.BatchNorm2d(conv_out_channels_b)

        # Define additional parallel convolution layers (Series C)   
        conv_out_channels_c = int(conv_out_channels*0.5)   
        self.conv1_c = nn.Conv2d(conv_out_channels_b, conv_out_channels_c, kernel_size=3, stride=1, padding=0)
        self.bn1_c = nn.BatchNorm2d(conv_out_channels_c)
        
        self.conv2_c = nn.Conv2d(conv_out_channels_b, conv_out_channels_c, kernel_size=3, stride=1, padding=0)
        self.bn2_c = nn.BatchNorm2d(conv_out_channels_c)
        
        self.conv3_c = nn.Conv2d(conv_out_channels_b, conv_out_channels_c, kernel_size=3, stride=1, padding=0)
        self.bn3_c = nn.BatchNorm2d(conv_out_channels_c)

        # Define additional parallel convolution layers (Series D)  
        conv_out_channels_d = conv_out_channels      
        self.conv1_d = nn.Conv2d(conv_out_channels_c, conv_out_channels_d, kernel_size=3, stride=1, padding=0)
        self.bn1_d = nn.BatchNorm2d(conv_out_channels_d)
        
        self.conv2_d = nn.Conv2d(conv_out_channels_c, conv_out_channels_d, kernel_size=3, stride=1, padding=0)
        self.bn2_d = nn.BatchNorm2d(conv_out_channels_d)

        # Define additional parallel convolution layers (Series E) 
        conv_out_channels_e = int(conv_out_channels*2)       
        self.conv1_e = nn.Conv2d(conv_out_channels_d, conv_out_channels_e, kernel_size=3, stride=1, padding=0)
        self.bn1_e = nn.BatchNorm2d(conv_out_channels_e)
        
        # Determine the flattened feature size after the convolution and pooling layers
        with torch.no_grad():
            mock_input = torch.zeros(1, num_input_channel, input_height, input_width)
            mock_output = self._forward_conv_layers(mock_input)
            self.flat_feature_size = mock_output.size(1)  # Total flattened size after concatenation

        # Fully connected layer based on the flattened feature size
        self.fc = nn.Linear(self.flat_feature_size, fc_out_features)

    def _forward_conv_layers(self, x):
        """Forward pass through each convolutional layer, flatten, and concatenate their outputs."""
        # Series A
        x1 = torch.relu(self.bn1_a(self.conv1_a(x)))
        x2 = torch.relu(self.bn2_a(self.conv2_a(x)))
        x3 = torch.relu(self.bn3_a(self.conv3_a(x)))
        
        # Series B
        x1 = torch.relu(self.bn1_b(self.conv1_b(x1)))
        x2 = torch.relu(self.bn2_b(self.conv2_b(x2)))
        x3 = torch.relu(self.bn3_b(self.conv3_b(x3)))

        # Series C
        x1 = torch.relu(self.bn1_c(self.conv1_c(x1)))
        x2 = torch.relu(self.bn2_c(self.conv2_c(x2)))
        x3 = torch.relu(self.bn3_c(self.conv3_c(x3)))

        # Series D
        x1 = torch.relu(self.bn1_d(self.conv1_d(x1)))
        x2 = torch.relu(self.bn2_d(self.conv2_d(x2)))

        # Series E
        x1 = torch.relu(self.bn1_e(self.conv1_e(x1)))
        
        x1_flat = x1.view(x1.size(0), -1)
        x2_flat = x2.view(x2.size(0), -1)
        x3_flat = x3.view(x3.size(0), -1)
        
        # Concatenate flattened features from both series
        x = torch.cat((x1_flat, x2_flat, x3_flat), dim=1)
        return x

    def forward(self, x):
        x = self._forward_conv_layers(x)
        x = self.fc(x)  # Project to feature vector of specified size
        return x


class FullSampleNormalization(nn.Module):
    def forward(self, x):
        # Compute mean and std over (sequence, channels, height, width) for each sample
        mean = x.mean(dim=(1, 2, 3, 4), keepdim=True)  # Mean over sequence, C, H, W
        std = x.std(dim=(1, 2, 3, 4), keepdim=True) + 1e-6  # Std over sequence, C, H, W
        return (x - mean) / std   
    

class ChannelWiseNormalization(nn.Module):
    def forward(self, x):
        # Compute mean and std over (sequence, height, width) for each channel separately
        mean = x.mean(dim=(1, 3, 4), keepdim=True)  # Mean over sequence, H, W
        std = x.std(dim=(1, 3, 4), keepdim=True) + 1e-6  # Std over sequence, H, W
        return (x - mean) / std
    

# Full CNN-LSTM model for predicting (x, y) coordinates
class CNN_LSTM_ObjectLocation(nn.Module):
    def __init__(self, cnn_feature_dim=128, lstm_hidden_size=64, lstm_num_layers=2, output_dim=2,
                 input_height=24, input_width=32, conv_out_channels=32, is_input_norm=False, is_seq_reshape=False, CNNextractor_version=1, 
                 num_input_channel=1, bg_info_cost_ratio=0, bg_processing_type='one-proj', is_channel_normalization=False):
        super(CNN_LSTM_ObjectLocation, self).__init__()
        self.is_input_norm = is_input_norm
        if is_input_norm:
            #self.input_norm = nn.InstanceNorm2d(1)  # Normalize each sample independently on (C, H, W)
            if is_channel_normalization:
                self.input_norm = ChannelWiseNormalization() 
            else:
                self.input_norm = FullSampleNormalization()  # Normalizes entire sample
        self.CNNextractor_version = CNNextractor_version
        if self.CNNextractor_version == 1:  # single layer
            self.cnn = ParallelCNNFeatureExtractor(input_height=input_height, input_width=input_width,conv_out_channels=conv_out_channels,
                                        fc_out_features=cnn_feature_dim, num_input_channel=num_input_channel)  
        elif self.CNNextractor_version == 2:  # double layer
            self.cnn = ParallelCNNFeatureExtractor2(input_height=input_height, input_width=input_width,conv_out_channels=conv_out_channels,
                                        fc_out_features=cnn_feature_dim, num_input_channel=num_input_channel)  
        elif self.CNNextractor_version == 3:  # double layers + both flattern
            self.cnn = ParallelCNNFeatureExtractor3(input_height=input_height, input_width=input_width,conv_out_channels=conv_out_channels,
                                        fc_out_features=cnn_feature_dim, num_input_channel=num_input_channel)  
        elif self.CNNextractor_version == 4:  # triple layers
            self.cnn = ParallelCNNFeatureExtractor4(input_height=input_height, input_width=input_width,conv_out_channels=conv_out_channels,
                                        fc_out_features=cnn_feature_dim, num_input_channel=num_input_channel)  
        elif self.CNNextractor_version == 41:  # triple layers
            self.cnn = ParallelCNNFeatureExtractor41(input_height=input_height, input_width=input_width,conv_out_channels=conv_out_channels,
                                        fc_out_features=cnn_feature_dim, num_input_channel=num_input_channel)  
        elif self.CNNextractor_version == 42:  # triple layers
            self.cnn = ParallelCNNFeatureExtractor42(input_height=input_height, input_width=input_width,conv_out_channels=conv_out_channels,
                                        fc_out_features=cnn_feature_dim, num_input_channel=num_input_channel)     
        elif self.CNNextractor_version == 5:  # triple layers + 2 flatterned
            self.cnn = ParallelCNNFeatureExtractor5(input_height=input_height, input_width=input_width,conv_out_channels=conv_out_channels,
                                        fc_out_features=cnn_feature_dim, num_input_channel=num_input_channel)     
        elif self.CNNextractor_version == 6:  # triple layers + 2 flatterned
            self.cnn = ParallelCNNFeatureExtractor6(input_height=input_height, input_width=input_width,conv_out_channels=conv_out_channels,
                                        fc_out_features=cnn_feature_dim, num_input_channel=num_input_channel) 
        elif self.CNNextractor_version == 61:  # triple layers + 2 flatterned
            self.cnn = ParallelCNNFeatureExtractor61(input_height=input_height, input_width=input_width,conv_out_channels=conv_out_channels,
                                        fc_out_features=cnn_feature_dim, num_input_channel=num_input_channel) 
        elif self.CNNextractor_version == 62:  # triple layers + 2 flatterned
            self.cnn = ParallelCNNFeatureExtractor62(input_height=input_height, input_width=input_width,conv_out_channels=conv_out_channels,
                                        fc_out_features=cnn_feature_dim, num_input_channel=num_input_channel)   

        self.bg_processing_type = bg_processing_type
        self.bg_info_cost_ratio = bg_info_cost_ratio
        if self.bg_processing_type != 'lstm-proj':
            self.lstm = nn.LSTM(input_size=cnn_feature_dim, hidden_size=lstm_hidden_size, num_layers=lstm_num_layers, batch_first=True)
            self.lstm_norm = nn.LayerNorm(lstm_hidden_size)
            self.fc_o1 = nn.Linear(lstm_hidden_size, lstm_hidden_size)  # Output layer for (x, y) coordinates
            self.fc_o2 = nn.Linear(lstm_hidden_size, output_dim) 
        if self.bg_info_cost_ratio !=0:
            if self.bg_processing_type == 'one-proj':
                self.fc_b = nn.Linear(lstm_hidden_size, output_dim) 
            elif self.bg_processing_type == 'two-proj':
                self.fc_b1 = nn.Linear(lstm_hidden_size, lstm_hidden_size)
                self.fc_b2 = nn.Linear(lstm_hidden_size, output_dim) 
            elif self.bg_processing_type == 'lstm-proj':
                half_lstm_hidden_size = int(lstm_hidden_size*0.5)
                self.lstm = nn.LSTM(input_size=cnn_feature_dim, hidden_size=half_lstm_hidden_size, num_layers=lstm_num_layers, batch_first=True)
                self.lstm_norm = nn.LayerNorm(half_lstm_hidden_size)
                self.lstm_b = nn.LSTM(input_size=cnn_feature_dim, hidden_size=half_lstm_hidden_size, num_layers=lstm_num_layers, batch_first=True)
                self.lstm_norm_b = nn.LayerNorm(half_lstm_hidden_size)

                self.fc_o1 = nn.Linear(lstm_hidden_size, lstm_hidden_size)  # Output layer for (x, y) coordinates
                self.fc_o2 = nn.Linear(lstm_hidden_size, output_dim) 
                
                self.fc_b1 = nn.Linear(half_lstm_hidden_size, half_lstm_hidden_size)  # Output layer for (x, y) coordinates
                self.fc_b2 = nn.Linear(half_lstm_hidden_size, output_dim) 
            else:
                raise ValueError(f"Invalid bg_processing_type: {self.bg_processing_type}. Expected one of ['one-proj', 'two-proj', 'lstm-proj'].")

        self.is_seq_reshape = is_seq_reshape
        
    def forward(self, x):

        if self.is_input_norm:
            x = self.input_norm(x)  # Normalize entire sample
        batch_size, sequence_length, C, H, W = x.size()
        if self.is_seq_reshape:
            x = x.view(batch_size * sequence_length, C, H, W)  # Combine batch and sequence dimensions
            cnn_out = self.cnn(x)  # Process all frames at once
            cnn_features = cnn_out.view(batch_size, sequence_length, -1)  # Reshape back
        else:
            cnn_features = []
            for t in range(sequence_length):
                cnn_out = self.cnn(x[:, t, :, :, :])  # Shape: (batch_size, cnn_feature_dim)
                cnn_features.append(cnn_out)
            # Stack CNN outputs and pass through LSTM
            cnn_features = torch.stack(cnn_features, dim=1)
        
        lstm_out, _ = self.lstm(cnn_features)
        lstm_out = self.lstm_norm(lstm_out)
        if self.bg_processing_type != 'lstm-proj':
            lstm_out = torch.relu(self.fc_o1(lstm_out))  # (batch_size, sequence_length, output_dim)
            coord_predictions = self.fc_o2(lstm_out)
        if self.bg_info_cost_ratio !=0:
            if self.bg_processing_type == 'one-proj':
                bg_predictions = self.fc_b(lstm_out)
            elif self.bg_processing_type == 'two-proj':
                bg_predictions = torch.relu(self.fc_b1(lstm_out)) 
                bg_predictions = self.fc_b2(bg_predictions)
            elif self.bg_processing_type == 'lstm-proj':
                bg_lstem_out, _ = self.lstm_b(cnn_features)
                bg_lstem_out = self.lstm_norm_b(bg_lstem_out)
                # combine both 
                lstm_out = torch.cat((lstm_out, bg_lstem_out), dim=-1)
                lstm_out = torch.relu(self.fc_o1(lstm_out))  # (batch_size, sequence_length, output_dim)
                coord_predictions = self.fc_o2(lstm_out)
                bg_predictions = torch.relu(self.fc_b1(bg_lstem_out)) 
                bg_predictions = self.fc_b2(bg_predictions)
        else:
            bg_predictions = coord_predictions
        return coord_predictions, bg_predictions

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.LSTM)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    

