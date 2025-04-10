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
    

class RGC_CNN_LSTM_ObjectLocation(nn.Module):
    def __init__(self, 
                 cnn_feature_dim=128, 
                 lstm_hidden_size=64, 
                 lstm_num_layers=2, 
                 output_dim=2,
                 input_height=24, 
                 input_width=32, 
                 input_depth=10,  # added: expected depth (D) of input for RGC module.
                 conv_out_channels=32, 
                 is_input_norm=False, 
                 is_seq_reshape=False, 
                 CNNextractor_version=1, 
                 is_channel_normalization=False):
        """
        Parameters:
          cnn_feature_dim: Dimension of features extracted by CNN.
          lstm_hidden_size: Hidden size for the LSTM.
          lstm_num_layers: Number of layers in the LSTM.
          output_dim: Dimension of the final output (e.g., 2 for (x,y)).
          input_height: Height of each input image.
          input_width: Width of each input image.
          input_depth: Depth (or number of frames) provided to the RGC module.
          conv_out_channels: Parameter for the CNN extractors.
          is_input_norm: Whether input normalization is applied.
          is_seq_reshape: Whether to reshape all frames together (for efficiency).
          CNNextractor_version: Selects which CNN feature extractor to use.
          is_channel_normalization: If true, use channel-wise normalization; otherwise use full sample norm.
        """
        super(RGC_CNN_LSTM_ObjectLocation, self).__init__()
        self.is_input_norm = is_input_norm
        if is_input_norm:
            if is_channel_normalization:
                self.input_norm = ChannelWiseNormalization()
            else:
                self.input_norm = FullSampleNormalization()
                
        # Initialize the RGC module (assumed imported from a separate file)
        # Using the parameters provided.
        self.rgc = RGC_Module(
            temporal_filters=3,
            in_channels=1,  # assumes input channel = 1 (adjust if needed)
            num_filters1=16,
            kernel_size1=3,
            stride1=2,
            pool_size=2,
            num_filters2=2,   # the expected output channels from RGC_module
            kernel_size2=3,
            stride2=2,
            input_shape=(input_height, input_width, input_depth),
            num_classes=10,
            dilation1=1,
            dilation2=2
        )
        # Set the new number of input channels for the CNN extractor according to the RGC module's output.
        # Here, we assume that the final output channel from RGC_module equals num_filters2.
        new_num_input_channel = 2
        
        self.CNNextractor_version = CNNextractor_version
        # Create the CNN feature extractor based on selected version.
        if self.CNNextractor_version == 1:
            self.cnn = ParallelCNNFeatureExtractor(
                input_height=input_height, 
                input_width=input_width,
                conv_out_channels=conv_out_channels,
                fc_out_features=cnn_feature_dim, 
                num_input_channel=new_num_input_channel
            )
        elif self.CNNextractor_version == 2:
            self.cnn = ParallelCNNFeatureExtractor2(
                input_height=input_height, 
                input_width=input_width,
                conv_out_channels=conv_out_channels,
                fc_out_features=cnn_feature_dim, 
                num_input_channel=new_num_input_channel
            )
        elif self.CNNextractor_version == 3:
            self.cnn = ParallelCNNFeatureExtractor3(
                input_height=input_height, 
                input_width=input_width,
                conv_out_channels=conv_out_channels,
                fc_out_features=cnn_feature_dim, 
                num_input_channel=new_num_input_channel
            )
        elif self.CNNextractor_version == 4:
            self.cnn = ParallelCNNFeatureExtractor4(
                input_height=input_height, 
                input_width=input_width,
                conv_out_channels=conv_out_channels,
                fc_out_features=cnn_feature_dim, 
                num_input_channel=new_num_input_channel
            )
        elif self.CNNextractor_version == 41:
            self.cnn = ParallelCNNFeatureExtractor41(
                input_height=input_height, 
                input_width=input_width,
                conv_out_channels=conv_out_channels,
                fc_out_features=cnn_feature_dim, 
                num_input_channel=new_num_input_channel
            )
        elif self.CNNextractor_version == 42:
            self.cnn = ParallelCNNFeatureExtractor42(
                input_height=input_height, 
                input_width=input_width,
                conv_out_channels=conv_out_channels,
                fc_out_features=cnn_feature_dim, 
                num_input_channel=new_num_input_channel
            )
        elif self.CNNextractor_version == 5:
            self.cnn = ParallelCNNFeatureExtractor5(
                input_height=input_height, 
                input_width=input_width,
                conv_out_channels=conv_out_channels,
                fc_out_features=cnn_feature_dim, 
                num_input_channel=new_num_input_channel
            )
        elif self.CNNextractor_version == 6:
            self.cnn = ParallelCNNFeatureExtractor6(
                input_height=input_height, 
                input_width=input_width,
                conv_out_channels=conv_out_channels,
                fc_out_features=cnn_feature_dim, 
                num_input_channel=new_num_input_channel
            )
        elif self.CNNextractor_version == 61:
            self.cnn = ParallelCNNFeatureExtractor61(
                input_height=input_height, 
                input_width=input_width,
                conv_out_channels=conv_out_channels,
                fc_out_features=cnn_feature_dim, 
                num_input_channel=new_num_input_channel
            )
        elif self.CNNextractor_version == 62:
            self.cnn = ParallelCNNFeatureExtractor62(
                input_height=input_height, 
                input_width=input_width,
                conv_out_channels=conv_out_channels,
                fc_out_features=cnn_feature_dim, 
                num_input_channel=new_num_input_channel
            )
        else:
            raise ValueError(f"Invalid CNNextractor_version: {self.CNNextractor_version}")

        # The rest of the architecture remains the same as in the original model.
        self.lstm = nn.LSTM(input_size=cnn_feature_dim, hidden_size=lstm_hidden_size, 
                            num_layers=lstm_num_layers, batch_first=True)
        self.lstm_norm = nn.LayerNorm(lstm_hidden_size)
        self.fc_o1 = nn.Linear(lstm_hidden_size, lstm_hidden_size)
        self.fc_o2 = nn.Linear(lstm_hidden_size, output_dim)
        
        self.is_seq_reshape = is_seq_reshape

    def forward(self, x):
        """
        x: Input tensor of shape (batch_size, D, C, H, W) where D is the depth (or original sequence length).
        """
        if self.is_input_norm:
            x = self.input_norm(x)

        batch_size, D, C, H, W = x.size()

        # --- Pass the input through the RGC module ---
        # The RGC module is expected to output a tensor of shape (batch_size, T, C_out, H_out, W_out),
        # where T is the new, shorter temporal dimension (T < D) and C_out is determined by RGC_module parameters.
        rgc_out = self.rgc(x)
        # Retrieve new temporal length (for example, T) from the output:
        T = rgc_out.size(1)

        # --- Process each “time slice” of the RGC output using the CNN extractor ---
        if self.is_seq_reshape:
            # Process all frames at once by merging batch and time dimensions.
            cnn_input = rgc_out.view(batch_size * T, rgc_out.size(2), rgc_out.size(3), rgc_out.size(4))
            cnn_out = self.cnn(cnn_input)
            cnn_features = cnn_out.view(batch_size, T, -1)
        else:
            cnn_features = []
            for t in range(T):
                # Note: using only past (or present) data — the sliding window moves forward.
                cnn_out = self.cnn(rgc_out[:, t, :, :, :])
                cnn_features.append(cnn_out)
            cnn_features = torch.stack(cnn_features, dim=1)

        # --- LSTM processing ---
        lstm_out, _ = self.lstm(cnn_features)
        lstm_out = self.lstm_norm(lstm_out)
        lstm_out = torch.relu(self.fc_o1(lstm_out))
        coord_predictions = self.fc_o2(lstm_out)
        # Set background predictions to be the same as coordinate predictions.
        bg_predictions = coord_predictions

        return coord_predictions, bg_predictions

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.LSTM)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class TwoLayerCNN(nn.Module):
    def __init__(self,
                 temporal_filters: int,
                 in_channels: int,
                 num_filters1: int,
                 kernel_size1: int,
                 stride1: int,
                 pool_size: int,
                 num_filters2: int,
                 kernel_size2: int,
                 stride2: int,
                 input_shape: tuple,  # (height, width, time)
                 num_classes: int = 10,
                 dilation1: int = 1,  # New dilation parameter for conv1
                 dilation2: int = 1   # New dilation parameter for conv2
                ):
        """
        Parameters:
          in_channels: Number of channels in the input image (e.g., 1 for grayscale, 3 for RGB).
          num_filters1: Number of output filters for the first conv layer.
          kernel_size1: Kernel size for the first conv layer (assumed square).
          stride1: Stride for the first conv layer.
          pool_size: Pooling window size after the first conv layer.
          num_filters2: Number of output filters for the second conv layer.
          kernel_size2: Kernel size for the second conv layer (assumed square).
          stride2: Stride for the second conv layer.
          input_shape: Tuple (height, width, time) of the input.
          num_classes: Number of output classes.
          dilation1: Dilation factor for conv1.
          dilation2: Dilation factor for conv2.

        Note:
          When dilation > 1, the effective kernel size becomes:
              effective_size = dilation * (kernel_size - 1) + 1.
          To maintain “same” output size (for odd effective kernel size), the symmetric
          padding should be set to:
              padding = (effective_size - 1) // 2 = dilation*(kernel_size - 1) // 2.
        """
        super(TwoLayerCNN, self).__init__()

        # Save input dimensions.
        H, W, D = input_shape

        # Save conv1 parameters for later use (and update for dilation).
        self.kernel_size1 = kernel_size1
        self.stride1 = stride1
        self.dilation1 = dilation1
        # Update symmetric padding for conv1 using the effective kernel size.
        self.padding1 = dilation1 * (kernel_size1 - 1) // 2

        # Define the temporal convolution (unchanged):
        self.conv_temporal = nn.Conv3d(in_channels=1,
                               out_channels=temporal_filters,
                               kernel_size=(1, 1, D),
                               stride=1,
                               padding=0)

        # Define the first spatial conv layer with dilation.
        self.conv1 = nn.Conv2d(in_channels=temporal_filters,
                               out_channels=num_filters1,
                               kernel_size=kernel_size1,
                               stride=stride1,
                               padding=self.padding1,
                               dilation=dilation1)
        self.pool = nn.MaxPool2d(kernel_size=pool_size)
        self.pool_size = pool_size  # save pool window size

        # Save conv2 parameters.
        self.kernel_size2 = kernel_size2
        self.stride2 = stride2
        self.dilation2 = dilation2
        self.padding2 = dilation2 * (kernel_size2 - 1) // 2

        # Define second conv layer (with no pooling afterward) with dilation.
        self.conv2 = nn.Conv2d(num_filters1, num_filters2,
                               kernel_size=kernel_size2,
                               stride=stride2,
                               padding=self.padding2,
                               dilation=dilation2)

        # Automatically compute the flattened size from a dummy input.
        self.flattened_size = self._get_conv_output((in_channels, *input_shape))
        print("Automatically computed flattened size:", self.flattened_size)

        # Define a fully connected layer for classification.
        self.fc = nn.Linear(self.flattened_size, num_classes)

    def _get_conv_output(self, shape):
        """
        Passes a dummy input through conv_temporal, conv1, pooling, and conv2 to compute the total
        number of features to be flattened.

        Parameters:
          shape: Tuple (in_channels, height, width, time)

        Returns:
          int: Flattened feature size.
        """
        with torch.no_grad():
            # shape: [1, in_channels, H, W, D]
            x = torch.zeros(1, *shape)
            x = F.relu(self.conv_temporal(x)).squeeze(-1)
            print("Dummy input shape after conv_temporal:", x.shape)
            x = F.relu(self.conv1(x))
            x = self.pool(x)
            x = F.relu(self.conv2(x))
            return int(x.numel())

    def forward(self, x):
        x = F.relu(self.conv_temporal(x))
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def visualize_kernel_centers(self, input_image, show_conv1=True, show_conv2=True):
        """
        Overlays the centers (receptive field center positions) of the convolutional kernels on the input image.
        The computation has been updated to account for dilation.

        Parameters:
          input_image: A NumPy array representing the image.
                       (If 2D, treated as grayscale; if 3D, assumed RGB.)
          show_conv1: Boolean. If True, overlay grid centers for conv1.
          show_conv2: Boolean. If True, overlay grid centers for conv2 (after pooling).
        """
        # If the input image has a third dimension (e.g., D), take the first slice.
        input_image = input_image[:, :, 0].squeeze()

        # Determine input dimensions (height, width).
        if input_image.ndim == 3:
            H, W, _ = input_image.shape
        else:
            H, W = input_image.shape

        plt.figure(figsize=(6,6))
        if input_image.ndim == 3:
            plt.imshow(input_image.astype(np.uint8))
        else:
            plt.imshow(input_image, cmap='gray')

        # ---------------------------------------------------
        # Compute Conv1 Output Dimensions (after conv1 but before pooling)
        # Using the formula for convolution output dimensions with dilation:
        #   output_dim = floor((input_dim + 2*padding - (dilation*(kernel_size - 1) + 1)) / stride) + 1
        out_h1 = ((H + 2 * self.padding1 - (self.dilation1*(self.kernel_size1 - 1) + 1)) // self.stride1) + 1
        out_w1 = ((W + 2 * self.padding1 - (self.dilation1*(self.kernel_size1 - 1) + 1)) // self.stride1) + 1

        if show_conv1:
            centers1 = []
            # For each output pixel in conv1's feature map, compute the corresponding input image coordinate.
            # The receptive field center is given by:
            #   center = index * stride - padding + (dilation*(kernel_size - 1))/2
            for j in range(out_h1):
                for i in range(out_w1):
                    cx = i * self.stride1 - self.padding1 + (self.dilation1 * (self.kernel_size1 - 1)) / 2.
                    cy = j * self.stride1 - self.padding1 + (self.dilation1 * (self.kernel_size1 - 1)) / 2.
                    centers1.append((cx, cy))
            centers1 = np.array(centers1)
            plt.scatter(centers1[:, 0], centers1[:, 1],
                        marker='o', s=20, label='Conv1 centers', color='red')

        # ---------------------------------------------------
        # Conv2 Centers (after pooling)
        if show_conv2:
            # First, compute the pooled dimensions.
            pool_h = out_h1 // self.pool_size
            pool_w = out_w1 // self.pool_size

            # Compute Conv2 output dimensions in the pooled space.
            out_h2 = ((pool_h + 2 * self.padding2 - (self.dilation2*(self.kernel_size2 - 1) + 1)) // self.stride2) + 1
            out_w2 = ((pool_w + 2 * self.padding2 - (self.dilation2*(self.kernel_size2 - 1) + 1)) // self.stride2) + 1

            centers2 = []
            for j in range(out_h2):
                for i in range(out_w2):
                    # Compute center in conv2's output (in pooled space).
                    cx_pool = i * self.stride2 - self.padding2 + (self.dilation2*(self.kernel_size2 - 1)) / 2.
                    cy_pool = j * self.stride2 - self.padding2 + (self.dilation2*(self.kernel_size2 - 1)) / 2.

                    # Map from the pooled coordinates back to the conv1 output coordinates.
                    # Each pooled cell spans a region in conv1 output of size self.pool_size.
                    cx_conv1 = cx_pool * self.pool_size + (self.pool_size - 1) / 2.
                    cy_conv1 = cy_pool * self.pool_size + (self.pool_size - 1) / 2.

                    # Map from conv1 output coordinates to input image coordinates.
                    cx = cx_conv1 * self.stride1 - self.padding1 + (self.dilation1 * (self.kernel_size1 - 1)) / 2.
                    cy = cy_conv1 * self.stride1 - self.padding1 + (self.dilation1 * (self.kernel_size1 - 1)) / 2.

                    centers2.append((cx, cy))
            centers2 = np.array(centers2)
            plt.scatter(centers2[:, 0], centers2[:, 1],
                        marker='x', s=20, label='Conv2 centers', color='blue')

        plt.title("Receptive Field Center Grids")
        plt.xlabel("Width")
        plt.ylabel("Height")
        plt.legend()
        plt.gca().invert_yaxis()  # Invert y-axis to match image coordinate system
        plt.show()

    def visualize_receptive_field_boxes(self, input_image, sample_rate=1, show_conv1=True, show_conv2=True):
        """
        Draws the receptive field rectangles on the input image,
        illustrating how each convolution layer covers the image.

        Parameters:
          input_image: NumPy array representing the image (if 3D, only the first channel is used).
          sample_rate: Determines the sampling of grid cells to draw (to reduce clutter).
          show_conv1: If True, draw receptive fields for conv1.
          show_conv2: If True, draw receptive fields for conv2 (after pooling).

        The effective receptive field size for a convolution with dilation is given by:
            effective_size = dilation * (kernel_size - 1) + 1.
        For conv2, which follows a pooling layer, the overall receptive field (in input image coordinates) is computed as:
            RF_conv2 = RF_conv1 + (pool_size - 1) * stride1 + (dilation2 * (kernel_size2 - 1)) * (stride1 * pool_size),
        where RF_conv1 = dilation1*(kernel_size1-1) + 1.
        """
        # Use only the first channel.
        img = input_image[:, :, 0].squeeze() if input_image.ndim >= 3 else input_image
        if img.ndim == 3:
            H, W, _ = img.shape
        else:
            H, W = img.shape

        # Start plotting the image.
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        if img.ndim == 3:
            ax.imshow(img.astype(np.uint8))
        else:
            ax.imshow(img, cmap='gray')

        # --- Conv1 receptive field parameters ---
        # Effective receptive field (RF) size for conv1.
        eff_rf_conv1 = self.dilation1 * (self.kernel_size1 - 1) + 1

        # Compute conv1 output dimensions.
        out_h1 = ((H + 2 * self.padding1 - (self.dilation1*(self.kernel_size1 - 1) + 1)) // self.stride1) + 1
        out_w1 = ((W + 2 * self.padding1 - (self.dilation1*(self.kernel_size1 - 1) + 1)) // self.stride1) + 1

        if show_conv1:
            # Loop over conv1 grid (with sampling to avoid clutter).
            for j in range(0, out_h1, sample_rate):
                for i in range(0, out_w1, sample_rate):
                    # Calculate the center position in input coordinates.
                    cx = i * self.stride1 - self.padding1 + (self.dilation1 * (self.kernel_size1 - 1)) / 2.
                    cy = j * self.stride1 - self.padding1 + (self.dilation1 * (self.kernel_size1 - 1)) / 2.
                    # Determine top-left coordinate for the receptive field box.
                    top_left_x = cx - eff_rf_conv1 / 2.
                    top_left_y = cy - eff_rf_conv1 / 2.
                    # Create and add the rectangle patch.
                    rect = patches.Rectangle((top_left_x, top_left_y), eff_rf_conv1, eff_rf_conv1,
                                             linewidth=1, edgecolor='red', facecolor='none')
                    ax.add_patch(rect)

        # --- Conv2 receptive field parameters ---
        # For conv2, we first compute the receptive field parameters in the pooled space.
        # Compute pooled dimensions (from conv1 output).
        pool_h = out_h1 // self.pool_size
        pool_w = out_w1 // self.pool_size

        # Effective receptive field of conv2 in the pooled space.
        eff_rf_conv2_pool = self.dilation2 * (self.kernel_size2 - 1) + 1

        # Map conv2 receptive field back to input coordinates.
        # First, compute conv1 effective RF (already calculated as eff_rf_conv1).
        # Then, the overall conv2 RF in input is:
        eff_rf_conv2 = eff_rf_conv1 + (self.pool_size - 1) * self.stride1 + \
                       self.dilation2 * (self.kernel_size2 - 1) * (self.stride1 * self.pool_size)

        if show_conv2:
            # Compute conv2 output dimensions.
            out_h2 = ((pool_h + 2 * self.padding2 - (self.dilation2*(self.kernel_size2 - 1) + 1)) // self.stride2) + 1
            out_w2 = ((pool_w + 2 * self.padding2 - (self.dilation2*(self.kernel_size2 - 1) + 1)) // self.stride2) + 1

            for j in range(0, out_h2, sample_rate):
                for i in range(0, out_w2, sample_rate):
                    # Calculate center in conv2's (pooled) output space.
                    cx_pool = i * self.stride2 - self.padding2 + (self.dilation2*(self.kernel_size2 - 1)) / 2.
                    cy_pool = j * self.stride2 - self.padding2 + (self.dilation2*(self.kernel_size2 - 1)) / 2.
                    # Map from pooled space to conv1 output coordinates.
                    cx_conv1 = cx_pool * self.pool_size + (self.pool_size - 1) / 2.
                    cy_conv1 = cy_pool * self.pool_size + (self.pool_size - 1) / 2.
                    # Map from conv1 output coordinates to input image coordinates.
                    cx_input = cx_conv1 * self.stride1 - self.padding1 + (self.dilation1*(self.kernel_size1 - 1)) / 2.
                    cy_input = cy_conv1 * self.stride1 - self.padding1 + (self.dilation1*(self.kernel_size1 - 1)) / 2.
                    # Determine the top-left coordinate for the conv2 receptive field box.
                    top_left_x = cx_input - eff_rf_conv2 / 2.
                    top_left_y = cy_input - eff_rf_conv2 / 2.
                    # Create and add the rectangle patch.
                    rect = patches.Rectangle((top_left_x, top_left_y), eff_rf_conv2, eff_rf_conv2,
                                             linewidth=1, edgecolor='blue', facecolor='none')
                    ax.add_patch(rect)

        ax.set_title("Receptive Field Boxes over the Input Image")
        ax.set_xlabel("Width")
        ax.set_ylabel("Height")
        ax.invert_yaxis()  # Invert y-axis to match image coordinates.
        plt.show()