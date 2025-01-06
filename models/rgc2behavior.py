import torch
import torch.nn as nn
import torch.nn.functional as F


import torch
import torch.nn as nn

class ParallelCNNFeatureExtractor(nn.Module):
    def __init__(self, input_height=24, input_width=32, conv_out_channels=16, fc_out_features=128):
        super(ParallelCNNFeatureExtractor, self).__init__()
        
        # Define parallel convolution layers with different kernel sizes
        self.conv1 = nn.Conv2d(1, conv_out_channels, kernel_size=4, stride=2, padding=0)
        self.bn1 = nn.BatchNorm2d(conv_out_channels)
        
        self.conv2 = nn.Conv2d(1, conv_out_channels, kernel_size=4, dilation=4, stride=8, padding=4)
        self.bn2 = nn.BatchNorm2d(conv_out_channels)
        
        self.conv3 = nn.Conv2d(1, conv_out_channels, kernel_size=4, dilation=8, stride=16, padding=8)
        self.bn3 = nn.BatchNorm2d(conv_out_channels)
        
        # Pooling layer to reduce spatial dimensions
        self.pool = nn.MaxPool2d(2, 2)
        
        # Determine the flattened feature size after the convolution and pooling layers
        with torch.no_grad():
            mock_input = torch.zeros(1, 1, input_height, input_width)
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
    def __init__(self, input_height=24, input_width=32, conv_out_channels=16, fc_out_features=128):
        super(ParallelCNNFeatureExtractor2, self).__init__()
        
        # Define parallel convolution layers with different kernel sizes (Series A)
        conv_out_channels_a = int(conv_out_channels*0.5)
        self.conv1_a = nn.Conv2d(1, conv_out_channels_a, kernel_size=4, stride=2, padding=0)
        self.bn1_a = nn.BatchNorm2d(conv_out_channels_a)
        
        self.conv2_a = nn.Conv2d(1, conv_out_channels_a, kernel_size=4, dilation=4, stride=8, padding=4)
        self.bn2_a = nn.BatchNorm2d(conv_out_channels_a)
        
        self.conv3_a = nn.Conv2d(1, conv_out_channels_a, kernel_size=4, dilation=8, stride=16, padding=8)
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
            mock_input = torch.zeros(1, 1, input_height, input_width)
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
    def __init__(self, input_height=24, input_width=32, conv_out_channels=16, fc_out_features=128):
        super(ParallelCNNFeatureExtractor2, self).__init__()
        
        # Define parallel convolution layers with different kernel sizes (Series A)
        conv_out_channels_a = int(conv_out_channels*0.5)
        self.conv1_a = nn.Conv2d(1, conv_out_channels_a, kernel_size=4, stride=2, padding=0)
        self.bn1_a = nn.BatchNorm2d(conv_out_channels_a)
        
        self.conv2_a = nn.Conv2d(1, conv_out_channels_a, kernel_size=4, dilation=4, stride=8, padding=4)
        self.bn2_a = nn.BatchNorm2d(conv_out_channels_a)
        
        self.conv3_a = nn.Conv2d(1, conv_out_channels_a, kernel_size=4, dilation=8, stride=16, padding=8)
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
            mock_input = torch.zeros(1, 1, input_height, input_width)
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
        x2_a_flat = x1_a.view(x1_a.size(0), -1)
        x3_a_flat = x1_a.view(x1_a.size(0), -1)
        
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


class FullSampleNormalization(nn.Module):
    def forward(self, x):
        # Compute mean and std over (sequence, channels, height, width) for each sample
        mean = x.mean(dim=(1, 2, 3, 4), keepdim=True)  # Mean over sequence, C, H, W
        std = x.std(dim=(1, 2, 3, 4), keepdim=True) + 1e-6  # Std over sequence, C, H, W
        return (x - mean) / std   

# Full CNN-LSTM model for predicting (x, y) coordinates
class CNN_LSTM_ObjectLocation(nn.Module):
    def __init__(self, cnn_feature_dim=128, lstm_hidden_size=64, lstm_num_layers=2, output_dim=2,
                 input_height=24, input_width=32, conv_out_channels=32, is_input_norm=False, is_seq_reshape=False, CNNextractor_version=1):
        super(CNN_LSTM_ObjectLocation, self).__init__()
        self.is_input_norm = is_input_norm
        if is_input_norm:
            #self.input_norm = nn.InstanceNorm2d(1)  # Normalize each sample independently on (C, H, W)
            self.input_norm = FullSampleNormalization()  # Normalizes entire sample
        self.CNNextractor_version = CNNextractor_version
        if self.CNNextractor_version == 1:
            self.cnn = ParallelCNNFeatureExtractor(input_height=input_height, input_width=input_width,conv_out_channels=conv_out_channels,
                                        fc_out_features=cnn_feature_dim)  # Assume CNNFeatureExtractor outputs cnn_feature_dim
        elif self.CNNextractor_version == 2:
            self.cnn = ParallelCNNFeatureExtractor2(input_height=input_height, input_width=input_width,conv_out_channels=conv_out_channels,
                                        fc_out_features=cnn_feature_dim)  
        elif self.CNNextractor_version == 3:
            self.cnn = ParallelCNNFeatureExtractor3(input_height=input_height, input_width=input_width,conv_out_channels=conv_out_channels,
                                        fc_out_features=cnn_feature_dim)  
        self.lstm = nn.LSTM(input_size=cnn_feature_dim, hidden_size=lstm_hidden_size, num_layers=lstm_num_layers, batch_first=True)
        self.lstm_norm = nn.LayerNorm(lstm_hidden_size)
        self.fc1 = nn.Linear(lstm_hidden_size, lstm_hidden_size)  # Output layer for (x, y) coordinates
        self.fc2 = nn.Linear(lstm_hidden_size, output_dim) 
        self.is_seq_reshape = is_seq_reshape

    def forward(self, x):

        if self.is_input_norm:
            x = self.input_norm(x)  # Normalize entire sample
        # print(f'x norm min {torch.min(x)}')
        # print(f'x norm max {torch.max(x)}')
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
        lstm_out = torch.relu(self.fc1(lstm_out))  # (batch_size, sequence_length, output_dim)
        coord_predictions = self.fc2(lstm_out)
        return coord_predictions

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.LSTM)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    

