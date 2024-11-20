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
        
        self.conv2 = nn.Conv2d(1, conv_out_channels, kernel_size=16, dilation=4, stride=8, padding=4)
        self.bn2 = nn.BatchNorm2d(conv_out_channels)
        
        self.conv3 = nn.Conv2d(1, conv_out_channels, kernel_size=32, dilation=8, stride=16, padding=8)
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

    

# Full CNN-LSTM model for predicting (x, y) coordinates
class CNN_LSTM_ObjectLocation(nn.Module):
    def __init__(self, cnn_feature_dim=128, lstm_hidden_size=64, lstm_num_layers=2, output_dim=2,
                 input_height=24, input_width=32, conv1_out_channels=16, conv2_out_channels=32, fc_out_features=128):
        super(CNN_LSTM_ObjectLocation, self).__init__()
        self.cnn = ParallelCNNFeatureExtractor(input_height=input_height, input_width=input_width,conv_out_channels=conv1_out_channels,
                                       fc_out_features=fc_out_features)  # Assume CNNFeatureExtractor outputs cnn_feature_dim
        self.lstm = nn.LSTM(input_size=cnn_feature_dim, hidden_size=lstm_hidden_size, num_layers=lstm_num_layers, batch_first=True)
        self.lstm_norm = nn.LayerNorm(lstm_hidden_size)
        self.fc1 = nn.Linear(lstm_hidden_size, lstm_hidden_size)  # Output layer for (x, y) coordinates
        self.fc2 = nn.Linear(lstm_hidden_size, output_dim) 

    def forward(self, x):
        batch_size, sequence_length, C, H, W = x.size()
        cnn_features = []

        # Pass each frame through CNN
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
    

