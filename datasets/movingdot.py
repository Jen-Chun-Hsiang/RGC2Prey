import torch
import os
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


class RandomMovingSpotDataset(Dataset):
    def __init__(self, sequence_length=20, grid_height=24, grid_width=32, prob_vis=0.5, num_samples=10000):
        """
        Initialize the dataset with configuration for sequence generation.
        
        Args:
            sequence_length (int): Number of frames in each sequence.
            grid_height (int): Height of the 2D grid.
            grid_width (int): Width of the 2D grid.
            prob_vis (float): Probability of the spot being visible in each frame.
            num_samples (int): Total number of samples in the dataset.
        """
        self.sequence_length = sequence_length
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.prob_vis = prob_vis
        self.num_samples = num_samples

    def __len__(self):
        # Return the total number of samples
        return self.num_samples

    def generate_sequence(self):
        """Generate a single sequence with random spot movements and visibility."""
        sequence = []
        coords = []
        visibility = []

        # Initial random position for the spot
        x, y = np.random.randint(0, self.grid_height), np.random.randint(0, self.grid_width)

        for _ in range(self.sequence_length):
            # Determine visibility
            visible = np.random.rand() < self.prob_vis
            frame = np.zeros((self.grid_height, self.grid_width), dtype=np.float32)

            if visible:
                frame[x, y] = 1.0  # Place the spot at (x, y)
                visibility.append(1)  # Spot is visible
            else:
                visibility.append(0)  # Spot is invisible
            
            # Append the frame and coordinates
            sequence.append(frame)
            coords.append([x, y])

            # Randomly move the spot by 1 or 2 pixels, ensuring it stays within the grid
            dx, dy = np.random.choice([-2, -1, 1, 2]), np.random.choice([-2, -1, 1, 2])
            x = np.clip(x + dx, 0, self.grid_height - 1)
            y = np.clip(y + dy, 0, self.grid_width - 1)

        return np.array(sequence), np.array(coords), np.array(visibility)

    def __getitem__(self, idx):
        """Get a single item from the dataset (randomly generated sequence)."""
        sequence, coords, visibility = self.generate_sequence()
        
        # Convert to PyTorch tensors and add channel dimension to sequence
        sequence_tensor = torch.tensor(sequence).unsqueeze(1)  # Shape: (1, grid_height, grid_width)
        coords_tensor = torch.tensor(coords, dtype=torch.float32)  # Shape: (sequence_length, 2)
        visible_tensor = torch.tensor(visibility, dtype=torch.int8)  # Shape: (sequence_length, 2)

        return sequence_tensor, coords_tensor, visible_tensor


class CNNFeatureExtractor(nn.Module):
    def __init__(self, input_height=24, input_width=32, conv1_out_channels=16, conv2_out_channels=32, fc_out_features=128):
        super(CNNFeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(1, conv1_out_channels, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(conv1_out_channels, conv2_out_channels, kernel_size=3, stride=1, padding=1)

        # Mock pass to determine the flattened feature size for the fully connected layer
        with torch.no_grad():
            mock_input = torch.zeros(1, 1, input_height, input_width)
            mock_output = self._forward_conv_layers(mock_input)
            self.flat_feature_size = mock_output.view(1, -1).size(1)

        # Define the fully connected layer based on the computed flattened feature size
        self.fc = nn.Linear(self.flat_feature_size, fc_out_features)

    def _forward_conv_layers(self, x):
        """Forward pass through the convolutional and pooling layers only."""
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        return x

    def forward(self, x):
        x = self._forward_conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)  # Project to feature vector of specified size
        return x
    

class ParallelCNNFeatureExtractor(nn.Module):
    def __init__(self, input_height=24, input_width=32, conv_out_channels=16, fc_out_features=128):
        super(ParallelCNNFeatureExtractor, self).__init__()
        
        # Define parallel convolution layers with different kernel sizes
        self.conv1x1 = nn.Conv2d(1, conv_out_channels, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(1, conv_out_channels, kernel_size=3, stride=1, padding=1)
        self.conv5x5 = nn.Conv2d(1, conv_out_channels, kernel_size=5, stride=1, padding=2)
        
        # Pooling layer to reduce spatial dimensions
        self.pool = nn.MaxPool2d(2, 2)
        
        # Determine the flattened feature size after the convolution and pooling layers
        with torch.no_grad():
            mock_input = torch.zeros(1, 1, input_height, input_width)
            mock_output = self._forward_conv_layers(mock_input)
            self.flat_feature_size = mock_output.view(1, -1).size(1)

        # Fully connected layer based on the flattened feature size
        self.fc = nn.Linear(self.flat_feature_size, fc_out_features)

    def _forward_conv_layers(self, x):
        """Forward pass through each convolutional layer and concatenate their outputs."""
        x1 = self.pool(torch.relu(self.conv1x1(x)))
        x2 = self.pool(torch.relu(self.conv3x3(x)))
        x3 = self.pool(torch.relu(self.conv5x5(x)))
        
        # Concatenate along the channel dimension
        x = torch.cat((x1, x2, x3), dim=1)
        return x

    def forward(self, x):
        x = self._forward_conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)  # Project to feature vector of specified size
        return x
    

# Full CNN-LSTM model for predicting (x, y) coordinates
class CNN_LSTM_ObjectLocation(nn.Module):
    def __init__(self, cnn_feature_dim=128, lstm_hidden_size=64, lstm_num_layers=2, output_dim=2,
                 input_height=24, input_width=32, conv1_out_channels=16, conv2_out_channels=32, fc_out_features=128):
        super(CNN_LSTM_ObjectLocation, self).__init__()
        # self.cnn = CNNFeatureExtractor(input_height=input_height, input_width=input_width, 
        #                                conv1_out_channels=conv1_out_channels, conv2_out_channels=conv2_out_channels,
        #                                fc_out_features=fc_out_features)  # Assume CNNFeatureExtractor outputs cnn_feature_dim
        self.cnn = CNNFeatureExtractor(input_height=input_height, input_width=input_width,conv_out_channels=conv1_out_channels,
                                       fc_out_features=fc_out_features)  # Assume CNNFeatureExtractor outputs cnn_feature_dim
        self.lstm = nn.LSTM(input_size=cnn_feature_dim, hidden_size=lstm_hidden_size, num_layers=lstm_num_layers, batch_first=True)
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
        
        # Predict coordinates
        lstm_out = torch.relu(self.fc1(lstm_out))  # (batch_size, sequence_length, output_dim)
        coord_predictions = self.fc2(lstm_out)
        return coord_predictions
    

# Define function to plot and save results
def plot_and_save_results(epoch_losses, model, dataloader, sequence_length, save_dir, file_name):
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Plot the loss over epochs
    plt.figure(figsize=(12, 12))
    
    # Loss plot
    plt.subplot(2, 2, 1)
    plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss per Epoch")
    plt.legend()
    
    # Sample batch for target and predictions trace
    sequences, targets, visible = next(iter(dataloader))
    outputs = model(sequences).detach()
    num_batches = targets.shape[0]
    batch_idx = np.random.choice(num_batches)

    # Helper function to plot traces
    def plot_trace(x_vals, y_vals, visible, label, line_style, color_visible, color_hidden):
        for i in range(sequence_length - 1):
            color = color_visible if visible[batch_idx, i + 1] == 1 else color_hidden
            plt.plot([x_vals[i], x_vals[i + 1]], [y_vals[i], y_vals[i + 1]],
                     color=color, linestyle=line_style, linewidth=1.5,
                     label=label if i == 0 else "")
    # Trace plot
    plt.subplot(2, 2, 2)
    plot_trace(targets[batch_idx, :, 0], targets[batch_idx, :, 1], visible, 'Target', '-', 'darkblue', 'lightblue')
    plot_trace(outputs[batch_idx, :, 0], outputs[batch_idx, :, 1], visible, 'Prediction', '--', 'maroon', 'lightcoral')
    plt.xlabel("X-coordinate")
    plt.ylabel("Y-coordinate")
    plt.title("Trace of Target and Model Predictions")
    plt.legend()
        
    # X-coordinate over time
    plt.subplot(2, 2, 3)
    plot_trace(range(sequence_length), targets[batch_idx, :, 0], visible, 'Target', '-', 'darkblue', 'lightblue')
    plot_trace(range(sequence_length), outputs[batch_idx, :, 0], visible, 'Prediction', '--', 'maroon', 'lightcoral')
    plt.xlabel("Time step")
    plt.ylabel("X-coordinate")
    plt.title("X-Coordinate Trace over Time")
    plt.legend()
    
    # Y-coordinate over time
    plt.subplot(2, 2, 4)
    plot_trace(range(sequence_length), targets[batch_idx, :, 1], visible, 'Target', '-', 'darkblue', 'lightblue')
    plot_trace(range(sequence_length), outputs[batch_idx, :, 1], visible, 'Prediction', '--', 'maroon', 'lightcoral')
    plt.xlabel("Time step")
    plt.ylabel("Y-coordinate")
    plt.title("Y-Coordinate Trace over Time")
    plt.legend()
    
    # Save the plot to the specified directory
    save_path = os.path.join(save_dir, file_name)
    plt.savefig(save_path)
    plt.show()

