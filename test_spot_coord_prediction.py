import os
import time
from datetime import datetime
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import logging

from datasets.movingdot import RandomMovingSpotDataset, CNN_LSTM_ObjectLocation, plot_and_save_results
# Dataset and DataLoader parameters
sequence_length = 50
num_epochs = 10
grid_height, grid_width = 24, 32
prob_vis = 0.5
num_samples = 10000  # Number of samples to generate in an epoch (arbitrary choice for demonstration)
batch_size = 64
sim_id = '11132401'

plot_save_folder  = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/CricketDataset/Figures/'
log_save_folder  = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/CricketDataset/Logs/'
file_name = f'spot_coords_prediction_{sim_id}'
timestr = datetime.now().strftime('%Y%m%d_%H%M%S')
# 
# Construct the full path for the log file
log_filename = os.path.join(log_save_folder, f'{file_name}_training_log_{timestr}.txt')

# Setup logging
logging.basicConfig(filename=log_filename,
                    level=logging.INFO,
                    format='%(asctime)s %(levelname)s:%(message)s')

# Instantiate the dataset and DataLoader
dataset = RandomMovingSpotDataset(sequence_length, grid_height, grid_width, prob_vis, num_samples)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Sample Training Loop
model = CNN_LSTM_ObjectLocation(cnn_feature_dim=128, lstm_hidden_size=64, lstm_num_layers=2, output_dim=2,
              input_height=24, input_width=32, conv1_out_channels=16, conv2_out_channels=32, fc_out_features=128)
# model = CNNFeatureExtractor(input_height=24, input_width=32, conv1_out_channels=16, conv2_out_channels=32, fc_out_features=128)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

epoch_losses = []  # To store the loss at each epoch

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    
    start_time = time.time()
    for sequences, targets, visible in dataloader:
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(sequences)
        
        # Mask for invisible targets
        # mask = (targets != -1).all(dim=2).float().unsqueeze(2)
        # masked_outputs = outputs * mask
        # masked_targets = targets * mask
        
        # Compute loss
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()

        
    
    # Average loss for the epoch
    avg_epoch_loss = epoch_loss / len(dataloader)
    epoch_losses.append(avg_epoch_loss)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_epoch_loss:.4f}")

    elapsed_time = time.time() - start_time
    logging.info( f"{file_name} Epoch [{epoch + 1}/{num_epochs}], Elapsed time: {elapsed_time:.2f} seconds \n"
                    f"\tLoss: {avg_epoch_loss:.4f} \n")



# Call the function to plot and save results
plot_and_save_results(epoch_losses, model, dataloader, sequence_length, save_dir=plot_save_folder, file_name=f"{file_name}.png")
