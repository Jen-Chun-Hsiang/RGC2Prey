import torch
import os
import re
import time
from datetime import datetime, timedelta


def save_checkpoint(epoch, model, optimizer, training_losses, scheduler=None, args=None, 
                    validation_losses=None, validation_contra_losses=None,
                    file_path=None, learning_rate_dynamics=None):
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'args': args,
        'training_losses': training_losses,
        'validation_losses': validation_losses,
        'validation_contra_losses': validation_contra_losses,
        'learning_rate_dynamics': learning_rate_dynamics
    }
    torch.save(checkpoint, file_path)


class CheckpointLoader:
    def __init__(self, file_path):
        self.checkpoint = None
        self.start_epoch = None
        self.training_losses = None
        self.validation_losses = None
        self.validation_contra_losses = None
        self.learning_rate_dynamics = None
        self.args = None
        self.checkpoint = torch.load(file_path)

    def load_args(self):
        self.args = self.checkpoint['args']
        return self.args

    def load_checkpoint(self, model, optimizer, scheduler=None):
        model.load_state_dict(self.checkpoint['model_state_dict'])
        optimizer.load_state_dict(self.checkpoint['optimizer_state_dict'])
        if scheduler is not None:
            scheduler.load_state_dict(self.checkpoint['scheduler_state_dict'])

        return model, optimizer, scheduler

    def load_epoch(self):
        """ Return the epoch at which training was interrupted. """
        self.start_epoch = self.checkpoint['epoch']
        return self.start_epoch

    def load_training_losses(self):
        """ Return the list of recorded training losses. """
        self.training_losses = self.checkpoint.get('training_losses', [])
        return self.training_losses

    def load_validation_losses(self):
        """ Return the list of recorded validation losses. """
        self.validation_losses = self.checkpoint.get('validation_losses', [])
        return self.validation_losses
    

def extract_experiment_identifiers(folder_path):
    """
    Extract unique experiment identifiers from checkpoint filenames.
    """
    pattern = re.compile(r'^(\d{8})_checkpoint_epoch_(\d+)\.pth$')
    experiment_epochs = {}
    
    for filename in os.listdir(folder_path):
        match = pattern.match(filename)
        if match:
            experiment_id, epoch = match.groups()
            epoch = int(epoch)
            file_path = os.path.join(folder_path, filename)
            created_time = os.path.getctime(file_path)
            
            if experiment_id not in experiment_epochs:
                experiment_epochs[experiment_id] = []
            experiment_epochs[experiment_id].append((epoch, created_time))
    
    return experiment_epochs

def estimate_completion_time(experiment_epochs, final_epoch):
    """
    Estimate the completion time for the final checkpoint.
    """
    estimated_times = {}
    for experiment_id, data in experiment_epochs.items():
        # Sort by epoch number
        data.sort()
        epochs, timestamps = zip(*data)
        
        if len(epochs) > 1:
            # Compute average time per epoch gap
            epoch_gap = epochs[1] - epochs[0]  # Assumes uniform epoch gaps
            time_gaps = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps) - 1)]
            avg_time_per_gap = sum(time_gaps) / len(time_gaps)
            
            # Predict final epoch completion time
            last_epoch, last_timestamp = epochs[-1], timestamps[-1]
            remaining_epochs = (final_epoch - last_epoch) // epoch_gap
            estimated_final_time = last_timestamp + (remaining_epochs * avg_time_per_gap)
            
            estimated_times[experiment_id] = datetime.fromtimestamp(estimated_final_time).strftime('%Y-%m-%d %H:%M:%S')
    
    return estimated_times

def save_estimation_results(output_folder, experiment_name, final_epoch, estimated_time):
    """
    Save the estimated completion time in a text file.
    """
    os.makedirs(output_folder, exist_ok=True)
    output_file = os.path.join(output_folder, f'{experiment_name}_test-estimation_{final_epoch}.pth')
    
    with open(output_file, 'w') as f:
        f.write(f'Estimated finish time for experiment {experiment_name} at epoch {final_epoch}: {estimated_time}\n')
    
    print(f'Estimation saved to {output_file}')


