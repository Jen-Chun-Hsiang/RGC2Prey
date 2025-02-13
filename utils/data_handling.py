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
    

def parse_checkpoints(folder, experiment_name):
    """
    Scan the folder for checkpoint files that match the pattern:
      {experiment_name}_checkpoint_epoch_{epoch}.pth
    and return a sorted list of tuples: (epoch, creation_time).
    """
    # Regex pattern to match the filenames for the given experiment.
    pattern = re.compile(rf'^{re.escape(experiment_name)}_checkpoint_epoch_(\d+)\.pth$')
    checkpoints = []
    
    for filename in os.listdir(folder):
        match = pattern.match(filename)
        if match:
            try:
                epoch = int(match.group(1))
            except ValueError:
                continue  # Skip files with non-integer epoch parts.
            file_path = os.path.join(folder, filename)
            # Get the file creation time (in Unix epoch seconds)
            ctime = os.path.getctime(file_path)
            checkpoints.append((epoch, ctime))
    
    # Sort checkpoints by epoch number (ascending)
    return sorted(checkpoints, key=lambda x: x[0])

def estimate_finish_time(checkpoints, final_epoch):
    """
    Given a sorted list of checkpoint tuples (epoch, creation_time) and the final epoch,
    compute the average time per epoch (using differences between available checkpoints)
    and extrapolate the finish time for the final checkpoint.
    """
    if len(checkpoints) < 2:
        raise ValueError("At least two checkpoint files are required to estimate time per epoch.")

    # Compute time per epoch from consecutive checkpoints.
    time_per_epoch_list = []
    for i in range(1, len(checkpoints)):
        prev_epoch, prev_time = checkpoints[i-1]
        curr_epoch, curr_time = checkpoints[i]
        epoch_diff = curr_epoch - prev_epoch
        if epoch_diff <= 0:
            continue  # Skip invalid or duplicate entries.
        time_diff = curr_time - prev_time
        time_per_epoch = time_diff / epoch_diff
        time_per_epoch_list.append(time_per_epoch)

    if not time_per_epoch_list:
        raise ValueError("Could not compute valid time differences between checkpoint files.")

    avg_time_per_epoch = statistics.mean(time_per_epoch_list)
    
    # Extrapolate finish time:
    # finish_time = last checkpoint time + (remaining epochs * avg time per epoch)
    last_epoch, last_time = checkpoints[-1]
    remaining_epochs = final_epoch - last_epoch
    if remaining_epochs < 0:
        raise ValueError("The provided final epoch is less than the last observed checkpoint epoch.")
    
    estimated_finish_time = last_time + remaining_epochs * avg_time_per_epoch
    return estimated_finish_time


