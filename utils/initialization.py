import random
import numpy as np
import torch
import time
import os
import logging
from datetime import datetime

def set_seed(seed):
    if seed == "fixed":
        seed = 42  # Fixed seed
    elif seed == "random":
        seed = int(time.time())  # Use current time as a random seed
    elif isinstance(seed, (int, float)):  # If it's a numeric value, use it directly
        seed = int(seed)  # Ensure it's an integer
    else:
        raise ValueError("Seed must be 'fixed', 'random', or a numeric value.")

    # Set the seed for all relevant libraries
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    return seed

def process_seed(seed_input):
    """
    Validate and process the seed input.
    
    Parameters:
        seed_input (str): The seed value provided by the user, which can be
                          "fixed", "random", or a numeric string.
    
    Returns:
        None. Calls `set_seed` with the appropriate value.
    
    Raises:
        ValueError: If the input is invalid.
    """
    try:
        if seed_input.isdigit():  # Check if it's a numeric value
            seed_value = int(seed_input)
        elif seed_input in ["fixed", "random"]:  # Check if it's "fixed" or "random"
            seed_value = seed_input
        else:
            raise ValueError("Invalid seed input. Must be 'fixed', 'random', or a numeric value.")
        
        # Call the seed-setting function
        return set_seed(seed_value)
    except ValueError as e:
        print(f"Error: {e}")
        raise


def worker_init_fn(worker_id):
    """
    Initialize a unique seed for each DataLoader worker.
    """
    # Ensure each worker has a unique seed
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def initialize_logging(experiment_name, log_save_folder):
    """
    Initialize logging for the script by creating a timestamped log file.

    Parameters:
        experiment_name:
        log_save_folder: str
            The folder where log files will be saved.

    Returns:
        None
    """
    # Generate timestamp string
    timestr = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Construct the full path for the log file
    file_name = f'{experiment_name}_cricket_location_prediction'
    log_filename = os.path.join(log_save_folder, f'{file_name}_training_log_{timestr}.txt')

    # Setup logging
    logging.basicConfig(
        filename=log_filename,
        level=logging.INFO,
        format='%(asctime)s %(levelname)s:%(message)s'
    )

