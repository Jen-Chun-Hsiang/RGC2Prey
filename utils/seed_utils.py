import random
import numpy as np
import torch
import time

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

    print(f"Random seed set to: {seed}")

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
        set_seed(seed_value)
    except ValueError as e:
        print(f"Error: {e}")
