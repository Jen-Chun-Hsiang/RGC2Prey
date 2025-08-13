import shlex
from collections import Counter
from typing import Dict, List, Tuple

def find_duplicate_parameters(command_string: str) -> Dict[str, int]:
    """
    Find duplicate parameters within a single command string.
    
    Args:
        command_string (str): The command string to analyze
        
    Returns:
        dict: Dictionary where keys are parameter names and values are their counts.
              Only includes parameters that appear more than once.
    """
    try:
        # Split the command using shlex to handle quotes properly
        parts = shlex.split(command_string.strip())
    except ValueError:
        # If shlex fails, fall back to simple splitting
        parts = command_string.strip().split()
    
    if not parts:
        return {}
    
    # Extract parameters (skip the main command)
    params = parts[1:]
    
    # Count parameter flags
    parameter_counts = Counter()
    
    i = 0
    while i < len(params):
        if params[i].startswith('--') or params[i].startswith('-'):
            flag = params[i]
            parameter_counts[flag] += 1
            
            # Check if next parameter is a value (doesn't start with -)
            if i + 1 < len(params) and not params[i + 1].startswith('-'):
                i += 2  # Skip both flag and its value
            else:
                i += 1  # Just skip the flag
        else:
            i += 1  # Skip positional arguments
    
    # Return only duplicates (count > 1)
    return {param: count for param, count in parameter_counts.items() if count > 1}


def analyze_command_parameters(command_string: str) -> Dict[str, any]:
    """
    Comprehensive analysis of parameters within a single command string.
    
    Args:
        command_string (str): The command string to analyze
        
    Returns:
        dict: Analysis results including duplicates, all parameters, and summary
    """
    try:
        parts = shlex.split(command_string.strip())
    except ValueError:
        parts = command_string.strip().split()
    
    if not parts:
        return {
            'duplicates': {},
            'all_parameters': {},
            'total_parameters': 0,
            'unique_parameters': 0,
            'has_duplicates': False
        }
    
    main_command = parts[0]
    params = parts[1:]
    
    # Track all parameters with their values and positions
    all_parameters = {}
    parameter_counts = Counter()
    parameter_positions = {}
    
    i = 0
    param_index = 0
    while i < len(params):
        if params[i].startswith('--') or params[i].startswith('-'):
            flag = params[i]
            parameter_counts[flag] += 1
            
            # Track positions for each occurrence
            if flag not in parameter_positions:
                parameter_positions[flag] = []
            parameter_positions[flag].append(param_index)
            
            # Check if next parameter is a value
            if i + 1 < len(params) and not params[i + 1].startswith('-'):
                value = params[i + 1]
                if flag not in all_parameters:
                    all_parameters[flag] = []
                all_parameters[flag].append(value)
                i += 2
            else:
                # Boolean flag
                if flag not in all_parameters:
                    all_parameters[flag] = []
                all_parameters[flag].append(True)  # Boolean flag
                i += 1
            
            param_index += 1
        else:
            i += 1  # Skip positional arguments
    
    duplicates = {param: count for param, count in parameter_counts.items() if count > 1}
    
    return {
        'main_command': main_command,
        'duplicates': duplicates,
        'all_parameters': all_parameters,
        'parameter_positions': parameter_positions,
        'total_parameters': sum(parameter_counts.values()),
        'unique_parameters': len(parameter_counts),
        'has_duplicates': len(duplicates) > 0
    }


def print_duplicate_analysis(command_string: str) -> None:
    """
    Print a formatted analysis of duplicate parameters in a command string.
    
    Args:
        command_string (str): The command string to analyze
    """
    analysis = analyze_command_parameters(command_string)
    
    print("=" * 60)
    print("COMMAND PARAMETER DUPLICATE ANALYSIS")
    print("=" * 60)
    print(f"Command: {analysis['main_command']}")
    print(f"Total parameters: {analysis['total_parameters']}")
    print(f"Unique parameters: {analysis['unique_parameters']}")
    print(f"Has duplicates: {'Yes' if analysis['has_duplicates'] else 'No'}")
    
    if analysis['has_duplicates']:
        print(f"\nDUPLICATE PARAMETERS FOUND:")
        print("-" * 40)
        for param, count in analysis['duplicates'].items():
            print(f"Parameter: {param}")
            print(f"  Appears {count} times")
            print(f"  Values: {analysis['all_parameters'][param]}")
            print(f"  Positions: {analysis['parameter_positions'][param]}")
            print()
    else:
        print("\n✓ No duplicate parameters found!")
    
    print("=" * 60)


def parse_config(config_str):
    """
    Parse a shell command string, extracting all --key=value,
    --key value, and boolean --flags. Returns dict: key → value or True.
    """
    args, params = shlex.split(config_str), {}
    i = 0
    while i < len(args):
        arg = args[i]
        if arg.startswith('--'):
            # handle --key=val
            if '=' in arg:
                key, val = arg[2:].split('=', 1)
                params[key] = val
                i += 1
            else:
                key = arg.lstrip('-')
                # handle --key value
                if i + 1 < len(args) and not args[i+1].startswith('--'):
                    params[key] = args[i+1]
                    i += 2
                else:
                    # boolean flag
                    params[key] = True
                    i += 1
        else:
            i += 1
    return params

def print_table(headers, rows):
    """
    Print a simple ASCII table with left-aligned columns.
    """
    cols = list(zip(*([headers] + rows)))
    widths = [max(len(str(cell)) for cell in col) for col in cols]
    # header
    header_line = " | ".join(str(headers[i]).ljust(widths[i]) for i in range(len(headers)))
    sep_line    = "-+-".join("-" * widths[i] for i in range(len(headers)))
    print(header_line)
    print(sep_line)
    # rows
    for row in rows:
        print(" | ".join(str(row[i]).ljust(widths[i]) for i in range(len(headers))))


def analysize_configs(configs):
    parsed = [parse_config(c) for c in configs]
    names  = [p.get('experiment_name', f'config_{i+1}') for i, p in enumerate(parsed)]

    # gather all parameter keys except experiment_name
    all_keys = set().union(*parsed) - {'experiment_name'}

    # 1) Table: parameters defined in ≥2 configs with differing values
    diff_rows = []
    for key in sorted(all_keys):
        vals = [p[key] for p in parsed if key in p]
        if len(vals) >= 2 and len(set(vals)) > 1:
            row = [key] + [p.get(key, '-') for p in parsed]
            diff_rows.append(row)

    print("\nParameters with differing values across experiments:\n")
    if diff_rows:
        print_table(['Parameter'] + names, diff_rows)
    else:
        print("  (none)\n")

    # 2) Table: parameters not present in every config
    common_keys = set.intersection(*(set(p.keys()) for p in parsed)) - {'experiment_name'}
    unique_rows = []
    for key in sorted(all_keys - common_keys):
        row = [key] + [('v' if key in p else '') for p in parsed]
        unique_rows.append(row)

    print("Parameters NOT present in every config (v = present):\n")
    if unique_rows:
        print_table(['Parameter'] + names, unique_rows)
    else:
        print("  (none)\n")