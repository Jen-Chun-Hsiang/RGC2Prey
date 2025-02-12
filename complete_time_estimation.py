import os
import argparse
import datetime
import pytz
from utils.data_handling import parse_checkpoints, estimate_finish_time


def main():
    parser = argparse.ArgumentParser(
        description="Estimate final checkpoint finish time for one or more experiments."
    )
    parser.add_argument("--experiment_names", type=str, nargs='+', required=True,
                        help="One or more unique experiment identifier(s) (e.g., '25021002 25021103')")
    parser.add_argument("--final_epoch", type=int, required=True,
                        help="Final checkpoint epoch number (e.g., 200)")
    parser.add_argument("--input_folder", type=str,  default=None,
                        help="Folder containing checkpoint files (default: current directory)")
    parser.add_argument("--output_folder", type=str, default=None,
                        help="Folder to save the estimation result (default: current directory)")
    args = parser.parse_args()

    args = parser.parse_args()
    if args.input_folder is None:
        args.input_folder = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/RGC2Prey/Results/CheckPoints/'
    if args.output_folder is None:
        args.output_folder = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/RGC2Prey/Results/TimeEST/'

    # Ensure the output folder exists.
    os.makedirs(args.output_folder, exist_ok=True)

    for exp in args.experiment_names:
        print(f"Processing experiment: {exp}")
        checkpoints = parse_checkpoints(args.input_folder, exp)
        if not checkpoints:
            print(f"  No checkpoint files found for experiment '{exp}' in folder '{args.input_folder}'. Skipping.")
            continue

        try:
            estimated_time = estimate_finish_time(checkpoints, args.final_epoch)
        except ValueError as e:
            print(f"  Error for experiment {exp}: {e}")
            continue

        # Define the Chicago timezone
        chicago_tz = pytz.timezone("America/Chicago")

        # Convert Unix timestamp to Chicago time
        chicago_time = datetime.datetime.fromtimestamp(estimated_time, tz=pytz.utc).astimezone(chicago_tz)

        output_text = (
            f"Estimated finish time for experiment {exp} at epoch {args.final_epoch}:\n"
            f"{chicago_time.strftime('%Y-%m-%d %H:%M:%S %Z')} (Chicago Time)\n"
        )
        
        output_filename = f"{exp}_test-estimation_{args.final_epoch}.txt"
        output_path = os.path.join(args.output_folder, output_filename)
        with open(output_path, 'w') as f:
            f.write(output_text)
        
        print(f"  Estimation saved to: {output_path}")


if __name__ == "__main__":
    main()