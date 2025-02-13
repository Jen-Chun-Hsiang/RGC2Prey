import argparse
from utils.data_handling import extract_experiment_identifiers, estimate_completion_time, save_estimation_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Estimate final checkpoint completion time.')
    parser.add_argument('--folder_path', type=str,  default=None, help='Path to folder containing checkpoint files')
    parser.add_argument('--experiment_name', type=str, help='Experiment identifier')
    parser.add_argument('--final_epoch', type=int, default=200, help='Final epoch checkpoint number')
    parser.add_argument('--output_folder', type=str,  default=None, help='Folder to save estimation results')
    
    args = parser.parse_args()
    if args.folder_path is None:
        args.folder_path = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/RGC2Prey/Results/CheckPoints/'
    if args.output_folder is None:
        args.output_folder = '/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/RGC2Prey/Results/TimeEST/'
    
    experiment_epochs = extract_experiment_identifiers(args.folder_path)
    estimated_times = estimate_completion_time(experiment_epochs, args.final_epoch)
    
    if args.experiment_name in estimated_times:
        save_estimation_results(args.output_folder, args.experiment_name, args.final_epoch, estimated_times[args.experiment_name])
    else:
        print(f'No data found for experiment {args.experiment_name}.')