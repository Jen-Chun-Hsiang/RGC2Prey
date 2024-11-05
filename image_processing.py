from utils.preprocessing import convert_images_to_png, process_and_crop_images, scale_image_to_max_size

def convert_file_type():
    folder_name = 'cricket'
    input_folder = "/storage1/fs1/KerschensteinerD/Active/Emily/PreyCaptureRGC/Simulation/Images" \
                   f"/{folder_name}/"
    output_folder = "/storage1/fs1/KerschensteinerD/Active/Emily/PreyCaptureRGC/Simulation/Images" \
                    f"/converted/{folder_name}/"
    convert_images_to_png(input_folder, output_folder)


def convert_resize_file():
    folder_name = 'cricket_exc'
    input_folder = "/storage1/fs1/KerschensteinerD/Active/Emily/PreyCaptureRGC/Simulation/Images" \
                   f"/{folder_name}/"
    folder_name = 'cricket'
    output_folder = "/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/CricketDataset/Images" \
                    f"/cropped/{folder_name}/"
    scale_image_to_max_size(input_folder, output_folder, max_side_length=100)


def convert_resize_crop_file():
    folder_name = 'cricket'
    input_folder = "/storage1/fs1/KerschensteinerD/Active/Emily/PreyCaptureRGC/Simulation/Images" \
                   f"/{folder_name}/"
    output_folder = "/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/CricketDataset/Images" \
                    f"/cropped/{folder_name}/"
    process_and_crop_images(input_folder, output_folder)


def execute_task(task_id):
    task_functions = {
        1: convert_file_type,
        2: convert_resize_file,
        3: convert_resize_crop_file,
    }
    func = task_functions.get(task_id)
    if func is None:
        raise ValueError("No task found for argument: {}".format(task_id))
    func()
    

if __name__ == "__main__":
    run_task_id = 2
    execute_task(run_task_id)