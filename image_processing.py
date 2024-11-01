from utils.preprocessing import convert_images_to_png, process_and_crop_images

if __name__ == "__main__":
    folder_name = 'floor'
    input_folder = "/storage1/fs1/KerschensteinerD/Active/Emily/PreyCaptureRGC/Simulation/Images" \
                   f"/{folder_name}/"
    # output_folder = "/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/CricketDataset/Images" \
    #               "/original/grass/"
    # convert_images_to_png(input_folder, output_folder)

    # input_folder = output_folder
    output_folder = "/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/CricketDataset/Images" \
                    f"/cropped/{folder_name}/"
    process_and_crop_images(input_folder, output_folder)