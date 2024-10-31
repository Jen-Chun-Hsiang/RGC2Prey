from utils.preprocessing import convert_images_to_png, process_and_crop_images

if __name__ == "__main__":
    input_folder = "/storage1/fs1/KerschensteinerD/Active/Emily/PreyCaptureRGC/Simulation/Images" \
                   "/grass/"
    output_folder = "/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/CricketDataset/Images" \
                   "/original/grass/"
    convert_images_to_png(input_folder, output_folder)

    input_folder = output_folder
    output_folder = "/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/CricketDataset/Images" \
                   "/cropped/grass/"
    process_and_crop_images(input_folder, output_folder)