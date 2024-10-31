from utils import convert_images_to_png

if __name__ == "__main__":
    input_folder = "/storage1/fs1/KerschensteinerD/Active/Emily/PreyCaptureRGC/Simulation/Images" \
                   "/grass/"
    output_folder = "/storage1/fs1/KerschensteinerD/Active/Emily/RISserver/CricketDataset/Images" \
                   "original/grass/"
    convert_images_to_png(input_folder, output_folder)