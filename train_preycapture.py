from models.simpleSC import RGC2SCNet

def parse_args():
    parser = argparse.ArgumentParser(description="Script for Model Training to get 3D RF in simulation")
    parser.add_argument('--config_name', type=str, default='neuro_exp1_2cell_030624', help='Config file name for data generation')

def main():
    args = parse_args()

if __name__ == '__main__':
    main()