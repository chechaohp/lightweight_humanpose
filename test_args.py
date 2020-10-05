import argparse
import os
from config import cfg
import copy
import yaml

def get_args():
    parser = argparse.ArgumentParser(description='Run Linear Classifer.')
    parser.add_argument('--student_file',required = True,help="student yaml file for training")
    parser.add_argument('--log', default = 'log', help="logging folder")
    args = parser.parse_args()
    return args

def get_student_cfg(cfg,args):
    if os.path.exists(args.student_file):
        with open(args.student_file) as file:
            student_file_cfg = yaml.load(file, Loader=yaml.FullLoader)
    else:
        print("File {} not exists".format(args.student_file))
        print("Please input a specific student config file")
        print("Exiting...")
        exit(-1)

    student_cfg = copy.deepcopy(cfg)
    for key, value in student_file_cfg.items():
        student_cfg[key] = value
    
    return student_file_cfg

def main():
    os.system('dir')

if __name__ == "__main__":
    main()

