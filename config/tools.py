import os
import yaml
import copy
from yacs import config

def get_student_cfg(cfg,args):
    if os.path.exists(args.student_file):
       with open(args.student_file) as file:
            student_file_cfg = config.load_cfg(file)
    else:
        print("File {} not exists".format(args.student_file))
        print("Please input a specific student config file")
        print("Exiting...")
        exit(-1)

    student_cfg = copy.deepcopy(cfg)
    for key, value in student_file_cfg.items():
        student_cfg[key] = value
    
    return student_cfg
