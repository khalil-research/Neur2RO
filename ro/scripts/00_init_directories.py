import os 
import sys
import subprocess

from argparse import ArgumentParser

import ro.params as params
from ro.utils import factory_get_path

#-----------------------------------------------------------------------#
#                                                                       #
#       File to initialize all directories for a specific problem       #  
#                                                                       #
#-----------------------------------------------------------------------#


def main(args):

    # load config and paths
    global cfg
    
    cfg = getattr(params, args.problem)

    # initialize get_path
    get_path = factory_get_path(args.problem)


    fp = get_path(cfg.data_path, cfg, "", suffix="")

    main_fp, prob_fp, _ = str(fp).split("/")  
    prob_fp = main_fp + "/" + prob_fp

    # initialize main directory
    if os.path.isdir(main_fp):
        print(f"main directory ({main_fp}) exists.")
    else:
        os.mkdir(main_fp)
        print(f"created main directory ({main_fp}).")

    # initialize problem directory
    if os.path.isdir(prob_fp):
        print(f"problem directory ({prob_fp}) exists.")
    else:
        os.mkdir(prob_fp)
        print(f"created problem directory ({prob_fp}).")

    # initialize all sub-directories
    # sub-directories to initialize
    sub_dirs = [
        'random_search',
        'ml_ccg_results',
        'ml_ccg_pga_results',
        'eval_results',
        'eval_results_pga',
        'eval_instances',
        'baseline_results',
    ]

    sub_dirs = list(map(lambda x: prob_fp + "/" + x, sub_dirs))
    
    print("subdirectories:")
    
    # check and create sub dirs if needed
    for sub_dir in sub_dirs:

        if os.path.isdir(sub_dir):
            print(f"    sub directory ({sub_dir}) exists.")
        else:
            os.mkdir(sub_dir)
            print(f"    created sub directory ({sub_dir})")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--problem', type=str, default="kp")
    args = parser.parse_args()
    main(args)