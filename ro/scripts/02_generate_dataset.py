from argparse import ArgumentParser

from ro.dm import factory_dm
from ro.utils import DataManagerModes as Modes



def main(args):
    data_manager = factory_dm(args.problem)

    if args.by_inst:
        data_manager.generate_dataset_by_inst(args.n_procs, args.debug)
    else:
        data_manager.generate_dataset(args.n_procs, args.debug)



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--problem', type=str, default="kp", help="Problem from ro/params.py.")
    parser.add_argument('--n_procs', type=int, default=1, help="Number of threads.")
    parser.add_argument('--debug', type=int, default=0, help="Indictor to enable/disable debugging.")
    parser.add_argument('--by_inst', type=int, default=0, help="Indictor to generate dataset by instance of not.")
    args = parser.parse_args()
    main(args)
