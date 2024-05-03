from argparse import ArgumentParser

from ro.dm import factory_dm
from ro.utils import DataManagerModes as Modes


def main(args):
    data_manager = factory_dm(args.problem)
    data_manager.generate_test_instances(args)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--problem', type=str, default="cb_test")
    parser.add_argument('--n_items', type=int, default=10, choices=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    parser.add_argument('--n_inst_per_size', type=int, default=100)
    args = parser.parse_args()
    main(args)
