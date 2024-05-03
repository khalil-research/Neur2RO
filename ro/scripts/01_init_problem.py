from argparse import ArgumentParser

from ro.dm import factory_dm

def main(args):
    data_manager = factory_dm(args.problem)
    data_manager.init_problem()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--problem', type=str, default="kp")
    args = parser.parse_args()
    main(args)
