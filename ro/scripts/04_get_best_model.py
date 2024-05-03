import argparse
import pickle as pkl
import shutil

import numpy as np

import ro.params as params
from ro.utils import factory_get_path


def parse_run_name(run_name):
    """ Maybe useful at some point. """
    pass


def get_best_model(args):
    cfg = getattr(params, args.problem)
    get_path = factory_get_path(args.problem)
    results_fp = get_path(cfg.data_path, cfg, ptype=f'random_search/{args.model_type}_tr_res', suffix='.pkl')
    results_fp_prefix = str(results_fp.stem)
    model_suffix = '.pt'
    model_fp = get_path(cfg.data_path, cfg, ptype=f'random_search/{args.model_type}', suffix=model_suffix)

    # get all tr_res files
    results_paths = [str(x) for x in model_fp.parent.iterdir()]
    results_paths = [x for x in model_fp.parent.iterdir() if results_fp_prefix in str(x.stem)]
    results_paths = [x for x in results_paths if "__" in str(x.stem)]

    # find_best_result
    best_criterion, best_results_path = np.infty, None

    print(f'Checking {len(results_paths)} model files...')
    for rp in results_paths:
        rdict = pkl.load(open(rp, 'rb'))
        if best_criterion > rdict[args.criterion]:
            best_criterion = rdict[args.criterion]
            best_results_path = rp

    # generate best model path and save
    parts = str(best_results_path.stem).split('_')
    parts.remove('tr')
    parts.remove('res')
    best_model_path = "_".join(parts) + model_suffix
    best_model_path = best_results_path.parent.joinpath(best_model_path)

    best_model_path = str(best_model_path)
    best_results_path = str(best_results_path)

    model_fp = str(model_fp)
    results_fp = str(results_fp)

    results_fp = results_fp.replace('random_search/', '')
    model_fp = model_fp.replace('random_search/', '')

    print(f'Best model: {best_results_path}')
    print(f'Best {args.criterion}: {best_criterion}')
    print(f"Saving to:", {model_fp})

    shutil.copy(best_results_path, results_fp)
    shutil.copy(best_model_path, model_fp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--problem', type=str, default='kp')
    parser.add_argument('--model_type', type=str, default='set_encoder')
    parser.add_argument('--criterion', type=str, default='val_mae')

    args = parser.parse_args()

    get_best_model(args)
