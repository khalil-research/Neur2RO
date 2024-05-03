import argparse

import time
import copy
import collections
import numpy as np
import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt

import ro.params as ro_params
from ro.two_ro import factory_two_ro
from ro.utils import factory_load_problem, factory_get_path

import importlib
ccg_functions = importlib.import_module('ro.scripts.05_run_ml_ccg')

# from ro.utils.cb import get_path, read_test_instance
# from ro.evaluation.ssa import factory_eval as factory_ssa
# from ro.evaluation.kadapt_x import factory_eval as factory_kadapt
# from ro.evaluation.ss_relax import factory_eval as factory_ss_relax


def get_test_instance(args, cfg):
    """ Gets test instance. """
    if 'kp' in args.problem:
        inst_dir =  cfg.data_path + "kp/eval_instances/"
        inst_name = f"RKP_{args.kp_correlation}"
        inst_name += f"_n{args.kp_n_items}"
        inst_name += f"_R{args.kp_R}"
        inst_name += f"_H{args.kp_H}"
        inst_name += f"_h{args.kp_h}"
        inst_name += f"_dev{args.kp_budget_factor}"
        if args.kp_delta == 1.0:
            inst_name += f"_d{int(args.kp_delta)}"
        else:
            inst_name += f"_d{args.kp_delta}"

        # initialize instance
        inst = {}
        inst['inst_name'] = inst_name

        # add parameters to inst
        inst['n_items'] = args.kp_n_items
        inst['correlation'] = args.kp_correlation
        inst['R'] = args.kp_R
        inst['H'] = args.kp_H
        inst['h'] = args.kp_h
        inst['budget_factor'] = args.kp_budget_factor
        inst['max_budget'] = inst['n_items'] * inst['budget_factor']

        # load instance
        fp_inst = inst_dir + inst_name
        with open(fp_inst, 'r') as f:
            f_lines = f.readlines()

        # get metainfo, only C is required
        inst_info = list(map(lambda x: float(x), f_lines[0].replace('\n', '').split(' ')))
        inst['C'] = inst_info[1]

        # store inst data
        inst_data = f_lines[1:]
        inst_data = list(map(lambda line: line.replace('\n', '').split(' ')[1:], inst_data))
        p_bar, p_hat, t, c, f = [], [], [], [], []
        for data_i in inst_data:
            p_bar.append(float(data_i[0]))
            p_hat.append(float(data_i[1]))
            t.append(float(data_i[2]))
            c.append(float(data_i[3]))
            f.append(float(data_i[4]))

        inst['p_bar'] = np.array(p_bar)
        inst['p_hat'] = np.array(p_hat)
        inst['t'] = np.array(t)
        inst['c'] = np.array(c)
        inst['f'] = np.array(f)
        inst['n_items'] = args.kp_n_items
        inst['max_budget'] = inst['budget_factor'] * inst['n_items']
        
    else:
        raise Exception(f"Not implemented for problem={args.problem}")

    return inst


def get_baseline_results(args, cfg, inst, two_ro):
    """ Gets baseline results. """
    if 'kp' in args.problem:
        # load results from csv
        result_dir = cfg.data_path + 'kp/baseline_results/ResultsForBranchAndPrice.csv'
        df = pd.read_csv(result_dir)
        inst_res = df.loc[inst['inst_name'] == df['File name']]
        inst_res = inst_res.to_dict()

        baseline_results = {
            'obj' : list(inst_res['Best primal bound'].values())[0],
            'time' : list(inst_res['Time'].values())[0],
        }

    else:
        raise Exception(f"Not implemented for problem={args.problem}")

    return baseline_results





def get_baseline_inst(args, inst, n_samples):
    """ Gets baseline instances. """
    # load results from baseline
    result_dir = self.cfg.data_path + 'kp/ResultsForBranchAndPrice.csv'
    df = pd.read_csv(result_dir)

    inst_res = df.loc[inst['inst_name'] == df['File name']]
    inst_res = inst_res.to_dict()

    self.baseline_results = {
        'obj' : list(inst_res['Best primal bound'].values())[0],
        'time' : list(inst_res['Time'].values())[0],
    }


def eval_pricing():
    pass



# -----------#
#     Main   #
# -----------#


def main(args):

    print("Evaluating ML-CCG solution for ")

    if "kp" in args.problem:
        print(f"   problem:              {args.problem}")
        print(f"   n_items:              {args.kp_n_items}")
        print(f"   correlation:          {args.kp_correlation}")
        print(f"   delta:                {args.kp_delta}")
        print(f"   h:                    {args.kp_h}")
        print(f"   budget_factor:        {args.kp_budget_factor}")
        print(f"   R:                    {args.kp_R}")
        print(f"   H:                    {args.kp_H}\n\n")

    # load config and paths
    cfg = getattr(ro_params, args.problem)

    get_path = factory_get_path(args.problem)

    if args.opt_type == "pga":
        fp_ml_res = get_path(cfg.data_path, cfg, ccg_functions.get_result_str(args, parent_dir="ml_ccg_results_pga"))
        fp_res = get_path(cfg.data_path, cfg, ccg_functions.get_result_str(args, p_type="results", parent_dir="eval_results_pga"))
    else:
        fp_ml_res = get_path(cfg.data_path, cfg, ccg_functions.get_result_str(args))
        fp_res = get_path(cfg.data_path, cfg, ccg_functions.get_result_str(args, p_type="results", parent_dir="eval_results"))

    # get test instance
    inst = get_test_instance(args, cfg)

    # initialize two_ro problem
    two_ro = factory_two_ro(args.problem)

    # get baseline results
    baseline_results = get_baseline_results(args, cfg, inst, two_ro)

    # get ml results
    with open(fp_ml_res, 'rb') as p:
        ml_results = pkl.load(p)

    x = ml_results['x']

    # init two_ro class
    obj, xi, y, r = two_ro.evaluate_first_stage_solution(x, inst)
    
    # compute optimality gap
    gap = 100 * (baseline_results['obj'] - obj)/baseline_results['obj']
            
    # report results
    print("\n  Learning-Based Algorithm:")
    print("     Objective: ", obj)
    print("     Time:      ", ml_results['opt_stats']['time'])

    print("  Baseline:")
    print("    Objective: ", baseline_results['obj'])
    print("    Time:      ", baseline_results['time'])

    print("  Gap:", "{:2f}%\n".format(gap))


    results = {
        'gap' : gap,
        'algo_obj' : obj,
        'algo_time' : ml_results['opt_stats']['time'],
        'baseline_obj' : baseline_results['obj'],
        'baseline_time' : baseline_results['time'],
        'inst_name' : inst['inst_name'],
        'opt_stats' : ml_results['opt_stats'],
        'ml_results' : ml_results,
    }


    # save results
    print(f"Saving results to: {fp_res}")
    with open(fp_res, 'wb') as p:
        pkl.dump(results, p)
    print("Done!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluates row generation with tiny set network for knapsack problem.')

    # evaluation type
    parser.add_argument('--eval_method', type=str, default="pricing", choices=["pricing"])

    # instance parameters
    parser.add_argument('--problem', type=str, default="kp", help='Number of items [do not change, see ]')
    
    # KP specific parameters
    parser.add_argument('--kp_n_items', type=int, default=20, choices=[20,30,40,50,60,70,80], help='Number of items [do not change, see ]')
    parser.add_argument('--kp_correlation', type=str, default='UN', choices=['UN', 'WC', 'ASC', 'SC'], help='Correlation type [do not change, see ]')
    parser.add_argument('--kp_delta', type=float, default=0.1, choices=[0.1, 0.5, 1],  help='Delta [do not change, see ]')
    parser.add_argument('--kp_h', type=int, default=40, choices=[40, 80], help='h [do not change, see ]')
    parser.add_argument('--kp_budget_factor', type=float, default=0.1, choices=[0.1, 0.15, 0.20], help='budget factor [do not change, see ]')
    parser.add_argument('--kp_R', type=int, default=1000, choices=[1000], help='h [do not change, see ]')
    parser.add_argument('--kp_H', type=float, default=100, choices=[100], help='budget factor [do not change, see ]')

    # type of ml model
    parser.add_argument('--model_type', type=str, default='nn_tiny_set', help='Number of items [do not change, see ]')

    # type of optimization
    parser.add_argument('--opt_type', type=str, default='adversarial', help='Number of items [do not change, see ]')

    # output
    parser.add_argument('--verbose', type=int, default=0, help='Verbose param for optimization model')

    args = parser.parse_args()

    main(args)
