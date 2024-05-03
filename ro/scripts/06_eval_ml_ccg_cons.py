import argparse

import time
import copy
import collections
import numpy as np
import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt

import ro.params as ro_params
from ro.utils.cb import read_test_instance as cb_read_test_instance
from ro.utils import factory_load_problem, factory_get_path
from ro.evaluation.ssa import factory_eval as factory_ssa
from ro.evaluation.kadapt_x import factory_eval as factory_kadapt
from ro.evaluation.ss_relax import factory_eval as factory_ss_relax


# from ro.scripts.run_ml_ccg import get_result_str
import importlib
ccg_functions = importlib.import_module('ro.scripts.05_run_ml_ccg')


def get_test_instance(args, cfg):
    """ Gets test instance for specified parameters. """
    if "cb" in args.problem:
        inst_dir = cfg.data_path + "cb/eval_instances/"
        inst = cb_read_test_instance(inst_dir, cfg, args.cb_inst_seed, args.cb_n_items)

    return inst


def get_k_adapt_res(args, cfg, k):
    """ Gets k-adaptibility result.  """
    if "cb" in args.problem:
        f_name = f'res_CB_n{args.cb_n_items}_xi4_s{args.cb_inst_seed}_K{k}.pkl'
        fp_kadapt_res = cfg.data_path + 'cb/k_adapt_res/' + f_name

    # load k-adaptability results
    with open(fp_kadapt_res, 'rb') as p:
        kadapt_results = pkl.load(p)

    return kadapt_results


# def check_feasibility(x, inst):
#     """ Checks feasibility of first-stage decisions. """
#     print('\n\nChecking feasibility of first-stage')

#     # initialize model
#     m = gp.Model()
#     m.setParam("OutputFlag", 0)

#     # define x variables
#     x_var = m.addVars(inst['n_items'], name="x", vtype="B")
#     m._x = x_var

#     # define xi variables
#     xi_var = m.addVars(inst['xi_dim'], name="xi", vtype="C", lb=-1, ub=1)
#     m._xi = xi_var

#     # fix x variable to parameter passed as input
#     for i in range(inst['n_items']):  # enumerate(x_var):
#         m._x[i].lb = x[i]
#         m._x[i].ub = x[i]

#     # objective function
#     cost_vector = cost_fun(inst, m._xi)
#     cons_x = sum(cost_vector[i] * x_var[i] for i in range(inst['n_items']))
#     m.setObjective(cons_x, sense=gp.GRB.MAXIMIZE)

#     # optimize
#     m.optimize()

#     # check for violation
#     wc_cons_obj = m.objVal

#     is_feas = wc_cons_obj < inst['budget']

#     print("  x:  ", x)
#     print("  xi: ", list(map(lambda y: y.x, m._xi.values())))
#     print("  Budget:                             ", inst['budget'])
#     print("  Worst-case objective (c(xi)^T x):   ", wc_cons_obj)
#     print("  Is feasible:                        ", is_feas)
#     print("\n\n")

#     return is_feas


def sample_random_xi(args, inst, n_samples):
    """ Samples random uncertainty vectors. """
    if 'cb' in args.problem: 
        xi = np.random.uniform(-1, 1, size=(n_samples, inst['xi_dim']))

    elif 'kp' in args.problem:
        raise Exception("To be implemented.")

    return xi



def eval_ccg():
    pass 


def eval_ccg_relax():
    pass 


def eval_k_adapt():
    pass


def eval_pricing():
    pass



# -----------#
#     Main   #
# -----------#


def main(args):

    # load config and paths
    cfg = getattr(ro_params, args.problem)
    get_path = factory_get_path(args.problem)

    # load nn results results
    print("Loading ML results ... ")
    
    if args.opt_type == "pga":
        fp_ml_res = get_path(cfg.data_path, cfg, ccg_functions.get_result_str(args, parent_dir="ml_ccg_results_pga"))
        fp_res = get_path(cfg.data_path, cfg, ccg_functions.get_result_str(args, p_type="results", parent_dir="eval_results_pga"))
    else:
        fp_ml_res = get_path(cfg.data_path, cfg, ccg_functions.get_result_str(args))
        fp_res = get_path(cfg.data_path, cfg, ccg_functions.get_result_str(args, p_type="results", parent_dir="eval_results"))

    # get instance
    base_inst = get_test_instance(args, cfg)

    # get ml results
    with open(fp_ml_res, 'rb') as p:
        ml_results = pkl.load(p)

    # get first-stage sol + worst-case scenarios for ML algorithm
    x_ml = ml_results['x']
    xi_ml = list(ml_results['opt_stats']['xi_added']['obj'])
    xi_ml += list(ml_results['opt_stats']['xi_added']['feas'])
    time_ml = ml_results['algo_time']

    print("    Done")

    # load k-adapt result
    print(f"Loading k-adapt (K={args.baseline_k}) results ... ")

    k_adapt_results = {}
    for k in args.baseline_k:

        # try:
        # load k-adaptability results
        kadapt_results = get_k_adapt_res(args, cfg, k)
       
        # get first-stage sol + worst-case scenarios for k-adapt
        x_kadapt = kadapt_results['x']
        xi_kadapt = np.concatenate(list(kadapt_results['scenarios'].values())).tolist()     # changed that it gets the scens from all subsets
        time_kadapt = kadapt_results['runtime']

        x_at_ml_time = np.nan
        sol_at_ml_time = False
        for x_time, x_inc in kadapt_results['inc_x_t'].items():
            # print(time_ml, x_time, x_inc)
            if x_time <= time_ml and len(x_inc) > 0:
                # print("   ADDING")
                x_at_ml_time = x_inc
                sol_at_ml_time = True

        # except:
        #     # using all zeros solution and no scenarios if file does not exist
        #     print(f"Failed to load k adaptibility results for k={k}.")
        #     kadapt_results = np.nan
        #     x_kadapt = np.nan
        #     xi_kadapt = np.nan
        #     time_kadapt = np.nan
        #     x_at_ml_time = np.nan
        #     sol_at_ml_time = False

        k_adapt_results[k] = {}
        k_adapt_results[k]['x'] = x_kadapt
        k_adapt_results[k]['xi'] = xi_kadapt
        k_adapt_results[k]['time'] = time_kadapt
        k_adapt_results[k]['results'] = kadapt_results
        k_adapt_results[k]['x_at_ml_time'] = x_at_ml_time
        k_adapt_results[k]['sol_at_ml_time'] = sol_at_ml_time

    
    print("    Done")

    if args.eval_method == "ssa":
        print("Loading scenarios ... ")
        xi_ml = list(ml_results['opt_stats']['xi_added']['obj'])
        xi_ml += list(ml_results['opt_stats']['xi_added']['feas'])

        xi_k_adapt = []
        for k in args.baseline_k:
            try: # handle exception with no solutions found
                xi_k = list(k_adapt_results[k]['results']['scenarios'].values())
                xi_k = np.concatenate(xi_k).tolist()  # changed that it gets the scens from all subsets
                xi_k_adapt += xi_k
            except:
                pass

        # add additional randomly sampled scenarios
        if args.ssa_n_sample_scenarios > 0:
            np.random.seed(args.ssa_n_sample_scenarios)
            print(f"Evaluating on {args.ssa_n_sample_scenarios} additional randomly sampled scenarios")
            sampled_scenarios = sample_random_xi(args, base_inst, args.ssa_n_sample_scenarios).tolist()

        # combine scenarios
        scenarios = xi_ml + xi_k_adapt

        if args.ssa_n_sample_scenarios > 0:
            scenarios += sampled_scenarios

    # evaluate for both solutions
    time_eval = time.time()
    if args.eval_method == "ssa":
        eval_m = factory_ssa(args.problem)
        eval_m.set_inst(base_inst)
        
        # get ml objective
        obj_ml = eval_m.get_obj(x_ml, scenarios, args.n_procs)

        # get k-adapt objectives
        obj_baseline = {}
        obj_at_ml_time = {}
        for k in args.baseline_k:
            x_kadapt = k_adapt_results[k]['x']
            obj_baseline[k] = eval_m.get_obj(x_kadapt, scenarios, args.n_procs)

            if k_adapt_results[k]['sol_at_ml_time']:
                x_kadapt_at_ml = k_adapt_results[k]['x_at_ml_time']
                obj_at_ml_time[k] = eval_m.get_obj(x_kadapt_at_ml, scenarios, args.n_procs)
            else:
                obj_at_ml_time[k] = np.nan 

    elif args.eval_method == "kadapt":
        time_limit_kadapt = args.eval_kadapt_time_limit
        eval_m = factory_kadapt(args.problem)
        eval_m.set_inst(base_inst)

        # get ml objective
        obj_ml = eval_m.run_algorithm(args.K_eval, time_limit_kadapt, x_eval=x_ml)

        # get k-adapt objectives
        obj_baseline = {}
        obj_at_ml_time = {}
        for k in args.baseline_k:
            x_kadapt = k_adapt_results[k]['x']
            obj_baseline[k] = eval_m.run_algorithm(args.K_eval, time_limit_kadapt, x_eval=x_kadapt)
            
            if k_adapt_results[k]['sol_at_ml_time']:
                x_kadapt_at_ml = k_adapt_results[k]['x_at_ml_time']
                obj_at_ml_time[k] = eval_m.run_algorithm(args.K_eval, time_limit_kadapt, x_eval=x_kadapt_at_ml)
            else:
                obj_at_ml_time[k] = np.nan 

    elif args.eval_method == "ss_relax":
        raise Exception("eval_method (ss_relax), not implemented for multiple k") # todo change this is using later
        eval_m = factory_ss_relax(args.problem)
        eval_m.set_inst(base_inst)
        print("Evaluating baseline ... ")
        obj_baseline = eval_m.adversarial_problem(x_kadapt)
        print("Evaluating ML ... ")
        obj_ml = eval_m.adversarial_problem(x_ml)

    time_eval = time.time() - time_eval

    # compute optimality gap
    gaps = {}
    for k in args.baseline_k:
        if np.isnan(time_ml) or np.isnan(k_adapt_results[k]['time']):
            gaps[k] = np.nan
        else:
            gaps[k] = 100 * (obj_baseline[k] - obj_ml) / obj_baseline[k]

    # compute optimality gaps at ml timeout
    gaps_at_ml_time = {}
    for k in args.baseline_k:
        if np.isnan(time_ml) or np.isnan(k_adapt_results[k]['sol_at_ml_time']):
            gaps_at_ml_time[k] = np.nan
        else:
            gaps_at_ml_time[k] = 100 * (obj_at_ml_time[k] - obj_ml) / obj_at_ml_time[k]

    # eval time
    print(f"\n\n  Evaluation time is {time_eval} with {args.n_procs}")

    # report results
    print("\n  Learning-Based Algorithm:")
    print("     Obj: ", obj_ml)
    print("     Time:      ", time_ml)

    for k in args.baseline_k:
        print(f"\n  Baseline (K={k}):")
        print("    Obj:               ", obj_baseline[k])
        print("    Obj at ml time:    ", obj_at_ml_time[k])
        print(f"    Gap:              ", "{:2f}%".format(gaps[k]))
        print(f"    Gap at ml time:   ", "{:2f}%".format(gaps_at_ml_time[k]))
        print("    Time:              ", k_adapt_results[k]['time'])

    results = {
        'ml' : ml_results,
        'k_adapt_results' : k_adapt_results,
        'obj_ml' : obj_ml,
        'time_ml' : time_ml,
        'obj_kadapt' : obj_baseline,
        'obj_k_adapt_at_ml_time' : obj_at_ml_time,
        'gaps' : gaps,
        'gaps_at_ml_time' : gaps_at_ml_time,
        'baseline_k' : args.baseline_k,
    }

    # add problem specific evaluation results
    if "cb" in args.problem:
        results['inst_seed'] = args.cb_inst_seed
        results['n_items'] = args.cb_n_items

    with open(fp_res, 'wb') as p:
        pkl.dump(results, p)

    # store results
    print(f"\nStore evaluation results to {fp_res}")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluates row generation with tiny set network for knapsack problem.')

    # evaluation type
    parser.add_argument('--eval_method', type=str, default="ssa", choices=["pricing", "kadapt", "ssa", "ss_relax"])

    # instance parameters
    parser.add_argument('--problem', type=str, default="cb", help='Number of items [do not change, see ]')
    
    # CB specific parameters
    parser.add_argument('--cb_n_items', type=int, default=10, choices=[10, 20, 30, 40, 50, 60, 70, 80], help='Number of items [do not change, see ]')
    parser.add_argument('--cb_inst_seed', type=int, default=1)
    parser.add_argument('--cb_use_exact_cons', type=int, default=1, help='use exact uncertainty for constraint satisfaction (only for adversarial)')
    parser.add_argument('--cb_use_first_and_second', type=int, default=1, help='Indictor to use or not use item specific model')

    # type of ml model
    parser.add_argument('--model_type', type=str, default='nn_tiny_set', help='Number of items [do not change, see ]')

    # algorithm choices
    parser.add_argument('--time_limit', type=float, default=3 * 3600, help='Time limit for solving')
    parser.add_argument('--opt_type', type=str, default="adversarial", choices=["adversarial", "pga"], help='Type of sampling')
    parser.add_argument('--obj_type', type=str, default="argmax", choices=["argmax"])

    # baseline
    parser.add_argument('--baseline', type=str, default="k_adapt", choices=["k_adapt", "ccg"])
    parser.add_argument('--baseline_k', type=int, nargs="+", default=[2, 5, 10])

    # k-adapt choices

    # eval choices -- SSA
    parser.add_argument('--ssa_n_sample_scenarios', type=int, default=10000)
    parser.add_argument('--ssa_sample_scenario_seed', type=int, default=1234)    # should be something else than training seed

    # eval choices -- K-adapt
    parser.add_argument('--K_eval', type=int, default=20)
    parser.add_argument('--eval_kadapt_time_limit', type=int, default=60)

    # n_procs, only applicable for SSA for now, but might be useful later
    parser.add_argument('--n_procs', type=int, default=2)

    # output
    parser.add_argument('--verbose', type=int, default=0, help='Verbose param for optimization model')

    args = parser.parse_args()

    main(args)
