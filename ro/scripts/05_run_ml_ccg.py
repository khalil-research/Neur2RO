import os
import time
import copy

from joblib import Parallel, delayed

import collections
import argparse
import numpy as np
import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt
from functools import partial
from itertools import repeat

import gurobipy as gp
from gurobi_ml import add_predictor_constr

from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_squared_error as MSE

import torch
import torch.nn as nn
from torch.utils.data.dataset import TensorDataset
from torch.utils.data import DataLoader

import torch.multiprocessing as mpt
import multiprocessing as mp

import ro.params as ro_params
from ro.utils import factory_load_problem, factory_get_path
from ro.approximator import factory_approximator


# -------------------------------------------------------#
#                     Gurobi Callbacks                   #
# -------------------------------------------------------#

def no_new_inc_callback(model, where):
    """ Callback function to terminate if no new incumebnts are found in time. """
    if where == gp.GRB.Callback.MIPNODE:
        # get model objective
        obj = model.cbGet(gp.GRB.Callback.MIPNODE_OBJBST)

        # check if objective value has been updated
        if abs(obj - model._cur_obj) > 1e-8:
            # If so, update incumbent and time
            model._cur_obj = obj
            model._time = time.time()

    # terminate if objective has not improved in _inc_time
    if time.time() - model._time > model._inc_time:
        model.terminate()


def optimize_with_inc_callback(m, time_limit=10800):
    """ Does optimization with above callback.  Allows one to retrieve gurobi variables.  """
    m._cur_obj = float('inf')
    m._time = time.time()
    m.optimize(no_new_inc_callback)

    if m.status == 11:
        m.setParam("TimeLimit", 1)
        m.optimize()
        m.setParam("TimeLimit", time_limit)

    return m


# -------------------------------------------------------#
#              Scenario compartive functions             #
# -------------------------------------------------------#

def is_xi_equal(xi_1, xi_2, tol=1e-3):
    """ Samples n_samples uncertainty vectors.  """
    abs_diff = np.abs(np.array(xi_1) - np.array(xi_2))
    if np.sum(abs_diff) < tol:
        return True
    return False


def is_xi_in_list(xi, xi_list, tol=1e-3):
    """ Samples n_samples uncertainty vectors.  """
    is_eq = list(map(lambda xi_in_list: is_xi_equal(xi, xi_in_list), xi_list))
    if np.sum(is_eq) == 0:
        return False
    return True


# ----------------------------------------------#
#          Adversarial specific functions       #
# ----------------------------------------------#

def get_worst_case_scenario(args, approximator, x, time_limit):
    """ Solves adversarial problems to get worst-case scenario(s).
        Options to be added:
            - LP relaxation
            - PGA
            - More generally, other ways of solving the adversarial problem.
    """
    approximator.set_first_stage_in_adversarial_model(x)

    wc_xis = dict()
    wc_objs = dict()

    time_start = time.time()

    # objective optimization
    if "obj" in approximator.adv_model:
        # optimize for worst-case scenario and objective
        approximator.adv_model["obj"].setParam("TimeLimit", time_limit)
        approximator.adv_model["obj"] = optimize_with_inc_callback(approximator.adv_model["obj"])

        # recover worst-case scenario and objective
        wc_xis["obj"] = list(map(lambda x: x.x, approximator.adv_model["obj"]._xi.values()))
        wc_objs['obj'] = approximator.adv_model["obj"].objVal

        # round values close to 0/1 for kp
        if 'kp' in args.problem:
            for i in range(len(wc_xis["obj"])):
                if wc_xis["obj"][i] < 1e-6:
                    wc_xis["obj"][i] = 0
                elif wc_xis["obj"][i] > 1 - 1e-6:
                    wc_xis["obj"][i] = 1

        # round values close to -1/1 for cb
        elif 'cb' in args.problem:
            for i in range(len(wc_xis["obj"])):
                if wc_xis["obj"][i] < -1 + 1e-6:
                    wc_xis["obj"][i] = -1
                elif wc_xis["obj"][i] > 1 - 1e-6:
                    wc_xis["obj"][i] = 1

    # feasibility optimization
    if "feas" in approximator.adv_model:
        remaining_time = time_limit - (time.time() - time_start)
        approximator.adv_model["feas"].setParam("TimeLimit", remaining_time)

        approximator.adv_model["feas"] = optimize_with_inc_callback(approximator.adv_model["feas"])
        wc_xis["feas"] = list(map(lambda x: x.x, approximator.adv_model["feas"]._xi.values()))
        wc_objs["feas"] = approximator.adv_model["feas"].objVal

    return wc_xis, wc_objs


def adv_pga_single(args, x_in, s):
    # build optimizer
    rng = np.random.RandomState(s)
    xi = torch.autograd.Variable(approximator.to_tensor(approximator._sample_random_xi(rng, x_in)), requires_grad=True)

    approximator.net.train()
    Optimizer = getattr(torch.optim, "Adam")
    opt = Optimizer([xi], lr=args.pga_lr)

    best_xi = None
    best_obj = None
    for ep in range(args.pga_epochs):
        y_pred = approximator.do_forward_pass_adv(x_in, xi)
        y_pred = torch.squeeze(y_pred)
        loss = - y_pred
        loss.backward()
        opt.step()
        # project to space
        with torch.no_grad():
            approximator.clamp_xi(xi)
            if approximator.check_xi(xi, x_in):
                break
        if best_obj is None or best_obj < y_pred.detach().cpu():
            best_obj = float(y_pred.detach().cpu())
            best_xi = xi.clone().detach().numpy()

    return best_xi, best_obj


def adv_pga_batch(args, approximator, x):
    # build optimizer
    xis_np = np.array([approximator._sample_random_xi() for i in range(args.pga_samples)])
    xis_t = approximator.to_tensor(xis_np)
    dataset = TensorDataset(xis_t)
    loader = DataLoader(dataset, batch_size=args.pga_samples)
    # xi = torch.nn.parameter.Parameter(approximator.to_tensor(approximator._sample_random_xi()), requires_grad=True)
    approximator.net.train()
    Optimizer = getattr(torch.optim, "Adam")
    opt = Optimizer([xis_t], lr=args.pga_lr)

    best_xi = None
    best_obj = None
    for ep in range(args.pga_epochs):
        for xi in loader:
            approximator.net.train()
            y_pred = approximator.do_forward_pass_adv_batch(x, xi)
            y_pred = torch.squeeze(y_pred)
            loss = - y_pred
            loss.backward()
            opt.step()
            # project to space
            with torch.no_grad():
                approximator.clamp_xi(xi)
            approximator.net.eval()
            if best_obj is None or best_obj < y_pred.detach().cpu():
                best_obj = float(y_pred.detach().cpu())
                best_xi = xi.clone().detach().numpy()
    return best_xi, best_obj

def get_worst_case_scenario_pga(args, x, time_limit):
    """ Solves adversarial problems to get worst-case scenario(s).
        Options to be added:
            - LP relaxation
            - PGA
            - More generally, other ways of solving the adversarial problem.
    """
    time_start = time.time()

    x_in = approximator.to_tensor(x)
    # results = []
    # for s in range(args.pga_samples):
    #     res = adv_pga_single(args, x_in, s)
    #     results.append(res)

    approximator.net.share_memory()
    with mpt.Pool(processes=args.n_procs) as pool:
        results = pool.starmap(adv_pga_single, zip(repeat(args), repeat(x_in), range(args.pga_samples)))
    # adv_pga_single(args, x_in, 0)
    # objective optimization
    # if "obj" in approximator.adv_model:
    # # feasibility optimization
    # if "feas" in approximator.adv_model:
    pga_obj = [r[1] for r in results if r[1] is not None]
    pga_xi = [r[0] for r in results if r[0] is not None]
    wc_objs = {"obj": max(pga_obj)}
    wc_xis = {"obj": pga_xi[np.argmax(pga_obj)]}

    # round values close to 0/1 for kp
    if 'kp' in args.problem:
        for i in range(len(wc_xis["obj"])):
            if wc_xis["obj"][i] < 1e-6:
                wc_xis["obj"][i] = 0
            elif wc_xis["obj"][i] > 1 - 1e-6:
                wc_xis["obj"][i] = 1

    # round values close to -1/1 for cb
    elif 'cb' in args.problem:
        for i in range(len(wc_xis["obj"])):
            if wc_xis["obj"][i] < -1 + 1e-6:
                wc_xis["obj"][i] = -1
            elif wc_xis["obj"][i] > 1 - 1e-6:
                wc_xis["obj"][i] = 1

    # add feasibilty
    if "feas" in approximator.adv_model:

        # feasibility for CB
        if "cb" in args.problem:
            remaining_time = time_limit - (time.time() - time_start)
            approximator.adv_model["feas"].setParam("TimeLimit", remaining_time)

            approximator.adv_model["feas"] = optimize_with_inc_callback(approximator.adv_model["feas"])
            wc_xis["feas"] = list(map(lambda x: x.x, approximator.adv_model["feas"]._xi.values()))
            wc_objs["feas"] = approximator.adv_model["feas"].objVal

        # raise exception for non-CB problems
        else: 
            raise Exception("Not implemented for classes other than CB!")

    return wc_xis, wc_objs


def cb_get_exact_cons_wc(m, n_iterations):
    """ Gets worst-case feasiblilty based on c(xi)^T x for all scenaios added.  Specifically for capital budgeting. """
    wc_feas_constrs = []
    for i in range(n_iterations):
        wc_cons = m.getConstrByName(f"wc_feas_{i}")
        if wc_cons is not None:
            wc_feas_constrs.append(wc_cons)

    if len(wc_feas_constrs) == 0:
        return -gp.GRB.INFINITY

    # compute lhs = rhs - slack
    wc_feas_vals = list(map(lambda x: x.rhs - x.slack, wc_feas_constrs))

    return np.max(wc_feas_vals)


def get_wc_objs_from_main_problem(args, approximator, n_iterations):
    """ Gets worst-case objectives (over argmax) of the main problem. """
    objs_to_compare = {}

    # add objective worst-case from main problem
    objs_to_compare['obj'] = approximator.main_model._argmax_ya.x

    # add feasibility worst-case from main problem
    if approximator.has_feas_adv_model:
        if 'cb' in args.problem and args.cb_use_exact_cons:
            objs_to_compare['feas'] = cb_get_exact_cons_wc(approximator.main_model, n_iterations)
        else:
            objs_to_compare['feas'] = approximator.main_model._argmax_yf.x

    return objs_to_compare


def get_xi_to_add_to_main_problem(wc_xis, wc_objs, objs_to_compare, xi_vals, tol):
    """ Gets xi to add to main problem, i.e., checks termination criteria of ML-CCG.  Specifically, the termination criteria are as follows:
            - If the objective gap between the adv_model and main_model argmax are less than tol
            - If both the obj and feas scenarios are already in xi_added.  Not completely needed, but helps
              in some cases with numerical instability.
        If the returned dictionary is empty, then termination criteria are met.
    """
    xi_to_add = {}

    if 'obj' in wc_xis:
        gap = np.abs(wc_objs['obj'] - objs_to_compare['obj']) / np.abs(wc_objs['obj'])
        if gap > tol and not is_xi_in_list(wc_xis['obj'], xi_vals['obj']):
            xi_to_add['obj'] = wc_xis['obj']

    if 'feas' in wc_xis:
        gap = np.abs(wc_objs['feas'] - objs_to_compare['feas']) / np.abs(wc_objs['feas'])
        if gap > tol and not is_xi_in_list(wc_xis['feas'], xi_vals['feas']):
            xi_to_add['feas'] = wc_xis['feas']

    return xi_to_add


# ----------------------------------------------#
#                       CCG                     #
# ----------------------------------------------#

def run_ml_ccg(args, approximator):
    """ Does row generation algorithm until max_n_models are added or convergence. """
    n_iterations = 0

    sols = []
    times_wc = []
    times_data = []
    times_constr = []
    times_optimization = []

    time_start = time.time()

    # indicator variables for worst-case scenario
    xi_vals = {"obj": [], "feas": []}
    scen_id_vars = {"obj": [], "feas": []}

    if args.opt_type == "both":
        pga_adv_obj = {"pga_output": [], "adv_gurobi": [], "pga_pass": [], "adv_pass": [], "pga_time": [], "adv_time": []}
        pga_adv_xi = []

    while True:

        if args.time_limit < time.time() - time_start:
            print("TIME_LIMIT reached")
            break

        # optimize master problem
        time_opt = time.time()
        print(f"    Reoptimizing...")

        # set time limit of main problem
        opt_time_limit = args.time_limit - (time.time() - time_start)
        time_limit = min(opt_time_limit, args.mp_time)
        approximator.main_model.setParam("TimeLimit", time_limit)

        # optimize main problem
        approximator.main_model = optimize_with_inc_callback(approximator.main_model)
        
        if args.mp_save:
            approximator.main_model.write(f'models/main_{n_iterations}.lp')

        # Note: This may not be the best way of doing this,
        # but it is a safer option if the model timesout with
        # a bad quality incumbent.  Ideally, we can check both solutions
        # and provide the best one.
        if args.time_limit < time.time() - time_start:
            print("TIME_LIMIT reached")
            break

        time_opt = time.time() - time_opt
        times_optimization.append(time_opt)

        # recover first-stage solution from main problem
        x_in = list(map(lambda x: x.x, approximator.main_model._x.values()))
        sols.append(x_in)

        # print stuff
        print(f"\n  Main problem (Iteration # {n_iterations})")
        print(f"    x: {x_in}")
        # todo: add other stuff to print

        # check break condition
        if n_iterations > args.max_n_models:
            break

        time_wc = time.time()
        time_remaining = time_limit - (time.time() - time_start)

        if args.opt_type == "adversarial":
            # solve adversarial problem(s)
            wc_xis, wc_objs = get_worst_case_scenario(args, approximator, x_in, time_remaining)

            if args.adv_save:
                approximator.adv_model['obj'].write(f'models/adv_{n_iterations}.lp')

        elif args.opt_type == "pga":
            wc_xis, wc_objs = get_worst_case_scenario_pga(args, x_in, time_remaining)

        elif args.opt_type == "both": # algo follows exact sols of adversarial, but finds wc_xis_pga too
            # solve adversarial problem(s)
            time_remaining = time_limit - (time.time() - time_start)
            adv_st = time.time()
            wc_xis, wc_objs = get_worst_case_scenario(args, approximator, x_in, time_remaining)
            pga_adv_obj["adv_time"].append(time.time() - adv_st)
            pga_st = time.time()
            wc_xis_pga, wc_objs_pga = get_worst_case_scenario_pga(args, x_in, time_remaining)
            pga_adv_obj["pga_time"].append(time.time() - pga_st)
            pga_adv_obj["pga_output"].append(wc_objs_pga["obj"])
            pga_adv_obj["adv_gurobi"].append(wc_objs["obj"])
            pga_adv_xi.append(wc_xis_pga["obj"])
            x_in_t = approximator.to_tensor(x_in)
            adv_pass = approximator.do_forward_pass_adv(x_in_t, approximator.to_tensor(wc_xis["obj"])).detach().numpy()[0][0]
            pga_pass = approximator.do_forward_pass_adv(x_in_t, approximator.to_tensor(wc_xis_pga["obj"])).detach().numpy()[0][0]
            pga_adv_obj["adv_pass"].append(adv_pass)
            pga_adv_obj["pga_pass"].append(pga_pass)
            print(f"PGA XI: {wc_objs_pga['obj']}")
            print(f"ADV XI: {wc_objs['obj']}")

        # get objectives to compare to
        objs_to_compare = get_wc_objs_from_main_problem(args, approximator, n_iterations)

        xi_to_add = get_xi_to_add_to_main_problem(wc_xis, wc_objs, objs_to_compare, xi_vals, args.tol)

        # if no scenarios to add, then terminate
        if not xi_to_add:
            print("No scenarios to be added, terminating")
            break

        # add scenarios to list
        if "obj" in xi_to_add:
            xi_vals["obj"].append(xi_to_add["obj"])

        if "feas" in xi_to_add:
            xi_vals["feas"].append(xi_to_add["feas"])

        time_wc = time.time() - time_wc
        times_wc.append(time_wc)

        # print stuff
        print(f"  Adversarial problem (Iteration # {n_iterations})")
        if "obj" in xi_to_add:
            wc_xi_obj = xi_to_add["obj"]
            print(f"    xi  (A-obj): {wc_xi_obj}")
        else:
            print(f"    Feasible in objective")
        if "feas" in xi_to_add:
            wc_xi_feas = xi_to_add["feas"]
            print(f"    xi  (A-feas): {wc_xi_feas}")
        else:
            print(f"    Feasible in constraints")
        # todo: add other stuff to print

        # add scenarios to argmax scenarios to main model
        print(f"    Adding predictive constr to main problem...")
        time_constr = time.time()

        if args.obj_type == "argmax":
            wc_xi_in = dict()
            # add worst-case objective scenario
            if "obj" in xi_to_add:
                approximator.add_worst_case_scenario_to_main(xi_to_add["obj"], n_iterations, scen_type="obj")

            # add worst-case feasibility scenario
            if "feas" in xi_to_add:
                approximator.add_worst_case_scenario_to_main(xi_to_add["feas"], n_iterations, scen_type="feas")

            scen_id_vars = approximator.change_worst_case_scen(xi_to_add, scen_id_vars, xi_vals, n_iterations)

        time_constr = time.time() - time_constr
        times_constr.append(time_constr)

        n_iterations += 1

    time_total = time.time() - time_start

    opt_stats = {
        "detailed_times": {
            'wc': times_wc,
            'constr': times_constr,
            'optimization': times_optimization,
        },
        'time': time_total,
        'n_iterations': n_iterations,
        'xi_added': xi_vals,
        'sols': sols
    }

    x = opt_stats['sols'][-1]
    if args.opt_type == "both":
        p = f'data/{args.problem}/pga_info'
        if args.problem == "cb":
            inst_seed = f"N{args.cb_n_items}_I{args.cb_inst_seed}"
        elif args.problem == "kp":
            inst_seed = f"N{args.kp_n_items}_corr{args.kp_correlation}"
        f = p + f'/pga_adv_SA{args.pga_samples}_EP{args.pga_epochs}_LR{args.pga_lr}_{inst_seed}.pkl'
        os.makedirs(p, exist_ok=True)
        with open(f, "wb") as handle:
            pkl.dump({"obj": pga_adv_obj, "xi": pga_adv_xi}, handle)
    return x, opt_stats, xi_vals


# -----------#
#     Main   #
# -----------#

def get_result_str(args, p_type="solving_results", parent_dir="ml_ccg_results"):
    """ Gets results string for problem type. """
    if 'cb' in args.problem:
        result_str = f"{parent_dir}/{p_type}_"
        result_str += f"n-{args.cb_n_items}_s-{args.cb_inst_seed}"

    elif 'kp' in args.problem:
        result_str = f"{parent_dir}/{p_type}_"
        result_str += f"n-{args.kp_n_items}_"
        result_str += f"c-{args.kp_correlation}_"
        result_str += f"d-{args.kp_delta}_"
        result_str += f"g-{args.kp_h}_"
        result_str += f"c-{args.kp_budget_factor}_"
        result_str += f"c-{args.kp_R}_"
        result_str += f"c-{args.kp_H}"

    return result_str


def main(args):
    global cfg

    # approximator = CapitalBudgetingApproximator(args, cfg, net, inst_params)

    # initialize main and adversarial problems
    print("Initializing main and adversarial problems ... ")

    approximator.initialize_main_model(args)
    approximator.initialize_adversarial_model(args)

    # run row generation
    print("Running ML-CCG ... ")

    x, opt_stats, scens = run_ml_ccg(args, approximator)
   
    print("Done ML-CCG")

    # report results
    print("\nLearning-Based Algorithm Stats:")
    print("     Time (tot):   ", opt_stats['time'])
    print("     Time (adv):   ", np.sum(opt_stats['detailed_times']['wc']))
    print("     Time (mp):    ", np.sum(opt_stats['detailed_times']['optimization']))
    print("     # iterations: ", opt_stats['n_iterations'])
    print("     x:            ", x)

    # store results
    results = {
        'x': x,
        'algo_time': opt_stats['time'],
        'opt_stats': opt_stats,
        'opt_type': args.opt_type,
        'obj_type': args.obj_type,
    }

    if args.opt_type == "adversarial":
        result_str = get_result_str(args, parent_dir="ml_ccg_results")

    elif args.opt_type == "pga":
        result_str = get_result_str(args, parent_dir="ml_ccg_results_pga")

    elif args.opt_type == "both":
        result_str = get_result_str(args, parent_dir="ml_ccg_results_both")

    # save results
    fp_res = get_path(cfg.data_path, cfg, result_str)

    with open(fp_res, 'wb') as p:
        pkl.dump(results, p)

    print(f"\nDone\n  Solving results saved to: {fp_res}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluates row generation with tiny set network for knapsack problem.')

    parser.add_argument('--problem', type=str, default="cb", help='Problem to evaluate on.')
    parser.add_argument('--model_type', type=str, default='set_encoder', help='Type of machine learning model.  Only set_encoder is implemented')
    parser.add_argument('--argmax_ind', type=int, default=0, help='Indicator to use alternative argmax formulation.')

    # KP specific parameters
    parser.add_argument('--kp_n_items', type=int, default=20, choices=[20,30,40,50,60,70,80], help='Number of items [do not change, see ]')
    parser.add_argument('--kp_correlation', type=str, default='UN', choices=['UN', 'WC', 'ASC', 'SC'], help='Correlation type [do not change, see ]')
    parser.add_argument('--kp_delta', type=float, default=0.1, choices=[0.1, 0.5, 1],  help='Delta [do not change, see ]')
    parser.add_argument('--kp_h', type=int, default=40, choices=[40, 80], help='h [do not change, see ]')
    parser.add_argument('--kp_budget_factor', type=float, default=0.1, choices=[0.1, 0.15, 0.20], help='budget factor [do not change, see ]')
    parser.add_argument('--kp_R', type=int, default=1000, choices=[1000], help='h [do not change, see ]')
    parser.add_argument('--kp_H', type=float, default=100, choices=[100], help='budget factor [do not change, see ]')

    # CB specific parameters
    parser.add_argument('--cb_n_items', type=int, default=10, choices=[10, 20, 30, 40, 50, 60, 70, 80],
                        help='Number of items [do not change, see ]')
    parser.add_argument('--cb_inst_seed', type=int, default=1)
    parser.add_argument('--cb_use_exact_cons', type=int, default=1,
                        help='use exact uncertainty for constraint satisfaction (only for adversarial)')
    parser.add_argument('--cb_use_first_and_second', type=int, default=1,
                        help='Indicator to use or not use item specific model')

    # tolerance for termination of ML-CCG (gap between adv and main problems)
    parser.add_argument('--tol', type=float, default=1e-4, help='gap for termination')

    # gaps/times for master and adversarial problems
    parser.add_argument('--mp_gap', type=float, default=1e-3, help='gap for master problem')
    parser.add_argument('--mp_time', type=float, default=10800, help='time for master problem (per iteration)')
    parser.add_argument('--adversarial_gap', type=float, default=1e-3, help='gap for adversarial problem (only for adversarial)')
    parser.add_argument('--adversarial_time', type=float, default=10800, help='time for adversarial problem (only for adversarial)')
    parser.add_argument('--mp_inc_time', type=float, default=180, help='termination if no new incumbents are found for master')
    parser.add_argument('--adversarial_inc_time', type=float, default=180, help='termination if no new incumbents are found for adversarial')
    parser.add_argument('--mp_focus', type=int, default=1, help='MIPFocus for master')
    parser.add_argument('--adversarial_focus', type=int, default=1, help='MIPFocus for adversarial')
    parser.add_argument('--mp_log', type=int, default=0, help='Gurobi logging for master')
    parser.add_argument('--adv_log', type=int, default=0, help='Gurobi logging for adversarial')

    parser.add_argument('--mp_save', type=int, default=0, help='Save Guorbi model for MP')
    parser.add_argument('--adv_save', type=int, default=0, help='Save Gurobi model for AP')

    # algorithm decision
    parser.add_argument('--time_limit', type=float, default=3 * 3600, help='Time limit for solving')
    parser.add_argument('--opt_type', type=str, default="adversarial", choices=["adversarial", "pga", "both"], help='Type of worst-case.')
    parser.add_argument('--obj_type', type=str, default="argmax", choices=["argmax"], help='Type of objective.')
    parser.add_argument('--n_uncertainty_samples', type=int, default=10000, help='# of uncertainty samples')
    parser.add_argument('--pga_samples', type=int, default=5)
    parser.add_argument('--pga_epochs', type=int, default=100)
    parser.add_argument('--pga_lr', type=float, default=1e-3, help='Learning rate.')
    parser.add_argument('--n_procs', type=int, default=1, help='Number of processors')

    parser.add_argument('--use_lp_relax', type=int, default=0,
                        help='use LP relaxation adversarial problem (only for adversarial)')
    parser.add_argument('--max_n_models', type=int, default=20, help='Maximum # of models to add before early stopping')
    parser.add_argument('--seed', type=int, default=1,
                        help='Seed (only affects randomness in uncertainty for sampling based optimization)')
    # output
    parser.add_argument('--verbose', type=int, default=0, help='Verbose param for optimization model')

    args = parser.parse_args()
    
    # initialize approximator
    np.random.seed(args.seed)

    # initialize get_path
    get_path = factory_get_path(args.problem)

    # load config and paths
    cfg = getattr(ro_params, args.problem)
    fp_nn = get_path(cfg.data_path, cfg, args.model_type, suffix='.pt')

    # load pytorch model
    net = torch.load(fp_nn)

    print("Neur2RO optimization for")

    if "kp" in args.problem:
        print(f"   problem:              {args.problem}")
        print(f"   n_items:              {args.kp_n_items}")
        print(f"   correlation:          {args.kp_correlation}")
        print(f"   delta:                {args.kp_delta}")
        print(f"   h:                    {args.kp_h}")
        print(f"   budget_factor:        {args.kp_budget_factor}")
        print(f"   R:                    {args.kp_R}")
        print(f"   H:                    {args.kp_H}")

        inst_params = {
            'n_items': args.kp_n_items,
            'correlation': args.kp_correlation,
            'delta': args.kp_delta,
            'h': args.kp_h,
            'budget_factor': args.kp_budget_factor,
            'R': args.kp_R,
            'H': args.kp_H,
        }

    elif "cb" in args.problem:
        print(f"   problem:              {args.problem}")
        print(f"   n_items:              {args.cb_n_items}")
        print(f"   inst_seed:            {args.cb_inst_seed}")
        print(f"   use_exact_cons:       {args.cb_use_exact_cons}")
        print(f"   use_first_and_second: {args.cb_use_first_and_second}")

        inst_params = {'n_items': args.cb_n_items, 'inst_seed': args.cb_inst_seed}

    global approximator
    approximator = factory_approximator(args, cfg, net, inst_params)

    main(args)
