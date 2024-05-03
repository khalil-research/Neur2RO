import argparse

import time
import copy
import collections
import numpy as np
import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt

import gurobipy as gp
from gurobi_ml import add_predictor_constr

from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_squared_error as MSE

import torch
import torch.nn as nn
from torch.utils.data.dataset import TensorDataset
from torch.utils.data import DataLoader

import ro.params as ro_params
from ro.utils.kp import get_path
from ro.two_ro.kp import Knapsack



#---------------------------------------------------#
#   Functions to read baseline data and results     #
#---------------------------------------------------#

def read_paper_inst(cfg, inst, inst_dir):
    """ Load instances data from paper. """
    inst_name = f"RKP_{inst['correlation']}"
    inst_name += f"_n{inst['n_items']}"
    inst_name += f"_R{inst['R']}"
    inst_name += f"_H{inst['H']}"
    inst_name += f"_h{inst['h']}"
    inst_name += f"_dev{inst['budget_factor']}"

    if inst['delta'] == 1.0:
        inst_name += f"_d{int(inst['delta'])}"
    else:
        inst_name += f"_d{inst['delta']}"

    inst['inst_name'] = inst_name

    fp_prob = inst_dir + inst_name
    with open(fp_prob, 'r') as f:
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
    
    return inst


def get_baseline_results_from_csv(cfg, inst):
    inst_dir =  cfg.data_path + "kp/eval_instances/"
    inst = read_paper_inst(cfg, inst, inst_dir)

    result_dir = cfg.data_path + 'kp/baseline_results/ResultsForBranchAndPrice.csv'
    df = pd.read_csv(result_dir)

    inst_res = df.loc[inst['inst_name'] == df['File name']]
    inst_res = inst_res.to_dict()
    
    return inst, inst_res


#-------------------------------------------------------#
#                   Pytorch Functions                   #
#-------------------------------------------------------#
def to_tensor(x):
    return torch.Tensor(x).float()

#-------------------------------------------------------#
#   Uncertainty sampling for scenario sampling model    #
#-------------------------------------------------------#

def sample_random_xi(inst):
    """ Samples random continous uncertainty vectors for knapsack problem.  """
    budget_max = inst['max_budget']
    budget = np.random.uniform(1/budget_max, budget_max)
    xi = np.random.uniform(0, 1, size=inst['n_items'])
    xi = budget * xi / xi.sum()
    return xi


def sample_xis(inst, n_samples):
    """ Samples n_samples uncertainty vectors.  """
    xi_samples = []
    for i in range(n_samples):
        xi = list(sample_random_xi(inst))
        xi_samples.append(xi)
    return xi_samples


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

#-----------------------------------------------#
#   Embeddings of ML models to Gurobi models    #
#-----------------------------------------------#

def init_grb_inst_variables(m, inst, net):
    """ Gets gurobi input variables corresponding for set based inputs for knapsack.  """
    # get scalers
    feat_scaler = net.feat_scaler
    feat_min = feat_scaler[0][1:]   # skip x, xi
    feat_max = feat_scaler[1][1:]   # skip x, xi

    # initialize list to store variables
    inst_vars = []

    for idx in range(inst['n_items']):
        inst_vals = [inst['c'][idx], 
                     inst['p_bar'][idx], 
                     inst['p_hat'][idx], 
                     inst['f'][idx], 
                     inst['t'][idx], 
                     inst['C']]
    
        inst_vals = (np.array(inst_vals) - feat_min) / (feat_max - feat_min)
    
        c_var = m.addVar(name=f"c_{idx}", vtype="C")
        c_var.lb = inst_vals[0]
        c_var.ub = inst_vals[0]
              
        p_bar_var = m.addVar(name=f"p_bar_{idx}", vtype="C")
        p_bar_var.lb = inst_vals[1]
        p_bar_var.ub = inst_vals[1]

        p_hat_var = m.addVar(name=f"p_hat_{idx}", vtype="C")
        p_hat_var.lb = inst_vals[2]
        p_hat_var.ub = inst_vals[2]

        f_var = m.addVar(name=f"f_{idx}", vtype="C")
        f_var.lb = inst_vals[3]
        f_var.ub = inst_vals[3]
        
        t_var = m.addVar(name=f"t_{idx}", vtype="C")
        t_var.lb = inst_vals[4]
        t_var.ub = inst_vals[4]

        C_var = m.addVar(name=f"c_{idx}", vtype="C")
        C_var.lb = inst_vals[5]
        C_var.ub = inst_vals[5]

        inst_vars.append([c_var, p_bar_var, p_hat_var, f_var, t_var, C_var])
    
    return inst_vars


def embed_value_network(m, net, x_embed, xi_embed, n_iterations):
    """ Embeds value network in gurobi model. """
    xi_embed_var = m.addVars(net.xi_post_agg_dims[-1], name=f"xi_embed_{n_iterations}", lb=-gp.GRB.INFINITY, obj=0)
    for i in range(net.xi_post_agg_dims[-1]):
        xi_embed_var[i].lb = xi_embed[i]
        xi_embed_var[i].ub = xi_embed[i]

    pred_in = m._x_embed.select() + xi_embed_var.select()
    pred_out =  m.addVar(name=f"pred_{n_iterations}", lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, obj=0)
    pred_constr = add_predictor_constr(m, net.value_net, pred_in, pred_out, name=f"pred_constr_{n_iterations}")

    return m, pred_out


def embed_setbased_model(m, inst, gp_input_vars, set_net, agg_dim, post_agg_net, post_agg_dim, agg_type):
    """ Embeds set-based predictive models. """
    # add predictive constraints for each element of the set
    set_outputs = []
    for gp_input_var in gp_input_vars:
        pre_sum_vars = m.addVars(agg_dim, vtype="C", lb=-gp.GRB.INFINITY)
        pred_constr = add_predictor_constr(m, set_net, gp_input_var, pre_sum_vars)
        set_outputs.append(pre_sum_vars)

    # initaialize gurobi variables for post-summation
    post_agg_vars = m.addVars(agg_dim, vtype="C", lb=-gp.GRB.INFINITY)

    # add constraints to set: "post-agg variables == AGG(pre-agg variables)""
    for i in range(agg_dim):
        agg_sum = 0
        for j in range(inst['n_items']):
            agg_sum += set_outputs[j].select()[i]

        if agg_type == "sum":
            k = m.addConstr(agg_sum == post_agg_vars[i])
        elif agg_type == "mean":
            k = m.addConstr(agg_sum / inst['n_items'] == post_agg_vars[i])

    # post-agg net variable
    gp_embed_vars = m.addVars(post_agg_dim, vtype="C", lb=-gp.GRB.INFINITY)
    pred_constr = add_predictor_constr(m, post_agg_net, post_agg_vars, gp_embed_vars)

    return gp_embed_vars



#----------------------------------------------#
#   Adversarial scenario specific functions    #
#----------------------------------------------#

def get_adversarial_model(m, net, inst):
    """ Gets worst-case scenario by solving adversarial NN problem. """
    # initialize gurobi model
    m_adversarial = gp.Model()
    m_adversarial.setObjective(0, sense=gp.GRB.MAXIMIZE)
    m_adversarial.setParam("OutputFlag", 0)
        
    # define gurobi variables for x_embed vector
    x_embed_var = m_adversarial.addVars(net.x_post_agg_dims[-1], name="x_embed", vtype="C", lb=-gp.GRB.INFINITY)
    m_adversarial._x_embed = x_embed_var

    # define fixed variables for instance parameters
    inst_vars = init_grb_inst_variables(m_adversarial, inst, net)

    # define gurobi variables for xi values that are passed into the set network
    xi_var = m_adversarial.addVars(inst['n_items'], name="xi", vtype="C", lb=0, ub=1)
    m_adversarial._xi = xi_var

    # constraints on xi
    lhs = 0
    for xi in m_adversarial._xi.select():
        lhs += xi
    m_adversarial.addConstr(lhs <= inst['max_budget'], name='xi_budget_constr')

    # get gurobi input variables for xi embedding set based input
    gp_input_vars = []
    for i in range(inst['n_items']):
        gp_input_var = [xi_var[i]] + inst_vars[i]
        gp_input_vars.append(gp_input_var)

    # add xi embedding network to gurobi model
    xi_embed_var = embed_setbased_model(
        m = m_adversarial,
        inst = inst,
        gp_input_vars = gp_input_vars,
        set_net = net.xi_embed_net,
        agg_dim = net.xi_embed_dims[-1],
        post_agg_net = net.xi_post_agg_net, 
        post_agg_dim = net.xi_post_agg_dims[-1],
        agg_type = net.agg_type)

    m_adversarial._xi_embed = xi_embed_var

    # add predictive constraint
    pred_in = x_embed_var.select() + xi_embed_var.select()
    pred_out =  m_adversarial.addVar(name=f"pred", lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, obj=1)
    pred_constr = add_predictor_constr(m_adversarial, net.value_net, pred_in, pred_out)

    return m_adversarial


def set_first_stage_in_adversarial_model(m_adversarial, net, x, x_features, n_iterations):
    """ Sets adversarial model to x value. """
    n_items_tensor = to_tensor([len(x)])

    # change x in features to x parameter
    x_features_ = x_features.clone()
    x_features_[:,:,0] = to_tensor(x)

    # compute embeding of x
    x_embed = net.x_embed_net(x_features_)
    x_embed = net.agg_tensor(x_embed, n_items_tensor)
    x_embed = net.x_post_agg_net(x_embed).detach().cpu().numpy()[0]

    # set x_embedding values
    for i, x_embed_var in enumerate(m_adversarial._x_embed.select()):
        x_embed_var.lb = x_embed[i]
        x_embed_var.ub = x_embed[i]

    return m_adversarial



def cb_terminate_at_root(model, where):
    """ callback to terminate at root node.  """
    if where == gp.GRB.Callback.MIPNODE:
        print("MIPNODE")
        xi = model.cbGetNodeRel(model._xi)
        all_vars = model.cbGetNodeRel(model._all_vars)
       
        model._xi_at_root = xi
        model._vars_at_root = all_vars
        model.terminate()



def get_scenario_worst_case_adversarial(m_adversarial, net, x, x_features, xi_features, label_scaler, adversarial_gap, adversarial_time, use_lp_relax, n_iterations):
    """ Gets worst-case scenario by solving adversarial NN problem. """
    # set first-stage decisions to x
    m_adversarial = set_first_stage_in_adversarial_model(m_adversarial, net, x, x_features, n_iterations)
    
    m_adversarial.setParam("OutputFlag", 0)
    m_adversarial.setParam("MIPGap", adversarial_gap)
    m_adversarial.setParam("TimeLimit", adversarial_time)

    # optimize
    if use_lp_relax:
        m_adversarial._all_vars = m_adversarial.getVars()
        m_adversarial.optimize(cb_terminate_at_root)
    else:
        m_adversarial.optimize()
    
    # get xi values
    if use_lp_relax:
        wc_xi = list(m_adversarial._xi_at_root.select())
    else:
        wc_xi = list(map(lambda x: x.x, m_adversarial._xi.select()))

    # get objective
    if use_lp_relax:
        # compute forward pass on scenario
        n_item_tensor = to_tensor([len(x)])

        x_features_ = x_features.clone()
        x_features_[:,:,0] = to_tensor(x)

        xi_features_ = xi_features.clone()
        xi_features_[:,:,0] = to_tensor(wc_xi)

        wc_obj_scaled = net(x_features_, xi_features_, n_item_tensor).detach().cpu().numpy()[0][0]
    else:
        wc_obj_scaled = m_adversarial.objVal  

    # scale objective 
    wc_obj = wc_obj_scaled * (label_scaler[1] - label_scaler[0]) + label_scaler[0]

    # round values close to 0/1
    for i in range(len(wc_xi)):
        if wc_xi[i] < 1e-6:
            wc_xi[i] = 0
        elif wc_xi[i] > 1 - 1e-6:
            wc_xi[i] = 1

    return wc_xi, wc_obj




#-------------------------------------------#
#   Scenario sampling specific functions    #
#-------------------------------------------#

def get_initial_features_forward_pass(xi_samples, inst, feat_scaler):
    """ Gets features from x (input later), xi (scenario sampling), and inst. """
    x_features = []
    xi_features = []

    x = [0] * inst['n_items'] # zeros for now, changed before forward pass

    # loop over each xi in xi_vals
    for xi in xi_samples:
        inst_feats = list(map(lambda i: [inst['c'][i], 
                        inst['p_bar'][i], 
                        inst['p_hat'][i], 
                        inst['f'][i], 
                        inst['t'][i],  
                        inst['C']], range(inst['n_items'])))
        
        # create for each item in SetBased model
        x_feats = []
        xi_feats = []
        for j in range(inst['n_items']):
            x_feats_tmp = [x[j]] + inst_feats[j]
            xi_feats_tmp = [xi[j]] + inst_feats[j]
            x_feats.append(x_feats_tmp)
            xi_feats.append(xi_feats_tmp)
            
        x_features.append(x_feats)
        xi_features.append(xi_feats)
        
    x_features = np.array(x_features)  
    xi_features = np.array(xi_features)  

    # scale features
    x_features = (x_features - feat_scaler[0]) / (feat_scaler[1] - feat_scaler[0])
    xi_features = (xi_features - feat_scaler[0]) / (feat_scaler[1] - feat_scaler[0])

    # get features as tensor
    x_features = to_tensor(x_features)
    xi_features = to_tensor(xi_features)


    return x_features, xi_features


def get_scenario_worst_case_sampling(x, xi_samples, net, x_features, xi_features, label_scaler):
    """ Gets worst-case scenario by for sampling based problem with forward pass with NN and scenarios. """
    n_items = to_tensor([len(x)])

    # set x in x_features
    x_features_ = torch.clone(x_features)
    n_samples = x_features_.shape[0]
    x_stacked = torch.stack([to_tensor(x)] * n_samples)
    x_features_[:,:,0] = x_stacked

    # forward pass
    preds = net(x_features_, xi_features, n_items, n_items)
    preds = preds.detach().cpu().numpy()

    # unscale predictions
    preds = preds * (label_scaler[1] - label_scaler[0]) + label_scaler[0]
        
    # otherwise, get most violated scenario and add cuts
    max_violated_idx = np.argmax(preds)
    max_violated_xi = xi_samples[max_violated_idx]
    max_violated_obj = np.max(preds)

    return max_violated_xi, max_violated_obj, max_violated_idx




#-------------------------------#
#   Row generation functions    #
#-------------------------------#

def run_row_generation(args, m, net, inst, max_n_models, time_limit=3*3600):
    """ Does row generation algorithm until max_n_models are added or convergence. """
    n_iterations = 0
    xi_added = []
    
    gaps = []
    sols = []
    times_wc = []
    times_data = []
    times_constr = []
    times_optimization = []

    # get input/output scalers
    label_scaler = net.label_scaler[inst['n_items']]
    feat_scaler = net.feat_scaler

    if args.opt_type == "adversarial":
        m_adversarial = get_adversarial_model(m, net, inst)

        # get {x,xi}_features for computing forward passes in embeddings 
        xi_zero = [[0] * inst['n_items']]
        x_features, xi_features = get_initial_features_forward_pass(xi_zero, inst, feat_scaler)

    elif args.opt_type == "sampling":
        print("\nSampling scenarios")

        time_sample = time.time()
        xi_samples = sample_xis(inst, args.n_uncertainty_samples)
        x_features, xi_features = get_initial_features_forward_pass(xi_samples, inst, feat_scaler)
        time_sample = time.time() - time_sample

        print(f"  Done in {time_sample} seconds\n")

    init_xi_features = xi_features[0]
    init_xi_features = init_xi_features.reshape(
        1, init_xi_features.shape[0], init_xi_features.shape[1])

    time_total = time.time()

    # indicator variables for worst-case scenario
    xi_vals = []
    scen_vars = []
    scen_id_vars = []

    scen_LB, scen_UB = -1, 2 # data normalized to [0,1] so should be reasonable

    while True:

        opt_time_limit = args.time_limit - (time.time() - time_total)
        if args.time_limit < time.time() - time_total:
            print("TIME_LIMIT reached")
            break

        ## Optimization
        time_opt = time.time()
        print(f"    Reoptimizing...")

        time_limit = min([opt_time_limit, args.mp_time])
        m.setParam("TimeLimit", time_limit)
        m.setParam("MIPGap", args.mp_gap)
        m.optimize()
        
        #print("\n\nARGMAX:", m._argmax_w)
        #print("xi_worst_case:", m._xi_worst_case, "\n\n")
        # Note: This may not be the best way of doing this,
        # but it is a safer option if the model timesout with
        # a bad quality incumbent.  Ideally, we can check both solutions
        # and provide the best one.
        if args.time_limit < time.time() - time_total:
            print("TIME_LIMIT reached")
            break

        time_opt = time.time() - time_opt
        times_optimization.append(time_opt)

        x_in = list(map(lambda x: x.x, m._x.select()))
        sols.append(x_in)

        ## Check break condition
        if n_iterations > max_n_models:
            break
        
        ## Find worst-case scenario with forward pass, then add corresponding constraints
        print(f"  Adding cutting plane #{n_iterations+1}")
                
        # create dataset for pass
        time_wc = time.time()
        
        if args.opt_type == "adversarial":
            wc_xi, wc_obj = get_scenario_worst_case_adversarial(m_adversarial, net, x_in, x_features, init_xi_features, label_scaler, args.adversarial_gap, args.adversarial_time, args.use_lp_relax, n_iterations)

        elif args.opt_type == "sampling":
            wc_xi, wc_obj, wc_idx = get_scenario_worst_case_sampling(x_in, xi_samples, net, x_features, xi_features, label_scaler)

        xi_vals.append(wc_xi)
        time_wc = time.time() - time_wc
        times_wc.append(time_wc)

        print(f"    x: {x_in}")
        print(f"    z:        {m._z.x}")
        print(f"    obj (wc): {wc_obj}")
        print(f"    xi  (wc): {wc_xi}")

        # check if no violated scenarios
        if args.obj_type == "max":
            # for max of valuenet outputs, we only require the max to be at least the wc_obj from scenarios
            gap = np.abs(m._z.x - wc_obj) / np.abs(wc_obj)
            gaps.append(gap)
            print(f"    gap between worst-case and argmax:   {gap}")
            if (gap <= args.tol):
               print("  No more violated scenarios!")
               break

        elif args.obj_type == "argmax":
            # for lp-based objective, we check that the argmax is at least the wc_obj from scenarios
            value_net_max = m._argmax_y.x
            value_net_max = value_net_max * (label_scaler[1] - label_scaler[0]) + label_scaler[0]
            gap = np.abs(value_net_max - wc_obj) / np.abs(wc_obj)
            gaps.append(gap)

            print(f"    argmax of gp model (valuenet):           {value_net_max}")
            if args.opt_type == "sampling":
                print(f"    max of forward pass:                 {wc_obj}")
            elif args.opt_type == "adversarial":
                print(f"    max of adversarial problem:          {wc_obj}")

            print(f"    gap between worst-case and argmax:   {gap}")
            if (gap <= args.tol) and n_iterations > 0:
               print("  No more violated scenarios!")
               break

        if n_iterations > 0 and is_xi_in_list(wc_xi, xi_added):
            print(f"  Attempting to add scenario already in list.")
            print(f"        Terminating with gap of {gap}%")
            break

        print(f"    Adding predictive constr...")
        time_constr = time.time()

        # get input for worst-case scenario network
        n_item_tensor = to_tensor([inst['n_items']])
        wc_xi_feats = init_xi_features.clone()
        wc_xi_feats[:, :, 0] = to_tensor(wc_xi)

        xi_embed = net.xi_embed_net(wc_xi_feats)
        xi_embed = net.agg_tensor(xi_embed, n_item_tensor)
        xi_embed = net.xi_post_agg_net(xi_embed).detach().cpu().numpy()[0]

        # add predictive contraints for worst-case scenario
        m, z_scen = embed_value_network(m, net, m._x_embed, xi_embed, n_iterations)
        z_scen_scaled = m.addVar(name=f"z_sc_{n_iterations}", lb=-gp.GRB.INFINITY)

        scen_vars.append(z_scen)
        m.addConstr((z_scen_scaled - label_scaler[0]) / (label_scaler[1] - label_scaler[0]) == z_scen, name=f"pc_sc_{n_iterations}")

        # add constraint to determine worst-case scenario
        if args.obj_type == "argmax":
            # remove constraints from previous argmax
            m.remove(m.getConstrByName("scen_sum"))
            m.remove(m.getConstrByName("scen_idx_sum"))
            for j in range(inst["n_items"]):
                m.remove(m.getConstrByName(f"xi_wc_constr_{j}"))

            # add argmax constraints
            scen_id_var = m.addVar(name=f"scen_id_{n_iterations}", vtype="B", obj=0)
            scen_id_vars.append(scen_id_var)

            m.addConstr(m._argmax_y >= z_scen)
            m.addConstr(m._argmax_y <= z_scen + (scen_UB - scen_LB) * (1 - scen_id_var))
            m.addConstr(sum(scen_id_vars) == 1, name="scen_sum")
            m.addConstr(gp.quicksum(i * scen_id_vars[i] for i in range(len(scen_id_vars))) == m._argmax_w, name='scen_idx_sum')

            # set xi_worst case as linear combination
            for j in range(inst["n_items"]):
                lhs = gp.quicksum(xi_vals[i][j] * scen_id_vars[i] for i in range(len(scen_id_vars)))
                m.addConstr(lhs ==  m._xi_worst_case[j], name=f"xi_wc_constr_{j}")

        elif args.obj_type == "max":
            m.addConstr(m._z >= z_scen_scaled, name=f"pc_bound_{n_iterations}")

        time_constr = time.time() - time_constr
        times_constr.append(time_constr)

        n_iterations += 1
        xi_added.append(wc_xi)
        gaps.append(gap)
        
    time_total = time.time() - time_total

    opt_stats = {
        "detailed_times" : {
            'wc' : times_wc,
            'constr' : times_constr,
            'optimization' : times_optimization,
        },
        'time' : time_total,
        'n_iterations' : n_iterations,
        'xi_added' : xi_added,
        'sols' : sols,
        'gaps' : gaps
    }

    x = opt_stats['sols'][-1]

    return m, x, opt_stats



#-------------------------------#
#   Master problem functions    #
#-------------------------------#

def build_optimization_model(cfg, args, inst, knapsack, net):
    """ Builds Gurobi model of the master problem.   """
    # Initialize gurobi model
    m = gp.Model()
    m.setParam("OutputFlag", args.verbose)
        
    # define first-stage gurobi variables
    x_var = m.addVars(inst['n_items'], name="x", vtype="B", obj=inst['f'] - inst['p_bar'])
    m._x = x_var
    m._inst_vars = init_grb_inst_variables(m, inst, net)

    # type of second-stage cost function [set_encoder, nn_p, lp]
    if args.obj_type == "max":
        z_var = m.addVar(name="z", vtype="C", lb=-1e10, obj=1)
        m._z = z_var

    elif args.obj_type == "argmax":
        # variables for second-stage
        y_var = m.addVars(inst["n_items"], name="y", vtype="B")
        r_var = m.addVars(inst["n_items"], name="r", vtype="B")
        m._y = y_var
        m._r = r_var

        # second-stage constraints
        lhs = 0
        for i in range(inst['n_items']):
            lhs += inst['c'][i] * y_var[i] + inst['t'][i] * r_var[i]
        m.addConstr(lhs <= inst['C'])

        for i in range(inst['n_items']):
            m.addConstr(y_var[i] <= x_var[i])
            m.addConstr(r_var[i] <= y_var[i])

        # variable for objective
        z_var = m.addVar(name="z", vtype="C", lb=-gp.GRB.INFINITY, obj=1)
        m._z = z_var

        # variables for argmax
        argmax_y_var = m.addVar(name="argmax_y", vtype="C", lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, obj=0)
        argmax_w_var = m.addVar(name="argmax_w", vtype="C", lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, obj=0)
        m._argmax_y = argmax_y_var
        m._argmax_w = argmax_w_var

        # variables for worst-case scenario
        xi_worst_case = m.addVars(inst["n_items"], name="xi_worse_case", vtype="C", lb=0, ub=1, obj=0)
        m._xi_worst_case = xi_worst_case

        #m.addConstr(0 == 0, name="xi_wc_constr")    # dummy constraints to be overwritten later
        m.addConstr(0 == 0, name="scen_sum")        # dummy constraints to be overwritten later
        m.addConstr(0 == 0, name="scen_idx_sum")    # dummy constraints to be overwritten later
        for j in range(inst["n_items"]):
            m.addConstr(0 == 0, name=f"xi_wc_constr_{j}")

        # constraint to bound second-stage lp objective
        lhs = 0
        for i in range(inst["n_items"]):
            lhs += y_var[i] * ((inst['p_hat'][i] * xi_worst_case[i]) - inst['f'][i])
            lhs += r_var[i] * (inst['p_hat'][i] * xi_worst_case[i])
        m.addConstr(lhs <= z_var, name='wc_obj_constr')

    # add x embedding network gurobi variables
    x_gp_input_vars = []
    for i in range(inst['n_items']):
        input_vars = [m._x[i]] + m._inst_vars[i]
        x_gp_input_vars.append(input_vars)

    x_embed_var = embed_setbased_model(
        m = m,
        inst = inst,
        gp_input_vars = x_gp_input_vars,
        set_net = net.x_embed_net,
        agg_dim = net.x_embed_dims[-1],
        post_agg_net = net.x_post_agg_net, 
        post_agg_dim = net.x_post_agg_dims[-1],
        agg_type = net.agg_type)

    m._x_embed = x_embed_var

    return m




#-----------#
#   Main    #
#-----------#

def main(args):
    
    np.random.seed(args.seed)

    # load config and paths
    cfg = getattr(ro_params, args.problem)
    fp_prob = get_path(cfg.data_path, cfg, "problem")
    fp_data = get_path(cfg.data_path, cfg, "ml_data")
    fp_nn = get_path(cfg.data_path, cfg, args.model_type, suffix='.pt')
    
    # load instances
    with open(fp_prob, 'rb') as p:
        problem = pkl.load(p)

    # init two_ro class
    knapsack = Knapsack()

    # load pytorch model
    net_ = torch.load(fp_nn)

    # get gurobi-ml compatible pytorch model
    net = copy.deepcopy(net_)
    net = net.eval()
    net.x_embed_net = net.get_grb_compatible_nn(net.x_embed_layers)
    net.x_post_agg_net = net.get_grb_compatible_nn(net.x_post_agg_layers)
    net.xi_embed_net = net.get_grb_compatible_nn(net.xi_embed_layers)
    net.xi_post_agg_net = net.get_grb_compatible_nn(net.xi_post_agg_layers)
    net.value_net = net.get_grb_compatible_nn(net.value_layers)

    # print instance info
    print(f"Optimizing and Evaluating for knapsack instance with:")
    print(f"    n_items={args.n_items}")
    print(f"    correlation={args.correlation}")
    print(f"    delta={args.delta}")
    print(f"    h={args.h}")
    print(f"    budget_factor={args.budget_factor}\n")
    
    # set parameters for instance
    problem['n_items'] = args.n_items
    problem['correlation'] = args.correlation
    problem['delta'] = args.delta
    problem['h'] = args.h
    problem['budget_factor'] = args.budget_factor
    problem['max_budget'] = problem['n_items'] * problem['budget_factor']
    
    # get baseline results
    base_inst, baseline_results = get_baseline_results_from_csv(cfg, problem)
    baseline_res = {
        'obj' : list(baseline_results['Best primal bound'].values())[0],
        'time' : list(baseline_results['Time'].values())[0],
    }

    # build optimization model
    m = build_optimization_model(cfg, args, base_inst, knapsack, net)

    # do row generation
    m, x, opt_stats = run_row_generation(args, m, net, base_inst, args.max_n_models, args.time_limit)

    # evaluate solution
    obj, xi, y, r = knapsack.evaluate_first_stage_solution(x, base_inst)
    
    # compute optimality gap
    try:
        gap = 100 * (baseline_res['obj'] - obj)/baseline_res['obj']
    except:
        print("\n\n No solution found by ml algo! \n\n")
        gap = - 1e5
        
    # report results
    print("\n  Learning-Based Algorithm:")
    print("     Objective: ", obj)
    print("     Time:      ", opt_stats['time'])

    print("  Baseline:")
    print("    Objective: ", baseline_res['obj'])
    print("    Time:      ", baseline_res['time'])

    print("  Gap:", "{:2f}%".format(gap))

    # store results
    results = {
        'gap' : gap,
        'algo_obj' : obj,
        'algo_time' : opt_stats['time'],
        'baseline_obj' : baseline_res['obj'],
        'baseline_time' : baseline_res['time'],
        'inst_name' : base_inst['inst_name'],
        'opt_stats' : opt_stats,
        'opt_type' : args.opt_type,
    }

    result_str = "results_"
    result_str += f"n-{args.n_items}_"
    result_str += f"c-{args.correlation}_"
    result_str += f"d-{args.delta}_"
    result_str += f"h-{args.h}_"
    result_str += f"b-{args.budget_factor}_"
    result_str += f"opt-{args.opt_type}_"
    result_str += f"obj-{args.obj_type}"

    # save results
    fp_res = get_path(cfg.data_path, cfg, 'appendix_results/' + result_str)

    with open(fp_res, 'wb') as p:
        pkl.dump(results, p)

    print("\nDONE")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script for evaluating kp set encoder in varying set-encoder instances.')

    parser.add_argument('--problem', type=str, default="kp", help='Number of items [do not change, see ]')
    parser.add_argument('--model_type', type=str, default='set_encoder', help='Number of items [do not change, see ]')

    # optimization type
    parser.add_argument('--opt_type', type=str, default="sampling", choices=["sampling", "adversarial"], help='Type of sampling')
    parser.add_argument('--obj_type', type=str, default="argmax", choices=["argmax", "max"], help='Type of objective')

    # instance parameters
    parser.add_argument('--n_items', type=int, default=20, choices=[20,30,40,50,60,70,80], help='Number of items [do not change, see ]')
    parser.add_argument('--correlation', type=str, default='UN', choices=['UN', 'WC', 'ASC', 'SC'], help='Correlation type [do not change, see ]')
    parser.add_argument('--delta', type=float, default=0.1, choices=[0.1, 0.5, 1],  help='Delta [do not change, see ]')
    parser.add_argument('--h', type=int, default=40, choices=[40, 80], help='h [do not change, see ]')
    parser.add_argument('--budget_factor', type=float, default=0.1, choices=[0.1, 0.15, 0.20], help='budget factor [do not change, see ]')

    # algorithm optimization gaps/time
    parser.add_argument('--mp_gap', type=float, default=1e-4, help='gap for main problem')
    parser.add_argument('--mp_time', type=float, default=180, help='time for main problem (per iteration)')
    parser.add_argument('--adversarial_gap', type=float, default=1e-4, help='gap for adversarial problem (only for adversarial)')
    parser.add_argument('--adversarial_time', type=float, default=180, help='time for adversarial problem (only for adversarial)')

    # algorithm decision 
    parser.add_argument('--time_limit', type=float, default=3*3600, help='Time limit for solving')
    parser.add_argument('--n_uncertainty_samples', type=int, default=10000, help='# of uncertainty samples (only for sampling)')
    parser.add_argument('--max_n_models', type=int, default=20, help='Maximum # of models to add before early stopping')
    parser.add_argument('--use_lp_relax', type=int, default=0, help='Use relaxation for adversarial problem')
    parser.add_argument('--seed', type=int, default=1, help='Seed (only affects randomness in uncertainty for sampling based optimization)')
    parser.add_argument('--tol', type=float, default=1e-6, help='Tolerance for termination')

    # output
    parser.add_argument('--verbose', type=int, default=0, help='Verbose param for optimization model')

    args = parser.parse_args()

    main(args)