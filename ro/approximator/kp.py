import time
import copy
import collections
import numpy as np
import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt

import gurobipy as gp
from gurobi_ml import add_predictor_constr

import torch
import torch.nn as nn

import ro.params as ro_params

from .approximator import Approximator

import ro.dm.kp as dm


class KnapsackApproximator(Approximator):

    def __init__(self, args, cfg, net, inst_params):
        """ Constructor for Knapsack Aproximator.  """
        self.cfg = cfg

        # initialize instance
        self.inst = self.get_instance(inst_params)

        self.n_items = self.inst['n_items']

        # get scalers
        self.feat_scaler = net.feat_scaler
        self.label_scaler = net.label_scaler[self.n_items]

        # initialize nn
        self.net = self.initialize_nn(net)

        # get features for forward pass
        self.x_features, self.xi_features = self.get_inst_nn_features()

        self.has_feas_adv_model = False


    def initialize_main_model(self, args):
        """ Gets the gurobi model for the main problem. """
        # Initialize gurobi model
        m = gp.Model()
        m.setParam("OutputFlag", args.verbose)
        m.setParam("MIPGap", args.mp_gap)
        m.setParam("TimeLimit", args.mp_time)
        m.setParam("MIPFocus", args.mp_focus)
        m._inc_time = args.mp_inc_time
            
        m._pred_out = {"obj": [], "feas": []}

        # define first-stage gurobi variables
        x_var = m.addVars(self.n_items, name="x", vtype="B", obj=self.inst['f'] - self.inst['p_bar'])
        m._x = x_var
        m._inst_vars = self.init_grb_inst_variables(m)

        if args.obj_type == "argmax":
            # variables for second-stage
            y_var = m.addVars(self.n_items, name="y", vtype="B")
            r_var = m.addVars(self.n_items, name="r", vtype="B")
            m._y = y_var
            m._r = r_var

            # second-stage constraints
            lhs = 0
            for i in range(self.n_items):
                lhs += self.inst['c'][i] * y_var[i] + self.inst['t'][i] * r_var[i]
            m.addConstr(lhs <= self.inst['C'], name="budget")

            for i in range(self.n_items):
                m.addConstr(y_var[i] <= x_var[i], name=f"yx_{i}")
                m.addConstr(r_var[i] <= y_var[i], name=f"ry_{i}")

            # variable for objective
            z_var = m.addVar(name="z", vtype="C", lb=-gp.GRB.INFINITY, obj=1)
            m._z = z_var

            # variables for argmax
            argmax_y_var = m.addVar(name="argmax_y", vtype="C", lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, obj=0)
            argmax_w_var = m.addVar(name="argmax_w", vtype="C", lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, obj=0)
            m._argmax_ya = argmax_y_var
            m._argmax_wa = argmax_w_var

            # variables for worst-case scenario
            xi_worst_case = m.addVars(self.n_items, name="xi_worse_case", vtype="C", lb=0, ub=1, obj=0)
            m._xi_worst_case = xi_worst_case

            #m.addConstr(0 == 0, name="xi_wc_constr")   # dummy constraints to be overwritten later
            m.addConstr(0 == 0, name="scen_sum")        # dummy constraints to be overwritten later
            m.addConstr(0 == 0, name="scen_idx_sum")    # dummy constraints to be overwritten later
            for j in range(self.n_items):
                m.addConstr(0 == 0, name=f"xi_wc_constr_{j}")

            # constraint to bound second-stage lp objective
            lhs = 0
            for i in range(self.n_items):
                lhs += y_var[i] * ((self.inst['p_hat'][i] * xi_worst_case[i]) - self.inst['f'][i])
                lhs += r_var[i] * (self.inst['p_hat'][i] * xi_worst_case[i])
            m.addConstr(lhs <= z_var, name='wc_obj_constr')

        else:
            raise Exception("Other types of objectives not implemented in class, only argmax.")

        # add x embedding network gurobi variables
        x_gp_input_vars = []
        for i in range(self.n_items):
            input_vars = [m._x[i]] + m._inst_vars[i]
            x_gp_input_vars.append(input_vars)

        x_embed_var = self.embed_setbased_model(
            m = m,
            gp_input_vars = x_gp_input_vars,
            set_net = self.net.x_embed_net,
            agg_dim = self.net.x_embed_dims[-1],
            post_agg_net = self.net.x_post_agg_net, 
            post_agg_dim = self.net.x_post_agg_dims[-1],
            agg_type = self.net.agg_type,
            name="x_embed")

        m._x_embed = x_embed_var

        self.main_model = m


    def initialize_adversarial_model(self, args):
        """ Gets the gurobi model(s) for the adversarial problem. """
        m_adv_obj = self._get_adv_obj(args)
        self.adv_model = {'obj' : m_adv_obj}


    def initialize_nn(self, net_):
        """ Initializes neural network for optimization.  """
        net = copy.deepcopy(net_)
        net = net.eval()
        net = net.cpu()
        net.x_embed_net = net.get_grb_compatible_nn(net.x_embed_layers)
        net.x_post_agg_net = net.get_grb_compatible_nn(net.x_post_agg_layers)
        net.xi_embed_net = net.get_grb_compatible_nn(net.xi_embed_layers)
        net.xi_post_agg_net = net.get_grb_compatible_nn(net.xi_post_agg_layers)
        net.value_net = net.get_grb_compatible_nn(net.value_layers)
        return net


    def init_grb_inst_variables(self, m):
        """ Initialize gurobi variables for problem features (i.e., input to NN).  """
        # get scalers
        feat_min = self.feat_scaler[0][1:]   # skip x, xi
        feat_max = self.feat_scaler[1][1:]   # skip x, xi

        # initialize list to store variables
        inst_vars = []

        for idx in range(self.n_items):
            inst_vals = [self.inst['c'][idx], 
                         self.inst['p_bar'][idx], 
                         self.inst['p_hat'][idx], 
                         self.inst['f'][idx], 
                         self.inst['t'][idx], 
                         self.inst['C']]
        
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


    def embed_value_network(self, xi_embed, n_iterations, scen_type):
        """ Embeds value network in gurobi model. """
        xi_embed_dim = xi_embed.shape[0]

        xi_embed_var = self.main_model.addVars(xi_embed_dim, name=f"xi_embed_{n_iterations}", lb=-gp.GRB.INFINITY, obj=0)
        for i in range(xi_embed_dim):
            xi_embed_var[i].lb = xi_embed[i]
            xi_embed_var[i].ub = xi_embed[i]

        pred_in = self.main_model._x_embed.select() + xi_embed_var.select()
        pred_out =  self.main_model.addVar(name=f"pred_{n_iterations}", lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, obj=0)
        pred_constr = add_predictor_constr(self.main_model, self.net.value_net, pred_in, pred_out, name=f"pred_constr_{n_iterations}_{scen_type}")
        
        return pred_out


    def add_worst_case_scenario_to_main(self, xi, n_iterations, scen_type):
        """ Adds worst-case scenario(s) to main problem. """        
        xi_embed = self.get_xi_embed(xi)

        # add predictive contraints for worst-case scenario to the main problem
        z_scen = self.embed_value_network(xi_embed, n_iterations, scen_type=scen_type)

        self.main_model._pred_out[scen_type].append(z_scen)


    def change_worst_case_scen(self, xi_to_add, scen_id_vars, xi_vals, n_iterations):
        """ Changes worst-case scenario constraints in main problem. """
        obj_LB, obj_UB = -10, 10

        # add scenario for objective
        if "obj" in xi_to_add:  

            # remove constraints from previous argmax
            self.main_model.remove(self.main_model.getConstrByName("scen_sum"))
            self.main_model.remove(self.main_model.getConstrByName("scen_idx_sum"))
            for j in range(self.n_items):
                self.main_model.remove(self.main_model.getConstrByName(f"xi_wc_constr_{j}"))

            # add argmax variables
            scen_id_var_a = self.main_model.addVar(name=f"scen_id_{n_iterations}", vtype="B", obj=0)
            scen_id_vars["obj"].append(scen_id_var_a)

            # get prediction of last added scenario
            new_pred = self.main_model._pred_out["obj"][-1]

            # add new constraints
            self.main_model.addConstr(self.main_model._argmax_ya >= new_pred)
            self.main_model.addConstr(self.main_model._argmax_ya <= new_pred + (obj_UB - obj_LB) * (1 - scen_id_var_a))

            self.main_model.addConstr(sum(scen_id_vars["obj"]) == 1, name="scen_sum")
            self.main_model.addConstr(gp.quicksum(i * scen_id_vars["obj"][i] for i in range(len(scen_id_vars["obj"]))) == self.main_model._argmax_wa, name='scen_idx_sum')

            # set xi_worst case as linear combination
            for j in range(self.n_items):
                lhs = gp.quicksum(xi_vals["obj"][i][j] * scen_id_vars["obj"][i] for i in range(len(scen_id_vars["obj"])))
                self.main_model.addConstr(lhs ==  self.main_model._xi_worst_case[j], name=f"xi_wc_constr_{j}")

        return scen_id_vars


    def get_x_embed(self, x):
        """ Gets scenario embedding for a particular x input. """
        x_tensor = self.x_features.clone()
        x_tensor[:,:,0] = self.to_tensor(x)

        x_embed = self.net.x_embed_net(x_tensor)
        x_embed = self.net.agg_tensor(x_embed, None)
        x_embed = self.net.x_post_agg_net(x_embed)

        x_embed = x_embed.detach().cpu().numpy()[0]

        return x_embed 


    def get_xi_embed(self, xi, scale=True):
        """ Gets scenario embedding for a particular scenario input. """
        xi_tensor = self.xi_features.clone()
        xi_tensor[:,:,0] = self.to_tensor(xi)

        xi_embed = self.net.xi_embed_net(xi_tensor)
        xi_embed = self.net.agg_tensor(xi_embed, None)
        xi_embed = self.net.xi_post_agg_net(xi_embed)

        xi_embed = xi_embed.detach().cpu().numpy()[0]

        return xi_embed 


    def get_instance(self, inst_params):
        """  Gets instances based on parameters.  """
        # directory for instances
        inst_dir = self.cfg.data_path + "kp/eval_instances/"

        # get name of instance
        inst_name = f"RKP_{inst_params['correlation']}"
        inst_name += f"_n{inst_params['n_items']}"
        inst_name += f"_R{inst_params['R']}"
        inst_name += f"_H{inst_params['H']}"
        inst_name += f"_h{inst_params['h']}"
        inst_name += f"_dev{inst_params['budget_factor']}"
        if inst_params['delta'] == 1.0:
            inst_name += f"_d{int(inst_params['delta'])}"
        else:
            inst_name += f"_d{inst_params['delta']}"

        # load and construct instance
        inst = {}
        inst['inst_name'] = inst_name

        fp_inst = inst_dir + inst_name
        with open(fp_inst, 'r') as f:
            f_lines = f.readlines()

        inst['n_items'] = inst_params['n_items']
        inst['correlation'] = inst_params['correlation']
        inst['R'] = inst_params['R']
        inst['H'] = inst_params['H']
        inst['h'] = inst_params['h']
        inst['budget_factor'] = inst_params['budget_factor']
        inst['max_budget'] = inst['n_items'] * inst['budget_factor']

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

    def do_forward_pass(x, xi, scale=True):
        """ does a forward pass for a specific x, xi pair """
        pass

    def get_inst_nn_features(self):
        """ Gets features from x (input later), xi (scenario sampling), and inst. """
        x_features = []
        xi_features = []

        # placeholders for x, xi
        x = [0] * self.n_items 
        xi = [0] * self.n_items

        # get instance features
        inst_feats = list(map(lambda i: [
                        self.inst['c'][i], 
                        self.inst['p_bar'][i], 
                        self.inst['p_hat'][i], 
                        self.inst['f'][i], 
                        self.inst['t'][i],  
                        self.inst['C']], range(self.n_items)))
            
        # create for each item in SetBased model
        x_feats = []
        xi_feats = []
        for j in range(self.n_items):
            x_feats_tmp = [x[j]] + inst_feats[j]
            xi_feats_tmp = [xi[j]] + inst_feats[j]
            x_feats.append(x_feats_tmp)
            xi_feats.append(xi_feats_tmp)

        x_features.append(x_feats)
        xi_features.append(xi_feats)
            
        x_features = np.array(x_features)  
        xi_features = np.array(xi_features)  

        # scale features
        x_features = (x_features - self.feat_scaler[0]) / (self.feat_scaler[1] - self.feat_scaler[0])
        xi_features = (xi_features - self.feat_scaler[0]) / (self.feat_scaler[1] - self.feat_scaler[0])

        # get features as tensor
        x_features = self.to_tensor(x_features)
        xi_features = self.to_tensor(xi_features)

        return x_features, xi_features

    def set_first_stage_in_adversarial_model(self, x):
        """ Sets adversarial model to x value. """
        x_embed = self.get_x_embed(x)

        # set x_embedding values
        for i, x_embed_var in enumerate(self.adv_model["obj"]._x_embed.select()):
            x_embed_var.lb = x_embed[i]
            x_embed_var.ub = x_embed[i]


    def embed_net_adversarial(self, m):
        """ Gets worst-case scenario by solving adversarial NN problem. """
        # define gurobi variables for x_embed vector
        x_embed_var = m.addVars(self.net.x_post_agg_dims[-1], name="x_embed", vtype="C", lb=-gp.GRB.INFINITY)
        m._x_embed = x_embed_var

        # define fixed variables for instance parameters
        inst_vars = self.init_grb_inst_variables(m)

        # define gurobi variables for xi values that are passed into the set network
        xi_var = m.addVars(self.n_items, name="xi", vtype="C", lb=0, ub=1)
        m._xi = xi_var

        # get gurobi input variables for xi embedding set based input
        gp_input_vars = []
        for i in range(self.n_items):
            gp_input_var = [xi_var[i]] + inst_vars[i]
            gp_input_vars.append(gp_input_var)

        # add xi embedding network to gurobi model
        xi_embed_var = self.embed_setbased_model(
            m = m,
            gp_input_vars = gp_input_vars,
            set_net = self.net.xi_embed_net,
            agg_dim = self.net.xi_embed_dims[-1],
            post_agg_net = self.net.xi_post_agg_net, 
            post_agg_dim = self.net.xi_post_agg_dims[-1],
            agg_type = self.net.agg_type,
            name = "xi_embed")

        m._xi_embed = xi_embed_var

        # add predictive constraint
        pred_in = x_embed_var.select() + xi_embed_var.select()
        pred_out =  m.addVar(name=f"pred", lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, obj=1)
        pred_constr = add_predictor_constr(m, self.net.value_net, pred_in, pred_out)

        m._pred_out_var = pred_out
        m._pred_out_constr = pred_constr

        return m


    def _get_adv_obj(self, args):
        """ Initialize objective based adversarial model. """
        # initialize gurobi model
        m = gp.Model()
        # m_adversarial.setObjective(0, sense=gp.GRB.MAXIMIZE)
        m.setParam("OutputFlag", args.verbose)
        m.setParam("MIPGap", args.adversarial_gap)
        m.setParam("TimeLimit", args.adversarial_time)
        m.setParam("MIPFocus", args.adversarial_focus)
        m._inc_time = args.adversarial_inc_time

        # embed adversarial network
        m = self.embed_net_adversarial(m)

        # constraints on xi
        lhs = 0
        for xi in m._xi.select():
            lhs += xi
        m.addConstr(lhs <= self.inst['max_budget'], name='xi_budget_constr')

        # set objective to max of NN output
        m.setObjective(m._pred_out_var, sense=gp.GRB.MAXIMIZE)

        return m

    def _sample_random_xi(self, rng, x):
        if rng is None:
            rng = np.random
        budget_max = self.n_items * self.inst['budget_factor']
        budget = np.random.uniform(1 / budget_max, budget_max)
        xi = rng.uniform(0, 1, size=self.n_items)
        xi = budget * xi / xi.sum()
        return xi

    def do_forward_pass_adv(self, x_in, xi, scale=True):
        """ does a forward pass for a specific x, xi pair """
        # create copies of tensors
        x_tensor = self.x_features.clone()
        xi_tensor = self.xi_features.clone()

        # set value of x in tensors
        x_tensor[:, :, 0] = x_in
        xi_tensor[:, :, 0] = xi

        # do a forward pass
        y_sc = self.net(x_tensor, xi_tensor, None, None)
        return y_sc


    def clamp_xi(self, xi):
        # project onto xi bounds
        xi.clamp_(0, 1)

    def check_xi(self, xi, x):
        xi_ = xi.detach().numpy()
        cons = sum(xi_[i] for i in range(self.n_items))
        if cons > self.inst["max_budget"]:
            return True
        return False
