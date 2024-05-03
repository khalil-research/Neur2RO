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
from ro.utils.cb import get_path, read_test_instance

from .approximator import Approximator


class CapitalBudgetingApproximator(Approximator):

    def __init__(self, args, cfg, net, inst_params):
        """ Constructor for Capital Budgeting Aproximator.  """
        self.cfg = cfg

        # initialize instance
        self.inst = self.get_instance(inst_params)

        # get scalers
        self.feat_scaler = net.feat_scaler
        self.label_scaler = net.label_scaler[inst_params['n_items']]

        # initialize nn
        self.net = self.initialize_nn(net)
        self.net.train()

        # cb inst specific parameters to store
        self.xi_dim = self.inst['xi_dim']
        self.loans = self.inst['loans']
        self.n_items = inst_params['n_items']
        self.inst_seed = inst_params['inst_seed']

        # cb arg specific parameters to store
        self.use_exact_cons = args.cb_use_exact_cons
        self.use_first_and_second = args.cb_use_first_and_second
        self.has_feas_adv_model = True

        # get features for forward pass
        self.x_features, self.xi_features = self.get_inst_nn_features()


    def initialize_main_model(self, args):
        """ Gets the gurobi model for the main problem. """
        m = gp.Model()
        m.setParam("OutputFlag", args.verbose)
        m.setParam("MIPGap", args.mp_gap)
        m.setParam("TimeLimit", args.mp_time)
        m.setParam("MIPFocus", args.mp_focus)
        m._inc_time = args.mp_inc_time

        # define first-stage gurobi variables
        x_var = m.addVars(self.n_items, name="x", vtype="B")
        m._x = x_var
        m._inst_vars_x = self.init_grb_inst_variables(m)

        if self.loans:
            x0_var = m.addVar(name="x0", vtype="C", lb=0, ub=self.inst['max_loan'])
            m._x0 = x0_var

        m._pred_out_desc = {"obj": [], "feas": []}
        m._pred_out = {"obj": [], "feas": []}

        # type of second-stage cost function [value_net, nn_p, lp]
        lower_bound = self.get_lower_bound()

        # variables for second-stage
        ya_var = m.addVars(self.n_items, name="ya", vtype="B")
        m._ya = ya_var

        if self.loans:
            y0a_var = m.addVar(name="y0a", vtype="C", lb=0, ub=self.inst['max_loan'])
            m._y0a = y0a_var

        # variable for objective
        z_var = m.addVar(name="z", vtype="C", lb=-lower_bound, obj=1)
        m._z = z_var

        # variables for argmax
        argmax_ya_var = m.addVar(name="argmax_ya", vtype="C", lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY)
        argmax_wa_var = m.addVar(name="argmax_wa", vtype="C", lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY)
        m._argmax_ya = argmax_ya_var
        m._argmax_wa = argmax_wa_var

        # variables for worst-case scenario
        xi_worst_case_obj = m.addVars(self.xi_dim, name="xi_worse_case_obj", vtype="C", lb=-1, ub=1)
        m._xi_worst_case_obj = xi_worst_case_obj

        # dummy constraints to be overwritten later
        m.addConstr(0 == 0, name="scen_sum_obj")
        m.addConstr(0 == 0, name="scen_idx_sum_obj")

        for j in range(self.xi_dim):
            m.addConstr(0 == 0, name=f"xi_wc_obj_constr_{j}")

        # static constraint
        for i in range(self.n_items):
            m.addConstr(ya_var[i] + x_var[i] <= 1)

        # objective: objective argmax constraint
        lhs = 0
        rev_vector = self.rev_fun(xi_worst_case_obj)
        for i in range(self.n_items):
            lhs += rev_vector[i] * (x_var[i] + self.inst['k'] * ya_var[i])
        if self.loans:
            lhs += - self.inst['l'] * (x0_var + self.inst['m'] * y0a_var)
        m.addConstr(-lhs <= z_var, name="obj_wc_obj")

        # objective: uncertain constraints
        # -- for worst case objective scenario
        cost_vector_a = self.cost_fun(xi_worst_case_obj)
        if self.loans:
            # budget constraints with loans
            lhs = 0
            for i in range(self.n_items):
                lhs += cost_vector_a[i] * x_var[i]
            m.addConstr(lhs <= self.inst['budget'] + x0_var, name="obj_wc_cons1_w_load")

            lhs = 0
            for i in range(self.n_items):
                lhs += cost_vector_a[i] * (x_var[i] + ya_var[i])
            m.addConstr(lhs <= self.inst['budget'] + x0_var + y0a_var, name="obj_wc_cons_w_loan")
        else:
            lhs = 0
            for i in range(self.n_items):
                lhs += cost_vector_a[i] * (x_var[i] + ya_var[i])
            m.addConstr(lhs <= self.inst['budget'], name="obj_wc_cons")

        # feasibility: only require if embedding NNs
        if self.use_exact_cons:
            pass 
            # todo: potentially add something here

        else:
            yf_var = m.addVars(self.n_items, name="yf", vtype="B")
            m._yf = yf_var

            if self.loans:
                y0f_var = m.addVar(name="y0f", vtype="C", lb=0, ub=self.inst['max_loan'])
                m._y0f = y0f_var

            # variables for argmax
            argmax_yf_var = m.addVar(name="argmax_yf", vtype="C", lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY)
            argmax_wf_var = m.addVar(name="argmax_wf", vtype="C", lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY)
            m._argmax_yf = argmax_yf_var
            m._argmax_wf = argmax_wf_var

            # variables for worst-case scenario
            xi_worst_case_feas = m.addVars(self.xi_dim, name="xi_worse_case_feas", vtype="C", lb=-1, ub=1)
            m._xi_worst_case_feas = xi_worst_case_feas
        
            # dummy constraints to be overwritten later
            m.addConstr(0 == 0, name="scen_sum_feas")
            m.addConstr(0 == 0, name="scen_idx_sum_feas")

            for j in range(self.xi_dim):
                m.addConstr(0 == 0, name=f"xi_wc_feas_constr_{j}")

            # static constraint
            for i in range(self.n_items):
                m.addConstr(yf_var[i] + x_var[i] <= 1)

            # -- for worst case feasibility scenario
            cost_vector_f = self.cost_fun(xi_worst_case_feas)
            if self.loans:
                # budget constraints with loans
                lhs = 0
                for i in range(self.n_items):
                    lhs += cost_vector_f[i] * x_var[i]
                m.addConstr(lhs <= self.inst['budget'] + x0_var, name="feas_wc_cons1_w_loan")

                lhs = 0
                for i in range(self.n_items):
                    lhs += cost_vector_f[i] * (x_var[i] + yf_var[i])
                m.addConstr(lhs <= self.inst['budget'] + x0_var + y0f_var, name="feas_wc_cons2_w_loan")
            else:
                lhs = 0
                for i in range(self.n_items):
                    lhs += cost_vector_f[i] * (x_var[i] + yf_var[i])
                m._feas_wc_cons = m.addConstr(lhs <= self.inst['budget'], name="feas_wc_cons")

        # add x embedding network gurobi variables
        x_gp_input_vars = []
        for i in range(self.n_items):
            input_vars = [m._x[i]] + m._inst_vars_x[i]
            x_gp_input_vars.append(input_vars)

        x_embed_var = self.embed_setbased_model(
            m=m,
            gp_input_vars=x_gp_input_vars,
            set_net=self.net.x_embed_net,
            agg_dim=self.net.x_embed_dims[-1],
            post_agg_net=self.net.x_post_agg_net,
            post_agg_dim=self.net.x_post_agg_dims[-1],
            agg_type=self.net.agg_type,
            name="x_embed")

        m._x_embed = x_embed_var

        self.main_model = m


    def initialize_adversarial_model(self, args):
        """ Gets the gurobi model(s) for the adversarial problem. """
        # objective adversarial model
        m_adv_obj = self._get_adv_obj(args)

        # feasibility adversarial model (exact)
        if self.use_exact_cons:
            if self.loans:
                raise Exception("Exact constraint optimization model not implmeneted for CB with loans")
            else:
                m_adv_cons = self._get_adv_cons_exact(args)

        # feasibility adversarial model (predicted)
        else:
            if self.loans:
                raise Exception("Predicted constraint optimization model not implmeneted for CB with loans")
            else:
                m_adv_cons = self._get_adv_cons(args)

        self.adv_model = {"obj": m_adv_obj, "feas": m_adv_cons}


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
        feat_min_x = self.feat_scaler['x'][0][1:]  # skip x
        feat_max_x = self.feat_scaler['x'][1][1:]  # skip x

        # initialize list to store variables
        inst_vars_x = []
        for idx in range(self.n_items):

            # get features and normalize
            inst_vals_x = [self.inst['c_bar'][idx], self.inst['r_bar'][idx]]
            inst_vals_x = (np.array(inst_vals_x) - feat_min_x) / (feat_max_x - feat_min_x)

            # add features as gurobi variables
            c_bar_var = m.addVar(name=f"c_bar_{idx}", vtype="C")
            c_bar_var.lb = inst_vals_x[0]
            c_bar_var.ub = inst_vals_x[0]

            r_bar_var = m.addVar(name=f"r_bar_{idx}", vtype="C")
            r_bar_var.lb = inst_vals_x[1]
            r_bar_var.ub = inst_vals_x[1]

            inst_vars_x.append([c_bar_var, r_bar_var])

        return inst_vars_x


    def embed_value_network(self, xi_embed, n_iterations, scen_type):
        """ Embeds value network in gurobi model. """
        xi_embed_var = self.main_model.addVars(self.net.xi_post_agg_dims[-1], name=f"xi_embed_{n_iterations}_{scen_type}", lb=-gp.GRB.INFINITY)
        for i in range(self.net.xi_post_agg_dims[-1]):
            xi_embed_var[i].lb = xi_embed[i]
            xi_embed_var[i].ub = xi_embed[i]

        pred_in = self.main_model._x_embed.select() + xi_embed_var.select()
        pred_out = self.main_model.addVars(2, name=f"pred_{n_iterations}_{scen_type}", lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY)
        pred_constr = add_predictor_constr(self.main_model, self.net.value_net, pred_in, pred_out, name=f"pred_constr_{n_iterations}_{scen_type}")
        self.main_model._pred_out[scen_type].append(pred_out)
        return pred_out


    def add_worst_case_scenario_to_main(self, xi, n_iterations, scen_type):
        """ Adds worst-case scenario(s) to main problem. """
        # get scenario embedding
        xi_embed = self.get_xi_embed(xi)

        # add predictive contraints for worst-case scenario to the main problem
        z_scen = self.embed_value_network(xi_embed, n_iterations, scen_type=scen_type)

        # get scaled objective
        z_scen_desc = self.main_model.addVars(2, name=f"z_sc_{n_iterations}_{scen_type}", lb=-gp.GRB.INFINITY)
        self.main_model._pred_out_desc[scen_type].append(z_scen_desc)
        self.main_model.addConstr((z_scen_desc[0] - self.label_scaler['obj'][0]) / (self.label_scaler['obj'][1] - self.label_scaler['obj'][0]) == z_scen[0])
        self.main_model.addConstr((z_scen_desc[1] - self.label_scaler['feas'][0]) / (self.label_scaler['feas'][1] - self.label_scaler['feas'][0]) == z_scen[1])


    def change_worst_case_scen(self, xi_to_add, scen_id_vars, xi_vals, n_iterations):
        """ """
        obj_LB, obj_UB = -1000, 1000
        feas_LB, feas_UB = -1000, 1000

        # add scenario for objective
        if "obj" in xi_to_add:  
            # remove constraints from previous argmax
            self.main_model.remove(self.main_model.getConstrByName("scen_sum_obj"))
            self.main_model.remove(self.main_model.getConstrByName("scen_idx_sum_obj"))
            for j in range(self.xi_dim):
                self.main_model.remove(self.main_model.getConstrByName(f"xi_wc_obj_constr_{j}"))

            # add argmax variables
            scen_id_vara = self.main_model.addVar(name=f"scen_id_a_{n_iterations}", vtype="B", obj=0)
            scen_id_vars["obj"].append(scen_id_vara)
            
            # get updated value to argmax over that includes first-stage objective
            rev_vector = self.rev_fun(xi_to_add["obj"])
            lhs = sum([rev_vector[i] * self.main_model._x[i] for i in range(self.n_items)])
            if self.loans:
                lhs += - inst['l'] * self.main_model._x0

            if self.use_first_and_second:
                new_pred = self.main_model._pred_out_desc["obj"][-1][0]
            else:
                new_pred = (- lhs + self.main_model._pred_out_desc["obj"][-1][0])

            # add new constraints
            self.main_model.addConstr(self.main_model._argmax_ya >= new_pred)
            self.main_model.addConstr(self.main_model._argmax_ya <= new_pred + (obj_UB - obj_LB) * (1 - scen_id_vara))

            self.main_model.addConstr(sum(scen_id_vars["obj"]) == 1, name="scen_sum_obj")
            self.main_model.addConstr(gp.quicksum(i * scen_id_vars["obj"][i] for i in range(len(scen_id_vars["obj"]))) == self.main_model._argmax_wa, name='scen_idx_sum_obj')

            # set xi_worst case as linear combination
            for j in range(self.xi_dim):
                lhs = gp.quicksum(xi_vals["obj"][i][j] * scen_id_vars["obj"][i] for i in range(len(scen_id_vars["obj"])))
                self.main_model.addConstr(lhs == self.main_model._xi_worst_case_obj[j], name=f"xi_wc_obj_constr_{j}")

        # add scenario for constraint
        if "feas" in xi_to_add:

            # constraints for not violating c(xi)^T x for current worst-case xi xi
            if self.use_exact_cons:
                cost_vector = self.cost_fun(xi_to_add["feas"])
                lhs = sum([cost_vector[i] * self.main_model._x[i] for i in range(self.n_items)])
                self.main_model.addConstr(lhs <= self.inst['budget'], name=f"wc_feas_{n_iterations}")

            # constraints for argmax of NN outputs
            else:
                # remove constraints from previous argmax
                self.main_model.remove(self.main_model.getConstrByName("scen_sum_feas"))
                self.main_model.remove(self.main_model.getConstrByName("scen_idx_sum_feas"))
                for j in range(self.xi_dim):
                    self.main_model.remove(self.main_model.getConstrByName(f"xi_wc_feas_constr_{j}"))

                # add argmax variables
                scen_id_varf = self.main_model.addVar(name=f"scen_id_f_{n_iterations}", vtype="B", obj=0)
                scen_id_vars["feas"].append(scen_id_varf)

                # get updated value to argmax over that includes lhs constraint of first-stage variable
                cost_vector = self.cost_fun(xi_to_add["feas"])
                lhs = sum([cost_vector[i] * self.main_model._x[i] for i in range(self.n_items)])
                # new_pred = (lhs + m._pred_out_desc["feas"][-1][1]) ## VERSION FOR ONLY PREDICTING Y
                new_pred = self.main_model._pred_out_desc["feas"][-1][1]  ## VERSION FOR PREDICTING Y + X

                # add new constraint
                self.main_model.addConstr(self.main_model._argmax_yf >= new_pred)
                self.main_model.addConstr(self.main_model._argmax_yf <= new_pred + (feas_UB - feas_LB) * (1 - scen_id_varf))

                self.main_model.addConstr(sum(scen_id_vars["feas"]) == 1, name="scen_sum_feas")
                self.main_model.addConstr(gp.quicksum(i * scen_id_vars["feas"][i] for i in range(len(scen_id_vars["feas"]))) == self.main_model._argmax_wf, name='scen_idx_sum_feas')

                # set xi_worst case as linear combination
                for j in range(self.xi_dim):
                    lhs = gp.quicksum(xi_vals["feas"][i][j] * scen_id_vars["feas"][i] for i in range(len(scen_id_vars["feas"])))
                    self.main_model.addConstr(lhs == self.main_model._xi_worst_case_feas[j], name=f"xi_wc_feas_constr_{j}")

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
        # compute phi/psi products
        phi = np.array(list(self.inst['phi'].values()))
        psi = np.array(list(self.inst['psi'].values()))
        phi_prod = 1 + phi @ xi / 2
        psi_prod = 1 + psi @ xi / 2

        # scale phi/psi products
        if scale:
            min_phi = self.feat_scaler['xi'][0][0] 
            max_phi = self.feat_scaler['xi'][1][0]
            phi_prod = (phi_prod - min_phi) / (max_phi - min_phi)

            min_psi = self.feat_scaler['xi'][0][1] 
            max_psi = self.feat_scaler['xi'][1][1] 
            psi_prod = (psi_prod - min_psi) / (max_psi - min_psi)

        # set values in tensor
        xi_tensor = self.xi_features.clone()
        xi_tensor[:,:,0] = self.to_tensor(phi_prod)
        xi_tensor[:,:,1] = self.to_tensor(psi_prod)

        xi_embed = self.net.xi_embed_net(xi_tensor)
        xi_embed = self.net.agg_tensor(xi_embed, None)
        xi_embed = self.net.xi_post_agg_net(xi_embed)

        xi_embed = xi_embed.detach().cpu().numpy()[0]

        return xi_embed


    def get_instance(self, inst_params):
        """  Gets instances based on parameters.  """
        inst_dir = self.cfg.data_path + "cb/eval_instances/"
        inst = read_test_instance(inst_dir, self.cfg, inst_params['inst_seed'], inst_params['n_items'])
        return inst


    def do_forward_pass(self, x, xi, scale=True):
        """ does a forward pass for a specific x, xi pair """
        # create copies of tensors
        x_tensor = self.x_features.clone()
        xi_tensor = self.xi_features.clone()

        # set value of x in tensors
        x_tensor[:,:,0] = self.to_tensor(x)

        # compute phi/psi products
        xi = np.array(xi)
        phi = np.array(list(inst['phi'].values()))
        psi = np.array(list(inst['psi'].values()))
        phi_prod = 1 + phi @ xi / 2
        psi_prod = 1 + psi @ xi / 2

        # scale phi/psi products
        if scale:
            min_phi = self.feat_scaler['xi'][0][0] 
            max_phi = self.feat_scaler['xi'][1][0]
            phi_prod = (phi_prod - min_phi) / (max_phi - min_phi)

            min_psi = self.feat_scaler['xi'][0][1] 
            max_psi = self.feat_scaler['xi'][1][1] 
            psi_prod = (psi_prod - min_psi) / (max_psi - min_psi)

        xi_tensor[:, :, 0] = self.to_tensor(phi_prod)
        xi_tensor[:, :, 1] = self.to_tensor(psi_prod)

        # do a forward pass
        y_sc = self.net(x_tensor, xi_tensor, None, None).detach().cpu().numpy()

        if scale:
            y_obj_min = self.label_scaler['obj'][0]
            y_obj_max = self.label_scaler['obj'][1]
            y_sc[0][0] = y_sc[0][0] * (y_obj_max - y_obj_min) + y_obj_min

        return y_sc

    def do_forward_pass_adv(self, x_in, xi, scale=True):
        """ does a forward pass for a specific x, xi pair """
        # create copies of tensors
        x_tensor = self.x_features.clone()
        xi_tensor = self.xi_features.clone()

        # set value of x in tensors
        x_tensor[:,:,0] = x_in
        # compute phi/psi products
        phi_np = np.array(list(self.inst['phi'].values()))
        psi_np = np.array(list(self.inst['psi'].values()))
        phi = self.to_tensor(phi_np)
        psi = self.to_tensor(psi_np)
        phi_prod = 1 + phi @ xi / 2
        psi_prod = 1 + psi @ xi / 2

        # scale phi/psi products
        if scale:
            min_phi = self.feat_scaler['xi'][0][0]
            max_phi = self.feat_scaler['xi'][1][0]
            phi_prod = (phi_prod - min_phi) / (max_phi - min_phi)

            min_psi = self.feat_scaler['xi'][0][1]
            max_psi = self.feat_scaler['xi'][1][1]
            psi_prod = (psi_prod - min_psi) / (max_psi - min_psi)

        xi_tensor[:,:,0] = self.to_tensor(phi_prod)
        xi_tensor[:,:,1] = self.to_tensor(psi_prod)

        # do a forward pass
        y_sc = self.net(x_tensor, xi_tensor, None, None)

        if scale:
            y_obj_min = self.label_scaler['obj'][0]
            y_obj_max = self.label_scaler['obj'][1]
            y_sc[0][0] = y_sc[0][0] * (y_obj_max - y_obj_min) + y_obj_min

        return y_sc

    def do_forward_pass_adv_batch(self, x, xi, scale=True):
        """ does a forward pass for a specific x, xi pair """
        # create copies of tensors
        xi = xi[0]
        x_tensor = self.x_features.clone()
        xi_tensor = self.xi_features.clone()
        # set value of x in tensors
        x_tensor = self.to_tensor(np.array([x_tensor[0, :, :]]*xi.shape[0]))
        xi_tensor = self.to_tensor(np.array([xi_tensor[0, :, :]]*xi.shape[0]))

        x_tensor[:,:,0] = self.to_tensor(np.array([x]*xi.shape[0]))
        # compute phi/psi products
        phi_np = np.array(list(self.inst['phi'].values()))
        psi_np = np.array(list(self.inst['psi'].values()))
        phi = self.to_tensor(phi_np)
        psi = self.to_tensor(psi_np)
        phi_prod = 1 + phi @ xi.T / 2
        psi_prod = 1 + psi @ xi.T / 2

        # scale phi/psi products
        if scale:
            min_phi = self.feat_scaler['xi'][0][0]
            max_phi = self.feat_scaler['xi'][1][0]
            phi_prod = (phi_prod - min_phi) / (max_phi - min_phi)

            min_psi = self.feat_scaler['xi'][0][1]
            max_psi = self.feat_scaler['xi'][1][1]
            psi_prod = (psi_prod - min_psi) / (max_psi - min_psi)

        xi_tensor[:,:,0] = phi_prod.T
        xi_tensor[:,:,1] = psi_prod.T

        # do a forward pass
        y_sc = self.net(x_tensor, xi_tensor, None, None)

        if scale:
            y_obj_min = self.label_scaler['obj'][0]
            y_obj_max = self.label_scaler['obj'][1]
            y_sc[:, 0] = y_sc[:, 0] * (y_obj_max - y_obj_min) + y_obj_min

        return y_sc[:, 0]

    def _sample_random_xi(self, rng=None, x=None):
        if rng is None:
            rng = np.random
        while True:
            xi = rng.uniform(-1, 1, size=self.inst['xi_dim'])
            if x is None:
                break
            cost_vector = self.cost_fun(xi)
            cons = sum(cost_vector[i] * x[i] for i in range(self.n_items))
            if cons <= self.inst["budget"]:
                break
        return xi

    def clamp_xi(self, xi):
        # project onto xi bounds
        xi.clamp_(-1, 1)

    def check_xi(self, xi, x):
        xi_ = xi.detach().numpy()
        cost_vector = self.cost_fun(xi_)
        cons = sum(cost_vector[i] * x[i] for i in range(self.n_items))
        if cons > self.inst["budget"]:
            return True
        return False

    def get_inst_nn_features(self):
        """ Gets features from x (input later), xi (scenario sampling), and inst. """
        x_features = []
        xi_features = []

        # zero placeholders for x, xi
        x_placeholder = [0] * self.n_items 
        phi_psi_placeholder = [0] * self.n_items

        # loop over each xi in xi_vals
        prob_feats_x = list(map(lambda i: [self.inst['c_bar'][i], self.inst['r_bar'][i]], range(self.n_items)))
        prob_feats_xi = list(map(lambda i: [self.inst['c_bar'][i], self.inst['r_bar'][i]], range(self.n_items)))

        # prob_feats = [inst['k'], inst['l'], inst['m']]        # unused problem features (fixed across instances)
        x_feats = []
        xi_feats = []
        for i in range(self.n_items):
            x_ft_i = [x_placeholder[i]] + prob_feats_x[i]
            x_feats.append(x_ft_i)

        for i in range(self.n_items):
            xi_ft_i = [phi_psi_placeholder[i], phi_psi_placeholder[i]] + prob_feats_xi[i]
            xi_feats.append(xi_ft_i)

        x_features.append(x_feats)
        xi_features.append(xi_feats)

        x_features = np.array(x_features)
        xi_features = np.array(xi_features)

        # scale features
        x_features = (x_features - self.feat_scaler['x'][0]) / (self.feat_scaler['x'][1] - self.feat_scaler['x'][0])
        xi_features = (xi_features - self.feat_scaler['xi'][0]) / (self.feat_scaler['xi'][1] - self.feat_scaler['xi'][0])

        # get features as tensor
        x_features = self.to_tensor(x_features)
        xi_features = self.to_tensor(xi_features)

        # xi_features = xi_features.reshape(1, xi_features.shape[0], xi_features.shape[1]) # reshape for forward pass

        return x_features, xi_features


    def set_first_stage_in_adversarial_model(self, x):
        """ Sets adversarial model to x value. """
        if self.loans:
            raise Exception("Not implemented with loans")

        for i in range(self.n_items):
            self.adv_model["obj"]._x[i].lb = x[i]
            self.adv_model["obj"]._x[i].ub = x[i]
            self.adv_model["feas"]._x[i].lb = x[i]
            self.adv_model["feas"]._x[i].ub = x[i]

        # embed x
        x_embed = self.get_x_embed(x)

        # set x embedd values
        for i, x_embed_var in enumerate(self.adv_model["obj"]._x_embed.select()):
            x_embed_var.lb = x_embed[i]
            x_embed_var.ub = x_embed[i]

        if not self.use_exact_cons:
            for i, x_embed_var in enumerate(self.adv_model["feas"]._x_embed.select()):
                x_embed_var.lb = x_embed[i]
                x_embed_var.ub = x_embed[i]


    def embed_net_adversarial(self, m):
        """ Gets worst-case scenario by solving adversarial NN problem. """
        # define gurobi variables for x_embed vector
        x_embed_var = m.addVars(self.net.x_post_agg_dims[-1], name="x_embed", vtype="C", lb=-gp.GRB.INFINITY)
        m._x_embed = x_embed_var

        # define fixed variables for instance parameters
        inst_vars_xi = self.init_grb_inst_variables(m)

        # define gurobi variables for xi values that are passed into the set network
        xi_var = m.addVars(self.xi_dim, name="xi", vtype="C", lb=-1, ub=1)
        m._xi = xi_var

        # compute product with phi, psi
        phi_prod_var = m.addVars(self.n_items, name="phi_prod", vtype="C", lb=-gp.GRB.INFINITY)   
        psi_prod_var = m.addVars(self.n_items, name="psi_prod", vtype="C", lb=-gp.GRB.INFINITY)  
        m._phi_prod_var = phi_prod_var
        m._psi_prod_var = psi_prod_var

        # add equality constriants to enfore values of phi/phi prod variables
        for i in range(self.n_items):
            phi_lhs, psi_lhs = 0, 0
            for j in range(self.xi_dim):
                phi_lhs += (self.inst['phi'][i][j] * xi_var[j]) / 2
                psi_lhs += (self.inst['psi'][i][j] * xi_var[j]) / 2
            phi_lhs += 1
            psi_lhs += 1
            m.addConstr(phi_prod_var[i] == phi_lhs, name=f'phi_{i}')
            m.addConstr(psi_prod_var[i] == psi_lhs, name=f'psi_{i}')

        # variables for scaled inputs for psi/phi products
        phi_prod_var_sc =  m.addVars(self.n_items, name="phi_prod_sc", vtype="C", lb=-gp.GRB.INFINITY)
        psi_prod_var_sc =  m.addVars(self.n_items, name="psi_prod_sc", vtype="C", lb=-gp.GRB.INFINITY)
        m._phi_prod_var_sc = phi_prod_var_sc
        m._psi_prod_var_sc = psi_prod_var_sc

        # constraints to ensure scaled inputs for psi/phi products
        for i in range(self.n_items):
            m.addConstr(phi_prod_var_sc[i] == (phi_prod_var[i] - self.feat_scaler['xi'][0][0])/(self.feat_scaler['xi'][1][0] - self.feat_scaler['xi'][0][0]), name=f'phi_prod_sc_{i}')
            m.addConstr(psi_prod_var_sc[i] == (psi_prod_var[i] - self.feat_scaler['xi'][0][0])/(self.feat_scaler['xi'][1][0] - self.feat_scaler['xi'][0][0]), name=f'psi_prod_sc_{i}')

        # get gurobi input variables for xi embedding set based input
        gp_input_vars = []
        for i in range(self.n_items):
            gp_input_var = [phi_prod_var_sc[i], psi_prod_var_sc[i]] + inst_vars_xi[i]
            gp_input_vars.append(gp_input_var)

        # add xi embedding network to gurobi model
        xi_embed_var = self.embed_setbased_model(
                                            m=m,
                                            gp_input_vars=gp_input_vars,
                                            set_net=self.net.xi_embed_net,
                                            agg_dim=self.net.xi_embed_dims[-1],
                                            post_agg_net=self.net.xi_post_agg_net,
                                            post_agg_dim=self.net.xi_post_agg_dims[-1],
                                            agg_type=self.net.agg_type,
                                            name='xi_embed')

        m._xi_embed = xi_embed_var

        # add predictive constraint
        pred_in = x_embed_var.select() + xi_embed_var.select()

        pred_out = m.addVars(2, name=f"pred", lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY)
        pred_constr = add_predictor_constr(m, self.net.value_net, pred_in, pred_out)
        m._pred_out_var = pred_out
        m._pred_out_constr = pred_constr

        pred_out = np.array([var for var in pred_out.select()])

        # get scaled objective
        pred_out_desc = m.addVars(2, name=f"pr_out_scaled", lb=-gp.GRB.INFINITY)
        m._pred_out_desc = pred_out_desc
        m.addConstr((pred_out_desc[0] - self.label_scaler['obj'][0]) / (self.label_scaler['obj'][1] - self.label_scaler['obj'][0]) == pred_out[0])
        m.addConstr((pred_out_desc[1] - self.label_scaler['feas'][0]) / (self.label_scaler['feas'][1] - self.label_scaler['feas'][0]) == pred_out[1])
        
        return m


    def _get_adv_obj(self, args):
        """ Initialize objective based adversarial model. """
        # initialize gurobi model
        m = gp.Model()
        m.setParam("OutputFlag", args.verbose)
        m.setParam("MIPGap", args.adversarial_gap)
        m.setParam("TimeLimit", args.adversarial_time)
        m.setParam("MIPFocus", args.adversarial_focus)
        m._inc_time = args.adversarial_inc_time

        m = self.embed_net_adversarial(m)

        # define variables (xi already defined in embed_net_adversarial)
        x_var = m.addVars(self.n_items, name="x", vtype="B")
        m._x = x_var
        if self.inst['loans']:
            x0_var = m.addVar(name="x0", vtype="C", lb=0, ub=self.inst['max_loan'])
            m._x0 = x0_var

        # objective
        rev_vector = self.rev_fun(m._xi)
        obj_x = 0
        for i in range(self.n_items):
            obj_x += rev_vector[i] * x_var[i]
        if self.loans:
            obj_x -= self.inst['l'] * x0_var       # prediction of y0 is already included in pred_out

        # use pred first + second-stage
        if self.use_first_and_second:
            total_obj = m._pred_out_desc[0]

        # use exact first and pred for second-stage
        else: 
            total_obj = (- obj_x + m._pred_out_desc[0])

        m.setObjective(total_obj, sense=gp.GRB.MAXIMIZE)

        # add constraint on budget for worst-case scenario
        cost_vector = self.cost_fun(m._xi)
        cons = sum(cost_vector[i] * x_var[i] for i in range(self.n_items))
        m.addConstr(cons <= self.inst['budget'])

        return m

    def _get_adv_cons(self, args):
        """ Initialize feasibility prediction-based adversarial model. """
        m = gp.Model()
        m.setParam("OutputFlag", args.verbose)
        m.setParam("TimeLimit", args.adversarial_time)
        m.setParam("MipGap", args.adversarial_gap)
        m.setParam("MIPFocus", args.adversarial_focus)
        m._inc_time = args.adversarial_inc_time

        m = self.embed_net_adversarial(m)

        # define variables (xi already defined in embed_net_adversarial)
        x_var = m.addVars(self.n_items, name="x", vtype="B")
        m._x = x_var

        # CONSTRAINT OBJECTIVE
        cost_vector = self.cost_fun(m._xi)
        #cons_x = sum(cost_vector[i] * x_var[i] for i in range(inst['n_items'])) # only needed if predicting y values
        #m.setObjective(cons_x + m._pred_out_desc[1], sense=gp.GRB.MAXIMIZE)
        m.setObjective(m._pred_out_desc[1], sense=gp.GRB.MAXIMIZE)

        return m


    def _get_adv_cons_exact(self, args):
        """ Initialize feasibility exact adversarial model. """
        m = gp.Model()
        #m.setParam("OutputFlag", args.verbose)
        m.setParam("OutputFlag", 0)
        m.setParam("MipGap", args.adversarial_gap)
        m.setParam("TimeLimit", args.adversarial_time)
        #m.setParam("MIPFocus", args.adversarial_focus)
        m._inc_time = args.adversarial_inc_time

        # define variables (xi already defined in embed_net_adversarial)
        x_var = m.addVars(self.n_items, name="x", vtype="B")
        m._x = x_var

        # define xi variables
        xi_var = m.addVars(self.xi_dim, name="xi", vtype="C", lb=-1, ub=1)
        m._xi = xi_var 

        # objective for exact constraint adv MILP
        cost_vector = self.cost_fun(m._xi)
        cons_obj = sum(cost_vector[i] * x_var[i] for i in range(self.n_items))
        m.setObjective(cons_obj, sense=gp.GRB.MAXIMIZE)

        return m


    # def _get_adv_cons_loans(self, args):
    #     """ Initialize feasiblity prediction-based adv model for loans. """
    #     m = gp.Model()
    #     m.setParam("OutputFlag", args.verbose)
    #     m.setParam("MipGap", args.adversarial_gap)
    #     m.setParam("TimeLimit", args.adversarial_time)
    #     m.setParam("MIPFocus", args.adversarial_focus)
    #     m._inc_time = args.adversarial_inc_time

    #     m = embed_net_adversarial(m, net, inst)

    #     # define variables (xi already defined in embed_net_adversarial)
    #     x_var = m.addVars(inst['n_items'], name="x", vtype="B")
    #     m._x = x_var
    #     x0_var = m.addVar(name="x0", vtype="C", lb=0, ub=inst['max_loan'])
    #     m._x0 = x0_var

    #     # ADVERSARIAL PROBLEM
    #     bigM = get_bigM(inst)
    #     # define variables
    #     num_cons = 2
    #     v_var = m.addVars(num_cons, name="v", vtype="B")
    #     m._v = v_var
    #     zeta_var = m.addVar(lb=-bigM, name="zeta", obj=-1)
    #     m._zeta = zeta_var
    #     # violation constraint
    #     m.addConstr(sum(v_var[l] for l in range(num_cons)) == 1)

    #     # budget constraints
    #     cost_vector = cost_fun(inst, m._xi)
    #     rhs = sum(cost_vector[i] * x_var[i] for i in range(inst['n_items'])) + m._pred_out_desc[1] - inst['budget']
    #     if inst['loans']:
    #         rhs += - x0_var
    #     m.addConstr(zeta_var + bigM * (v_var[0] - 1) <= rhs)

    #     m.addConstr(zeta_var + bigM * (v_var[1] - 1) <= sum(cost_vector[i] * x_var[i] for i in range(inst['n_items']))
    #         - inst['budget'] - x0_var)

    #     return m


    # def _get_adv_cons_loans_exact(self, args):
    #     """ Initialize feasiblity exact adv model for loans. """
    #     # todo: Implement this with capital budgeting + loans
    #     raise Exception("Adversarial feasibility model not implemented for CB with loans.  ") 


    def cost_fun(self, xi):
        return np.array([(1 + sum(self.inst['phi'][i][j] * xi[j] for j in range(self.xi_dim)) / 2) *
                         self.inst['c_bar'][i]
                         for i in range(self.n_items)])

    def rev_fun(self, xi):
        return np.array([(1 + sum(self.inst['psi'][i][j] * xi[j] for j in range(self.xi_dim)) / 2) *
                         self.inst['r_bar'][i]
                         for i in range(self.n_items)])

    def get_bigM(self):
        return sum([(1 + 1 / 2) * self.inst['c_bar'][i] for i in range(self.n_items)])

    def get_lower_bound(self):
        return sum([(1 + 1 / 2) * self.inst['r_bar'][i] for i in range(self.n_items)])

    def get_init_scen(self):
        return np.zeros(self.xi_dim)



# -------------------------------------------------------#
#                       CB Functions                     #
# -------------------------------------------------------#

