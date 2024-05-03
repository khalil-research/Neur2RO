from abc import ABC, abstractmethod

import torch

import gurobipy as gp
from gurobi_ml import add_predictor_constr

class Approximator(ABC):

    @abstractmethod
    def initialize_main_model(self):
        """ Gets the gurobi model for the main problem. """
        pass

    @abstractmethod
    def initialize_adversarial_model(self):
        """ Gets the gurobi model(s) for the adversarial problem. """
        pass

    @abstractmethod
    def initialize_nn(self, net_):
        pass

    @abstractmethod
    def init_grb_inst_variables(self, m):
        """ Initialize gurobi variables for problem features (i.e., input to NN).  """
        pass

    @abstractmethod
    def add_worst_case_scenario_to_main(self, xi, n_iterations, scen_type):
        """ Adds worst-case scenario(s) to main problem. """
        pass

    @abstractmethod
    def change_worst_case_scen(self, xi_to_add, scen_id_vars, xi_vals, n_iterations):
        """ Changes worst-case scenario constraints in main problem. """
        pass

    @abstractmethod
    def get_x_embed(self, x):
        """ Gets scenario embedding for a particular x input. """
        pass 

    @abstractmethod
    def get_xi_embed(self, xi):
        """ Gets scenario embedding for a particular scenario input. """
        pass

    @abstractmethod
    def get_instance(self, cfg, inst_params):
        """  Gets instances based on parameters.  """
        pass

    @abstractmethod
    def do_forward_pass(x, xi, scale=True):
        pass

    @abstractmethod
    def get_inst_nn_features(self):
        """ Gets features from x (input later), xi (scenario sampling), and inst. """
        pass

    @abstractmethod
    def embed_net_adversarial(self, m):
        """ Embeds adversarial network.  """
        pass

    @abstractmethod
    def embed_value_network(self, xi_embed, n_iterations, scen_type):
        """ Embeds value network in main problem.  """
        pass

    def to_tensor(self, x):
        """ Converts numpy/list to a tensor. """
        return torch.Tensor(x).float()

    def embed_setbased_model(self, m, gp_input_vars, set_net, agg_dim, post_agg_net, post_agg_dim, agg_type, name):
        """ Embeds set-based predictive models. """
        # add predictive constraints for each element of the set
        n_items_in_set = len(gp_input_vars)
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
            for j in range(n_items_in_set):
                # print(set_outputs[j])
                # agg_sum += set_outputs[j].values()[i]
                agg_sum += set_outputs[j][i]
            if agg_type == "sum":
                k = m.addConstr(agg_sum == post_agg_vars[i])
            elif agg_type == "mean":
                k = m.addConstr(agg_sum / n_items_in_set == post_agg_vars[i])

        # post-agg net variable
        gp_embed_vars = m.addVars(post_agg_dim, vtype="C", lb=-gp.GRB.INFINITY, name=name)
        pred_constr = add_predictor_constr(m, post_agg_net, post_agg_vars, gp_embed_vars)

        return gp_embed_vars
