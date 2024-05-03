from gurobipy import GRB
import gurobipy as gp
import pickle as pkl
import numpy as np
import os

from ro.utils.cb import get_path
from .ccg import CCGAlgorithm


class CapitalBudgetingCCG(CCGAlgorithm):
    def __init__(self, problem_config):
        self.cfg = problem_config
        self.rng = np.random.RandomState()
        super().__init__()

    def set_inst(self, inst):
        self.inst = inst
        self.lower_bound = self.get_lower_bound()
        self.upper_bound = self.get_upper_bound()
        self.init_scen = self.get_init_scen()
        self.bigM = self.get_bigM()
        self.inst_seed = self.inst['inst_seed']

    def run_algorithm(self, time_limit=30*60, x_init=None):
        if x_init is not None:
            p = "ccg_eval_x_res"
            ml = True
            self.results_path = get_path(self.cfg.data_path, self.cfg, p, ml=ml)
        else:
            save_path = self.cfg.data_path + "cb/ccg_res/"
            os.makedirs(save_path, exist_ok=True)
            self.results_path = save_path + f"res_{self.inst['inst_name']}.pkl"

        obj = self.algorithm(time_limit, x_init=x_init)
        return obj

    def master_problem_update(self, xi, m):
        # load variables
        theta = m._theta

        x_var = m._x
        y_var = m.addVars(self.inst['n_items'], vtype="C", lb=0, ub=1)
        m._y.append(y_var)
        if self.inst['loans']:
            x0_var = m._x0
            y0_var = m.addVar(vtype="C", lb=0, ub=self.inst['max_loan'])
            m._y0.append(y0_var)

        # add new constraints
        # objective constraint
        lhs = 0
        rev_vector = self.rev_fun(xi)
        for i in range(self.inst['n_items']):
            lhs += rev_vector[i] * (x_var[i] + self.inst['k'] * y_var[i])
        if self.inst['loans']:
            lhs += - self.inst['l'] * (x0_var + self.inst['m'] * y0_var)
        m.addConstr(-lhs <= theta)

        # static constraint
        for i in range(self.inst['n_items']):
            m.addConstr(y_var[i] + x_var[i] <= 1)

        # uncertain constraints
        cost_vector = self.cost_fun(xi)
        if self.inst['loans']:
            # budget constraints with loans
            lhs = 0
            for i in range(self.inst['n_items']):
                lhs += cost_vector[i] * x_var[i]
            m.addConstr(lhs <= self.inst['budget'] + x0_var)

            lhs = 0
            for i in range(self.inst['n_items']):
                lhs += cost_vector[i] * (x_var[i] + y_var[i])
            m.addConstr(lhs <= self.inst['budget'] + x0_var + y0_var)
        else:
            lhs = 0
            for i in range(self.inst['n_items']):
                lhs += cost_vector[i] * (x_var[i] + y_var[i])
            m.addConstr(lhs <= self.inst['budget'])

        # solve
        m.update()
        m.optimize()
        # get results
        theta_sol = theta.x
        x_sol = np.array([var.X for i, var in x_var.items()])
        if self.inst['loans']:
            x_sol = [x_sol, x0_var.X]
        y_sol = []
        for k in range(len(m._y)):
            y_new = np.array([var.X for i, var in m._y[k].items()])
            y_sol.append(y_new)
        if self.inst['loans']:
            y_sol = [y_sol, np.array([var.X for var in m._y0])]

        return theta_sol, x_sol, y_sol, m

    def master_problem_build(self, x_init=None):
        # first-stage objective
        m = gp.Model()

        # objective
        theta_var = m.addVar(lb=-self.lower_bound, ub=0, name="theta", obj=1)
        m._theta = theta_var
        # first-stage variables
        x_var = m.addVars(self.inst['n_items'], name="x", vtype="B")
        if x_init is not None:
            if self.inst['loans']:
                x_init, x0_init = x_init
            for i, x in enumerate(x_init):
                x_var[i].lb = x
                x_var[i].ub = x
        m._x = x_var

        if self.inst['loans']:
            x0_var = m.addVar(name="x0", vtype="C", lb=0, ub=self.inst['max_loan'])
            if x_init is not None:
                x0_var.lb = x0_init
                x0_var.ub = x0_init
            m._x0 = x0_var
        # second-stage constraints
        m._y = list()
        y_var = m.addVars(self.inst['n_items'], name=f"y_0", vtype="C", lb=0, ub=1)
        m._y.append(y_var)
        if self.inst['loans']:
            m._y0 = list()
            y0_var = m.addVar(name="y0", vtype="C", lb=0, ub=self.inst['max_loan'])
            m._y0.append(y0_var)

        # constraints
        # objective constraint
        lhs = 0
        rev_vector = self.rev_fun(self.init_scen)
        for i in range(self.inst['n_items']):
            lhs += rev_vector[i]*(x_var[i] + self.inst['k']*y_var[i])
        if self.inst['loans']:
            lhs += - self.inst['l'] * (x0_var + self.inst['m'] * y0_var)
        m.addConstr(-lhs <= theta_var)

        # static constraint
        for i in range(self.inst['n_items']):
            m.addConstr(y_var[i] + x_var[i] <= 1)

        # uncertain constraints
        cost_vector = self.cost_fun(self.init_scen)
        if self.inst['loans']:
            # budget constraints with loans
            lhs = 0
            for i in range(self.inst['n_items']):
                lhs += cost_vector[i]*x_var[i]
            m.addConstr(lhs <= self.inst['budget'] + x0_var)

            lhs = 0
            for i in range(self.inst['n_items']):
                lhs += cost_vector[i]*(x_var[i] + y_var[i])
            m.addConstr(lhs <= self.inst['budget'] + x0_var + y0_var)
        else:
            lhs = 0
            for i in range(self.inst['n_items']):
                lhs += cost_vector[i]*(x_var[i] + y_var[i])
            m.addConstr(lhs <= self.inst['budget'])

        m.setParam("OutputFlag", 0)
        m.optimize()

        # get results
        theta_sol = theta_var.x
        x_sol = np.array([var.X for i, var in x_var.items()])
        y_sol = [np.array([var.X for i, var in y_var.items()])]
        if self.inst['loans']:
            x_sol = [x_sol, x0_var.X]
            y_sol = [y_sol, [y0_var.X]]

        return theta_sol, x_sol, y_sol, m

    # SUBPROBLEM
    def sub_problem_build(self, x):        # TODO
        # model
        m = gp.Model("Adversarial Problem")
        m.Params.OutputFlag = 0
        m.setObjective(0, sense=gp.GRB.MAXIMIZE)
        # variables
        d_obj_var = m.addVar(lb=-gp.GRB.INFINITY, ub=0, name="d_obj")
        m._d_obj = d_obj_var
        d_cons_var = m.addVar(lb=-gp.GRB.INFINITY, ub=0, name="d_cons")
        m._d_cons = d_cons_var
        d_stat_var = m.addVars(self.inst['n_items'], lb=-gp.GRB.INFINITY, ub=0, name="d_stat")
        m._d_stat = d_stat_var

        xi_var = m.addVars(self.inst['xi_dim'], lb=-1, ub=1)
        m._xi = xi_var

        z_var = m.addVar(lb=-gp.GRB.INFINITY, ub=0, name="z")
        m._z = z_var
        # objective
        rev_vector = self.rev_fun(xi_var)
        m._rev_vector = rev_vector
        cost_vector = self.cost_fun(xi_var)
        m._cost_vector = cost_vector

        return m

    def sub_problem(self, m, x):
        obj_part = 0
        cons_part = 0
        stat_part = 0
        for i in range(self.inst['n_items']):
            obj_part += m._rev_vector[i] * x[i]
            cons_part += m._cost_vector[i] * x[i]
            stat_part += m._d_stat[i] * (1 - x[i])
        m.setObjective(m._d_obj*(m._z + obj_part) + m._d_cons*(self.inst['budget'] - cons_part) + stat_part,
                       sense=gp.GRB.MAXIMIZE)

        for c in m.getQConstrs():
            m.remove(c)
        m.update()

        for i in range(self.inst['n_items']):
            m.addConstr(m._d_obj*(-self.inst['k'] * m._rev_vector[i]) + m._d_cons*(m._cost_vector[i]) + m._d_stat[i] <= 0
                        , name=f"dual_cons[{i}]")
        m.update()
        # solve
        m.Params.OutputFlag = 0
        m.Params.NonConvex = 2  # todo: Q is not PSD
        m.optimize()
        xi_sol = np.array([var.X for i, var in m._xi.items()])

        return m.ObjVal, xi_sol

    def cost_fun(self, xi):
        return np.array([(1 + sum(self.inst['phi'][i][j] * xi[j] for j in range(self.inst['xi_dim'])) / 2) *
                         self.inst['c_bar'][i]
                         for i in range(self.inst['n_items'])])

    def rev_fun(self, xi):
        return np.array([(1 + sum(self.inst['psi'][i][j] * xi[j] for j in range(self.inst['xi_dim'])) / 2) *
                         self.inst['r_bar'][i]
                         for i in range(self.inst['n_items'])])

    def get_lower_bound(self):
        return sum([(1 + 1 / 2) * self.inst['r_bar'][i] for i in range(self.inst['n_items'])])

    def get_upper_bound(self):
        return 0

    def get_init_scen(self):
        return np.zeros(self.inst['xi_dim'])

    def get_bigM(self):
        return sum([(1 + 1 / 2) * self.inst['c_bar'][i] for i in range(self.inst['n_items'])])
