import os

from gurobipy import GRB
import gurobipy as gp
import numpy as np
import time

from .kadapt import KAdaptability


class CapitalBudgetingKAdapt(KAdaptability):
    def __init__(self, problem_config):
        self.cfg = problem_config
        self.rng = np.random.RandomState(self.cfg.seed)
        super().__init__()

    def set_inst(self, inst):
        self.inst = inst
        self.lower_bound = self.get_lower_bound()
        self.upper_bound = self.get_upper_bound()
        self.init_scen = self.get_init_scen()
        self.bigM = self.get_bigM()
        self.inst_seed = self.inst['seed']

    def run_algorithm(self, K, time_limit=30*60, x_eval=None):
        if x_eval is None:
            save_path = self.cfg.data_path + "cb/k_adapt_res/"
            os.makedirs(save_path, exist_ok=True)
            self.results_path = save_path + f"res_{self.inst['inst_name']}_K{K}.pkl"
        else:
            save_path = self.cfg.data_path + "cb/eval/k_adapt/"
            os.makedirs(save_path, exist_ok=True)
            self.results_path = save_path + f"res_{self.inst['inst_name']}_K{K}.pkl"

        results = self.algorithm(K, time_limit, x_eval=x_eval)
        return results['obj']

    def master_problem_update(self, K, k, xi, m):
        # load variables
        theta = m.getVarByName("theta")

        x_var = m._x
        y_var = m._y
        if self.inst['loans']:
            x0_var = m._x0
            y0_var = m._y0

        # add new constraints
        # objective constraint
        lhs = 0
        rev_vector = self.rev_fun(xi)
        for i in range(self.inst['n_items']):
            lhs += rev_vector[i] * (x_var[i] + self.inst['k'] * y_var[k][i])
        if self.inst['loans']:
            lhs += - self.inst['l'] * (x0_var + self.inst['m'] * y0_var[k])
        m.addConstr(-lhs <= theta)

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
                lhs += cost_vector[i] * (x_var[i] + y_var[k][i])
            m.addConstr(lhs <= self.inst['budget'] + x0_var + y0_var[k])
        else:
            lhs = 0
            for i in range(self.inst['n_items']):
                lhs += cost_vector[i] * (x_var[i] + y_var[k][i])
            m.addConstr(lhs <= self.inst['budget'])

        # solve
        m.setParam("TimeLimit", self.time_left())
        m.optimize()

        if m.status == 9:   # time limit reached
            return None

        # get results
        theta_sol = theta.x
        x_sol = np.array([var.X for i, var in x_var.items()])
        y_sol = []
        for k in range(K):
            y_sol.append(np.array([var.X for i, var in y_var[k].items()]))
        if self.inst['loans']:
            x_sol = [x_sol, x0_var.X]
            y_sol = [y_sol, np.array([var.X for i, var in y0_var.items()])]

        return [theta_sol, x_sol, y_sol, m]

    def master_problem_build(self, K, tau, x_eval=None):
        # first-stage objective
        m = gp.Model()

        # objective
        theta = m.addVar(lb=self.lower_bound, ub=0, name="theta", obj=1)

        # first-stage variables
        x_var = m.addVars(self.inst['n_items'], name="x", vtype="B")
        m._x = x_var

        if x_eval is not None:
            for i in range(self.inst['n_items']):
                m._x[i].lb = x_eval[i]
                m._x[i].ub = x_eval[i]

        if self.inst['loans']:
            x0_var = m.addVar(name="x0", vtype="C", lb=0, ub=self.inst['max_loan'])
            m._x0 = x0_var
        # second-stage constraints
        y_var = dict()
        for k in np.arange(K):
            y_var[k] = m.addVars(self.inst['n_items'], name=f"y_{k}", vtype="B")
        m._y = y_var
        if self.inst['loans']:
            y0_var = m.addVars(K, name="y0", vtype="C", lb=0, ub=self.inst['max_loan'])
            m._y0 = y0_var

        # constraints
        for k in range(K):
            # static constraint
            for i in range(self.inst['n_items']):
                m.addConstr(y_var[k][i] + x_var[i] <= 1)
            for xi in tau[k]:
                # objective constraint
                lhs = 0
                rev_vector = self.rev_fun(xi)
                for i in range(self.inst['n_items']):
                    lhs += rev_vector[i]*(x_var[i] + self.inst['k']*y_var[k][i])
                if self.inst['loans']:
                    lhs += - self.inst['l'] * (x0_var + self.inst['m'] * y0_var[k])
                m.addConstr(-lhs <= theta)

                # uncertain constraints
                cost_vector = self.cost_fun(xi)
                if self.inst['loans']:
                    # budget constraints with loans
                    lhs = 0
                    for i in range(self.inst['n_items']):
                        lhs += cost_vector[i]*x_var[i]
                    m.addConstr(lhs <= self.inst['budget'] + x0_var)

                    lhs = 0
                    for i in range(self.inst['n_items']):
                        lhs += cost_vector[i]*(x_var[i] + y_var[k][i])
                    m.addConstr(lhs <= self.inst['budget'] + x0_var + y0_var[k])
                else:
                    lhs = 0
                    for i in range(self.inst['n_items']):
                        lhs += cost_vector[i]*(x_var[i] + y_var[k][i])
                    m.addConstr(lhs <= self.inst['budget'])

        m.setParam("OutputFlag", 0)
        m.setParam("TimeLimit", self.time_left())
        m.optimize()

        if m.status == 9:   # time limit reached
            return None
        # get results
        theta_sol = theta.x
        x_sol = np.array([var.X for i, var in x_var.items()])
        y_sol = []
        for k in range(K):
            y_sol.append(np.array([var.X for i, var in y_var[k].items()]))
        if self.inst['loans']:
            x_sol = [x_sol, x0_var.X]
            y_sol = [y_sol, np.array([var.X for i, var in y0_var.items()])]

        return [theta_sol, x_sol, y_sol, m]

    # SEPARATION FUN = SUBPROBLEM
    def sub_problem(self, K, x_input, y_input, theta, tau):
        if self.inst['loans']:
            x, x0 = x_input
            y, y0 = y_input
        else:
            x = x_input
            y = y_input

        # model
        m = gp.Model("Separation Problem")
        m.Params.OutputFlag = 0
        # variables
        zeta_var = m.addVar(lb=-self.bigM, name="zeta", obj=-1)
        xi_var = m.addVars(self.inst['xi_dim'], lb=-1, ub=1, name="xi")
        if self.inst['loans']:
            num_cons = 3
        else:
            num_cons = 2
        z_index = [(k, i) for k in np.arange(K) for i in range(num_cons)]
        z_var = m.addVars(z_index, name="z", vtype=GRB.BINARY)
        for k in np.arange(K):
            # z constraint
            m.addConstr(sum(z_var[k, l] for l in range(num_cons)) == 1)
            if len(tau[k]) > 0:
                # objective constraint
                rev_vector = self.rev_fun(xi_var)
                rhs = 0
                for i in range(self.inst['n_items']):
                    rhs += rev_vector[i] * (x[i] + self.inst['k'] * y[k][i])

                if self.inst['loans']:
                    rhs += self.inst['l'] * (x0 + self.inst['m'] * y0[k])
                rhs += theta
                m.addConstr(zeta_var + self.bigM * (z_var[k, 0] - 1) <= -rhs)

                # budget constraints
                cost_vector = self.cost_fun(xi_var)
                rhs = sum(cost_vector[i] * (x[i] + y[k][i]) for i in range(self.inst['n_items'])) - self.inst['budget']
                if self.inst['loans']:
                    rhs += - x0 - y0[k]
                m.addConstr(zeta_var + self.bigM * (z_var[k, 1] - 1) <= rhs)

                if self.inst['loans']:
                    m.addConstr(zeta_var + self.bigM * (z_var[k, 2] - 1)
                                <= sum(cost_vector[i] * x[i] for i in range(self.inst['n_items']))
                                - self.inst['budget'] - x0)

        # solve
        m.setParam("TimeLimit", self.time_left())
        m.optimize()

        if m.status == 9:   # time limit reached
            return None

        zeta_sol = zeta_var.X
        xi_sol = np.array([var.X for i, var in xi_var.items()])

        return [zeta_sol, xi_sol]

    def cost_fun(self, xi):
        return np.array([(1 + sum(self.inst['phi'][i][j] * xi[j] for j in range(self.inst['xi_dim'])) / 2) *
                         self.inst['c_bar'][i]
                         for i in range(self.inst['n_items'])])

    def rev_fun(self, xi):
        return np.array([(1 + sum(self.inst['psi'][i][j] * xi[j] for j in range(self.inst['xi_dim'])) / 2) *
                         self.inst['r_bar'][i]
                         for i in range(self.inst['n_items'])])

    def get_lower_bound(self):
        return - sum([(1 + 1 / 2) * self.inst['r_bar'][i] for i in range(self.inst['n_items'])])

    def get_upper_bound(self):
        return 0

    def get_init_scen(self):
        return np.zeros(self.inst['xi_dim'])

    def get_bigM(self):
        return sum([(1 + 1 / 2) * self.inst['c_bar'][i] for i in range(self.inst['n_items'])])
