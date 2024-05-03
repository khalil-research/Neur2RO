import gurobipy as gp
import numpy as np

from .ss_relax import SecondStageRelax


class CapitalBudgetingRelax(SecondStageRelax):
    def __init__(self, problem_config):
        self.cfg = problem_config
        self.rng = np.random.RandomState()
        super().__init__()

    def set_inst(self, inst):
        self.inst = inst

    def adversarial_problem_max(self, x):
        # model
        m = gp.Model("Adversarial Problem")
        m.Params.OutputFlag = 0
        # variables
        d_obj_var = m.addVar(lb=-gp.GRB.INFINITY, ub=0, name="d_obj")
        d_cons_var = m.addVar(lb=-gp.GRB.INFINITY, ub=0, name="d_cons")
        d_stat_var = m.addVars(self.inst['n_items'], lb=-gp.GRB.INFINITY, ub=0, name="d_stat")

        xi_var = m.addVars(self.inst['xi_dim'], lb=-1, ub=1)

        z_var = m.addVar(lb=-gp.GRB.INFINITY, ub=0, name="z")

        obj_part = 0
        cons_part = 0
        stat_part = 0
        rev_vector = self.rev_fun(xi_var)
        cost_vector = self.cost_fun(xi_var)
        for i in range(self.inst['n_items']):
            obj_part += rev_vector[i] * x[i]
            cons_part += cost_vector[i] * x[i]
            stat_part += d_stat_var[i] * (1 - x[i])
        # m.setObjective(d_obj_var*(z_var + obj_part) + d_cons_var*(self.inst['budget'] - cons_part) + stat_part,
        #                sense=gp.GRB.MAXIMIZE)

        for i in range(self.inst['n_items']):
            m.addConstr(d_obj_var*(-self.inst['k'] * rev_vector[i]) + d_cons_var*(cost_vector[i]) + d_stat_var[i] <= 0)
        # solve
        m.Params.OutputFlag = 0
        m.Params.NonConvex = 2
        m.optimize()

        return m.ObjVal

    def adversarial_problem_min(self, x):
        # model
        m = gp.Model("Adversarial Problem")
        m.Params.OutputFlag = 0
        # variables
        d_obj_var = m.addVar(lb=-gp.GRB.INFINITY, ub=0, name="d_obj")
        d_cons_var = m.addVar(lb=0, name="d_cons")
        d_stat_var = m.addVars(self.inst['n_items'], lb=0, name="d_stat")

        xi_var = m.addVars(self.inst['xi_dim'], lb=-1, ub=1, name="xi")

        z_var = m.addVar(lb=0, ub=self.get_lower_bound(), name="z")

        obj_part = 0
        cons_part = 0
        stat_part = 0
        rev_vector = self.rev_fun(xi_var)
        cost_vector = self.cost_fun(xi_var)
        for i in range(self.inst['n_items']):
            obj_part += rev_vector[i] * x[i]
            cons_part += cost_vector[i] * x[i]
            stat_part += d_stat_var[i] * (1 - x[i])
        m.setObjective((z_var - obj_part)*d_obj_var + d_cons_var*(self.inst['budget'] - cons_part) + stat_part)

        for i in range(self.inst['n_items']):
            m.addConstr(d_obj_var*(self.inst['k'] * rev_vector[i]) + d_cons_var*(cost_vector[i]) + d_stat_var[i] >= 0)
        m.update()
        # solve
        m.Params.OutputFlag = 0
        m.Params.NonConvex = 2  # Q is not PSD
        m.write("adv_problem.mps")
        m.optimize()

        return - m.ObjVal

    def adversarial_problem(self, x):
        # model
        m = gp.Model("Adversarial Problem")
        m.Params.OutputFlag = 0
        # variables
        d_obj_var = m.addVar(lb=-gp.GRB.INFINITY, ub=0, name="d_obj")
        d_cons_var = m.addVar(lb=0, name="d_cons")
        d_stat_var = m.addVars(self.inst['n_items'], lb=0, name="d_stat")

        xi_var = m.addVars(self.inst['xi_dim'], lb=-1, ub=1, name="xi")

        obj_part = 0
        cons_part = 0
        stat_part = 0
        rev_vector = self.rev_fun(xi_var)
        cost_vector = self.cost_fun(xi_var)
        for i in range(self.inst['n_items']):
            obj_part += rev_vector[i] * x[i]
            cons_part += cost_vector[i] * x[i]
            stat_part += d_stat_var[i] * (1 - x[i])
        m.setObjective((-obj_part)*d_obj_var + d_cons_var*(self.inst['budget'] - cons_part) + stat_part)

        for i in range(self.inst['n_items']):
            m.addConstr(d_obj_var*(self.inst['k'] * rev_vector[i]) + d_cons_var*(cost_vector[i]) + d_stat_var[i] >= 0)

        # z constraint
        m.addConstr(-d_obj_var >= 1)
        m.update()
        # solve
        m.Params.OutputFlag = 0
        m.Params.NonConvex = 2  # Q is not PSD
        m.write("adv_problem.mps")
        m.optimize()

        return - m.ObjVal

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
