import os

from gurobipy import GRB
import gurobipy as gp
import pickle as pkl
import numpy as np
import multiprocessing as mp

from ro.utils.cb import get_path
from .ssa import SSAProblem


class CapitalBudgetingSSA(SSAProblem):
    def __init__(self, problem_config):
        self.cfg = problem_config
        self.rng = np.random.RandomState()

    def set_inst(self, inst):
        self.inst = inst

    def set_eval_problem(self, x):
        m = gp.Model()

        # variables
        y_var = m.addVars(self.inst['n_items'], name="y", vtype="B")
        m._y = y_var

        xi_var = m.addVars(self.inst['xi_dim'], name="xi", vtype="C", lb=-1, ub=1)
        m._xi = xi_var

        # objective
        rev_vector = self.rev_fun(xi_var)
        obj = 0
        for i in range(self.inst['n_items']):
            obj += rev_vector[i]*(x[i] + self.inst['k']*y_var[i])

        m.setObjective(-obj)

        # constraints
        for i in range(self.inst['n_items']):
            m.addConstr(y_var[i] + x[i] <= 1, name=f"static_cons[{i}]")

        cost_vector = self.cost_fun(xi_var)
        lhs = 0
        for i in range(self.inst['n_items']):
            lhs += cost_vector[i] * (x[i] + y_var[i])
        m.addConstr(lhs <= self.inst['budget'], name="budget_cons")

        return m

    def eval_problem(self, xi, m, x=None):
        m.setParam("OutputFlag", 0)
        m.setParam("Threads", 1)
        for i, xi_val in enumerate(xi):
            gp_var = m.getVarByName(f'xi[{i}]')
            gp_var.lb = xi_val
            gp_var.ub = xi_val

        m.optimize()
        return m.ObjVal

    def mp_queue_eval(self, x, xi_q, sol_mp, i):
        print(f"  Starting process {i}")
        m = self.set_eval_problem(x)
        m.update()
        while True:
            if xi_q.empty():
                break
   
            xi = xi_q.get()
            sol = self.eval_problem(xi, m)
            sol_mp.append(sol)
        print(f'  Done process {i}')
        return None

    def get_obj(self, x, scens, n_procs=1):

        if n_procs > 1:

            # initialize queue xi values 
            xi_q = mp.Queue()
            for xi in scens:
                xi_q.put(xi)

            # initialize list for solutions
            manager = mp.Manager()
            sol_mp = manager.list()

            # start processes
            procs = []
            for i in range(n_procs):
                proc = mp.Process(target=self.mp_queue_eval, args=(x, xi_q, sol_mp, i))
                procs.append(proc)
                proc.start()
            
            # collect finished processes
            for proc in procs:
                proc.join()
                proc.close()

            return max(sol_mp)

        else:

            # process each job sequentially otherwise
            m = self.set_eval_problem(x)
            m.update()
            sol = []
            for xi in scens:
                new_sol = self.eval_problem(xi, m)
                sol.append(new_sol)

            return max(sol)


    def cost_fun(self, xi):
        return np.array([(1 + sum(self.inst['phi'][i][j] * xi[j] for j in range(self.inst['xi_dim'])) / 2) *
                         self.inst['c_bar'][i]
                         for i in range(self.inst['n_items'])])


    def rev_fun(self, xi):
        return np.array([(1 + sum(self.inst['psi'][i][j] * xi[j] for j in range(self.inst['xi_dim'])) / 2) *
                         self.inst['r_bar'][i]
                         for i in range(self.inst['n_items'])])