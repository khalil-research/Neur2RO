from multiprocessing import Pool, Manager

import gurobipy as gp
import numpy as np

from .two_ro import TwoStageRO


class CapitalBudgeting(TwoStageRO):

    def __init__(self):
        pass

    def solve_second_stage(self, x, xi, instance, gap=0.02, time_limit=600, threads=1, verbose=1):
        """ Solves the second-stage problem for a given first-stage, uncertainty pair.  This function can be infeasible if 
            (x, xi) is over the budget. 
        """
        # first-stage objective
        x_obj = self.rev_fun(xi, instance)
        # second-stage objective
        y_obj = x_obj.copy() * instance['k']

        x0_obj = 0
        y0_obj = 0
        if instance['loans']:
            x0_obj = -instance['l']
            y0_obj = -instance['l'] * instance['m']
        m = gp.Model()

        # first-stage variables
        x_var = m.addVars(instance['n_items'], name="x", vtype="B", obj=-x_obj)
        if instance['loans']:
            x0_var = m.addVar(name="x0", vtype="C", lb=0, ub=instance['max_loan'], obj=-x0_obj)
            x0_var.lb = x[-1]
            x0_var.ub = x[-1]
        for i in range(instance['n_items']):
            x_var[i].lb = x[i]
            x_var[i].ub = x[i]

        # second-stage constraints
        y_var = m.addVars(instance['n_items'], name="y", vtype="B", obj=-y_obj)
        if instance['loans']:
            y0_var = m.addVar(name="y0", vtype="C", lb=0, ub=instance['max_loan'], obj=-y0_obj)

        # store variables in model
        m._x = x_var
        m._y = y_var
        if instance['loans']:
            m._x0 = x0_var
            m._y0 = y0_var

        # constraints
        for i in range(instance['n_items']):
            m.addConstr(y_var[i] + x_var[i] <= 1)

        cost_vector = self.cost_fun(xi, instance)
        if instance['loans']:
            # budget constraints with loans
            lhs = 0
            for i in range(instance['n_items']):
                lhs += cost_vector[i]*x_var[i]
            m.addConstr(lhs <= instance['budget'] + x0_var)

            lhs = 0
            for i in range(instance['n_items']):
                lhs += cost_vector[i]*(x_var[i] + y_var[i])
            m.addConstr(lhs <= instance['budget'] + x0_var + y0_var)
        else:
            lhs = 0
            for i in range(instance['n_items']):
                lhs += cost_vector[i]*(x_var[i] + y_var[i])
            m.addConstr(lhs <= instance['budget'])

        m.setParam("OutputFlag", 0)
        m.optimize()

        # if infeasible
        if m.status != 2:
            return np.nan, np.nan, np.nan, np.nan, m

        # compute first-stage objective
        fs_obj = 0
        for i in range(instance['n_items']):
            fs_obj += x_var[i].x * x_var[i].obj
        if instance['loans']:
            fs_obj += x0_var.x * x0_var.obj

        # compute second-stage objective
        ss_obj = 0
        for i in range(instance['n_items']):
            ss_obj += y_var[i].x * y_var[i].obj
        if instance['loans']:
            ss_obj += y0_var.x * y0_var.obj

        # check uncertain constraint
        # budget constraint
        cost_vector = self.cost_fun(xi, instance)
        lhs = 0
        ss_lhs = 0
        for i in range(instance['n_items']):
            lhs += cost_vector[i] * (x[i] + y_var[i].x)
            ss_lhs += cost_vector[i] * y_var[i].x
        if instance['loans']:
            lhs -= x0_var.x + y0_var.x
            ss_lhs -= y0_var.x

        return fs_obj, ss_obj, lhs, ss_lhs, m


    def get_constr_label(self, x, xi, instance):
        """ Computes constriant label for c(xi)^T x. """
        cost_vector = self.cost_fun(xi, instance)
        if instance['loans']:
            raise Exception("Constr label not implemented for CB w/ loans")
        else:
            lhs_cost = np.dot(x, cost_vector)
        return lhs_cost
        

    def solve_second_stage_slack(self, x, xi, instance, gap=0.02, time_limit=600, threads=1, verbose=1):
        """ Solves the second-stage problem for a given first-stage, uncertainty pair. """
        # first-stage objective
        x_obj = self.rev_fun(xi, instance)
        # second-stage objective
        y_obj = x_obj.copy() * instance['k']

        x0_obj = 0
        y0_obj = 0
        if instance['loans']:
            x0_obj = -instance['l']
            y0_obj = -instance['l'] * instance['m']
        m = gp.Model()

        # first-stage variables
        x_var = m.addVars(instance['n_items'], name="x", vtype="B", obj=-x_obj)
        if instance['loans']:
            x0_var = m.addVar(name="x0", vtype="C", lb=0, ub=instance['max_loan'], obj=-x0_obj)
            x0_var.lb = x[-1]
            x0_var.ub = x[-1]
        for i in range(instance['n_items']):
            x_var[i].lb = x[i]
            x_var[i].ub = x[i]

        # second-stage constraints
        y_var = m.addVars(instance['n_items'], name="y", vtype="B", obj=-y_obj)
        if instance['loans']:
            y0_var = m.addVar(name="y0", vtype="C", lb=0, ub=instance['max_loan'], obj=-y0_obj)

        # store variables in model
        m._x = x_var
        m._y = y_var
        if instance['loans']:
            m._x0 = x0_var
            m._y0 = y0_var

        # slack variable
        bigM = self.get_bigMSlack(instance)
        s_var = m.addVar(name="slack", vtype="C", lb=0, obj=bigM)

        # constraints
        for i in range(instance['n_items']):
            m.addConstr(y_var[i] + x_var[i] <= 1)

        cost_vector = self.cost_fun(xi, instance)
        if instance['loans']:
            # budget constraints with loans
            lhs = 0
            for i in range(instance['n_items']):
                lhs += cost_vector[i]*x_var[i]
            m.addConstr(lhs <= instance['budget'] + x0_var + s_var)

            lhs = 0
            for i in range(instance['n_items']):
                lhs += cost_vector[i]*(x_var[i] + y_var[i])
            m.addConstr(lhs <= instance['budget'] + x0_var + y0_var + s_var)
        else:
            lhs = 0
            for i in range(instance['n_items']):
                lhs += cost_vector[i]*(x_var[i] + y_var[i])
            m.addConstr(lhs <= instance['budget'] + s_var)

        m.setParam("OutputFlag", 0)
        m.optimize()
        # if infeasible
        if m.status != 2:
            return np.nan, np.nan, np.nan, np.nan, m

        # compute first-stage objective
        fs_obj = 0
        for i in range(instance['n_items']):
            fs_obj += x_var[i].x * x_var[i].obj
        if instance['loans']:
            fs_obj += x0_var.x * x0_var.obj

        # compute second-stage objective
        ss_obj = 0
        for i in range(instance['n_items']):
            ss_obj += y_var[i].x * y_var[i].obj
        if instance['loans']:
            ss_obj += y0_var.x * y0_var.obj

        # check uncertain constraint
        # budget constraint
        cost_vector = self.cost_fun(xi, instance)
        lhs = 0
        ss_lhs = 0
        for i in range(instance['n_items']):
            lhs += cost_vector[i] * (x[i] + y_var[i].x)
            ss_lhs += cost_vector[i] * y_var[i].x
        if instance['loans']:
            lhs -= x0_var.x + y0_var.x
            ss_lhs -= y0_var.x
        return fs_obj, ss_obj, lhs, ss_lhs, m

    def solve_constr_violation(self, x, xi, instance):
        """ Solves the second-stage problem for a given first-stage, uncertainty pair. """
        m = gp.Model()

        eta_var = m.addVar(lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, name="zeta", obj=-1)
        # first-stage variables
        x_var = m.addVars(instance['n_items'], name="x", vtype="B")
        if instance['loans']:
            x0_var = m.addVar(name="x0", vtype="C", lb=0, ub=instance['max_loan'])
            x0_var.lb = x[-1]
            x0_var.ub = x[-1]
        for i in range(instance['n_items']):
            x_var[i].lb = x[i]
            x_var[i].ub = x[i]

        # second-stage constraints
        y_var = m.addVars(instance['n_items'], name="y", vtype="B")
        if instance['loans']:
            y0_var = m.addVar(name="y0", vtype="C", lb=0, ub=instance['max_loan'])

        # store variables in model
        m._x = x_var
        m._y = y_var
        if instance['loans']:
            m._x0 = x0_var
            m._y0 = y0_var

        # budget constraints
        cost_vector = self.cost_fun(xi, instance)
        rhs = sum(cost_vector[i] * (x_var[i] + y_var[i]) for i in range(instance['n_items'])) - instance['budget']
        if instance['loans']:
            rhs += - x0_var - y0_var
        m.addConstr(eta_var <= -rhs)

        if instance['loans']:
            rhs = sum(cost_vector[i] * x_var[i] for i in range(instance['n_items'])) - instance['budget'] - x0_var
            m.addConstr(eta_var <= -rhs)

        m.setParam("OutputFlag", 0)
        m.optimize()

        # budget constraint
        cost_vector = self.cost_fun(xi, instance)
        lhs = 0
        ss_lhs = 0
        for i in range(instance['n_items']):
            lhs += cost_vector[i] * (x[i] + y_var[i].x)
            ss_lhs += cost_vector[i] * y_var[i].x
        if instance['loans']:
            lhs -= x0_var.x + y0_var.x
            ss_lhs -= y0_var.x
        return lhs, ss_lhs

    def sample_new_inst(self, cfg, seed):
        """ Samples data, does not use instances from paper. """
        np.random.seed(seed)
        inst = {}

        # ## SAMPLE EVERYTHING
        #n_items = [10]
        n_items = cfg.n_items
        k = [.8]
        loans = [0]
        l = [.12]
        m = [1.2]
        xi_dim = [4]

        # cfg.n_items = int(np.random.choice(n_items))
        # cfg.k = np.random.choice(k)
        # cfg.loans = np.random.choice(loans)
        # cfg.l = np.random.choice(l)
        # cfg.m = np.random.choice(m)
        # cfg.xi_dim = np.random.choice(xi_dim)

        # inst['n_items'] = cfg.n_items
        # inst['k'] = cfg.k
        # inst['loans'] = cfg.loans
        # inst['l'] = cfg.l
        # inst['m'] = cfg.m
        # inst['xi_dim'] = cfg.xi_dim

        inst['n_items'] = int(np.random.choice(n_items))
        inst['k'] = np.random.choice(k)
        inst['loans'] = np.random.choice(loans)
        inst['l'] = np.random.choice(l)
        inst['m'] = np.random.choice(m)
        inst['xi_dim'] = np.random.choice(xi_dim)

        c_bar = np.random.uniform(1, 10, size=inst['n_items'])
        r_bar = c_bar / 5
        budget = sum(c_bar) / 2
        max_loan = sum(c_bar) * 1.5 - budget

        # define phi and psi, with unit simplex randomness
        phi_vector = dict()
        psi_vector = dict()

        for p in np.arange(inst['n_items']):
            x = {0: 0}
            y = {0: 0}
            for i in range(1, inst['xi_dim']):
                x[i] = np.random.uniform(0, 1)
                y[i] = np.random.uniform(0, 1)
            x[inst['xi_dim']] = 1
            y[inst['xi_dim']] = 1
            x_values = sorted(x.values())
            y_values = sorted(y.values())
            x = dict()
            for i in np.arange(len(x_values)):
                x[i] = x_values[i]
                y[i] = y_values[i]
            phi = dict()
            psi = dict()
            for i in range(1, inst['xi_dim'] + 1):
                phi[i - 1] = x[i] - x[i - 1]
                psi[i - 1] = y[i] - y[i - 1]
            phi_vector[p] = list(phi.values())
            psi_vector[p] = list(psi.values())

        inst['c_bar'] = c_bar
        inst['r_bar'] = r_bar
        inst['budget'] = budget
        inst['max_loan'] = max_loan
        inst['phi'] = phi_vector
        inst['psi'] = psi_vector

        return inst

    def cost_fun(self, xi, inst):
        return np.array([(1 + sum(inst['phi'][i][j] * xi[j] for j in range(inst['xi_dim'])) / 2) * inst['c_bar'][i]
                         for i in range(inst['n_items'])])

    def rev_fun(self, xi, inst):
        return np.array([(1 + sum(inst['psi'][i][j] * xi[j] for j in range(inst['xi_dim'])) / 2) * inst['r_bar'][i]
                        for i in range(inst['n_items'])])

    def get_bigM(self, inst):
        return sum([(1 + 1 / 2) * inst['c_bar'][i] for i in range(inst['n_items'])])

    def get_bigMSlack(self, inst):
        return sum([(1 + 1 / 2) * inst['r_bar'][i] for i in range(inst['n_items'])])


    # TODO: MAX MIN PROBLEM NOT WORKING FOR CONSTRAINT UNCERTAINTY
    # def solve_pricing_problem_subway(self, x, xi, z):
    #     """ Solves pricing problem, i.e., gets y corresponding to minimization problem. """
    #     m = gp.Model()
    #     m.setParam("OutputFlag", 0)
    #     # first-stage variables
    #     x_var = m.addVars(inst['n_items'], name="x", vtype="B")
    #     if inst['loans']:
    #         x0_var = m.addVar(name="x0", vtype="C", lb=0, ub=inst['max_loan'])
    #         x0_var.lb = x[-1]
    #         x0_var.ub = x[-1]
    #     for i in range(inst['n_items']):
    #         x_var[i].lb = x[i]
    #         x_var[i].ub = x[i]
    #
    #     # second-stage constraints
    #     y_var = m.addVars(inst['n_items'], name="y", vtype="B")
    #     if inst['loans']:
    #         y0_var = m.addVar(name="y0", vtype="C", lb=0, ub=inst['max_loan'])
    #
    #     # store variables in model
    #     m._x = x_var
    #     m._y = y_var
    #     if inst['loans']:
    #         m._x0 = x0_var
    #         m._y0 = y0_var
    #
    #     # violation vars
    #     bigM = self.get_bigM()
    #     num_cons = 3 if inst['loans'] else 2
    #     v_var = m.addVars(num_cons, name="v", vtype="B")
    #     m._v = v_var
    #     zeta_var = m.addVar(lb=-gp.GRB.INFINITY, name="zeta", obj=-1)
    #     m._zeta = zeta_var
    #
    #     # CONSTRAINTS
    #     m.addConstr(sum(v_var[l] for l in range(num_cons)) == 1)
    #     # objective constraint
    #     rev_vector = self.rev_fun(xi)
    #     rhs = 0
    #     for i in range(inst['n_items']):
    #         rhs += rev_vector[i] * (x_var[i] + inst['k'] * y_var[i])
    #     if inst['loans']:
    #         rhs += inst['l'] * (x0_var + inst['m'] * y0_var)  # prediction of y0 is already included in pred_out
    #     # try to find a objective that is less than z
    #     # so difference between rhs and z needs to be as big as possible
    #     m.addConstr(zeta_var + bigM * (v_var[0] - 1) <= z + rhs)
    #
    #     # budget constraints
    #     cost_vector = self.cost_fun(xi)
    #     rhs = sum(cost_vector[i] * (x_var[i] + y_var[i]) for i in range(inst['n_items'])) - inst['budget']
    #     if inst['loans']:
    #         rhs += - x0_var - y0_var
    #     m.addConstr(zeta_var + bigM * (v_var[1] - 1) <= rhs)
    #
    #     if inst['loans']:
    #         m.addConstr(zeta_var + bigM * (v_var[2] - 1)
    #                     <= sum(cost_vector[i] * x_var[i] for i in range(inst['n_items']))
    #                     - inst['budget'] - x0_var)
    #
    #     for i in range(inst['n_items']):
    #         m.addConstr(y_var[i] + x_var[i] <= 1)
    #
    #     m.optimize()
    #     y = np.array([var.x for var in y_var.values()])
    #     zeta_sol = zeta_var.x
    #     return y, zeta_sol

    # def evaluate_first_stage_solution(self, x, prob):
    #     """ Evaluates the first-stage solution by solving the pricing problem. """
    #     def solve_pricing_problem(x, xi):
    #         _, m_price = self.solve_second_stage(x, xi, inst)
    #         y = list(map(lambda x: x.x, m_price._y.values()))
    #         return y
    #
    #     def get_pricing_constr(xi_var, z_var, x, y, m):
    #         """ Computes constraints based on pricing problem """
    #         if inst['loans']:
    #             x0 = x[-1]
    #             x = x[:-1]
    #             y0 = y[-1]
    #             y = y[:-1]
    #         # objective constraint
    #         lhs1 = 0
    #         rev_vector = self.rev_fun(xi_var)
    #         for i in range(inst['n_items']):
    #             lhs1 += rev_vector[i]*(x[i] + y[i] * inst['k'])
    #         if inst['loans']:
    #             lhs1 += - inst['l'] * (x0 + y0 * inst['m'])
    #         # budget constraint
    #         cost_vector = self.cost_fun(xi_var)
    #         lhs2 = 0
    #         for i in range(inst['n_items']):
    #             lhs2 += cost_vector[i] * (x[i] + y[i])
    #         rhs2 = inst['budget']
    #         if inst['loans']:
    #             rhs2 += x0 + y0
    #         # add constraints
    #         m.addConstr(-lhs1 >= z_var)
    #         m.addConstr(lhs2 <= rhs2)
    #         return m
    #
    #     inst = prob
    #     if inst['loans']:
    #         x0 = x[-1]
    #         x = x[:-1]
    #     # initialize model pricing problem
    #     m_eval = gp.Model()
    #     m_eval.setParam("OutputFlag", 0)
    #     m_eval.setObjective(0, sense=gp.GRB.MAXIMIZE)
    #     # variables
    #     xi_var = m_eval.addVars(inst['xi_dim'], name='xi', vtype='C', lb=-1, ub=1)
    #     z_var = m_eval.addVar(name='z', obj=1, lb=-gp.GRB.INFINITY, ub=1e-10, vtype='C')
    #
    #     # constraints
    #     if inst['loans']:
    #         # budget constraints with loans
    #         lhs = 0
    #         cost_vector = self.cost_fun(xi_var)
    #         for i in range(inst['n_items']):
    #             lhs += cost_vector[i]*x[i]
    #         m_eval.addConstr(lhs <= inst['budget'] + x0)
    #
    #     # pricing problem
    #     print(f"  Solving Pricing Problem")
    #     pp_iter = 0
    #     added_y = []
    #
    #     m_eval.optimize()
    #     xi = list(map(lambda x: x.x, xi_var.values()))
    #     while True:
    #         print(f"    Iteration: {pp_iter + 1} ")
    #
    #         if pp_iter > 500:
    #             return None, None, None
    #
    #         y = solve_pricing_problem(x, xi)
    #         added_y.append(y)
    #         m_eval = get_pricing_constr(xi_var, z_var, x, y, m_eval)
    #         m_eval.update()
    #
    #         m_eval.optimize()
    #
    #         xi = list(map(lambda x: x.x, xi_var.values()))
    #
    #         # CHECK IF ROBUST
    #         zeta = self.solve_robust_check(x, added_y, z_var.x)  # get all y-s
    #         print(f"zeta = {zeta}")
    #         if zeta < 1e-10:
    #             print("    Done!")
    #             break
    #
    #         pp_iter += 1
    #
    #     return z_var.x, xi, added_y
    #
    # def solve_robust_check(self, x, y, obj):
    #     if inst['loans']:
    #         x_var, x0_var = x
    #         y_var, y0_var = y
    #     else:
    #         x_var = x
    #         y_var = y
    #     num_ss = len(y_var)
    #     # model
    #     m = gp.Model("Separation Problem")
    #     m.Params.OutputFlag = 0
    #     # variables
    #     zeta_var = m.addVar(lb=-gp.GRB.INFINITY, name="zeta", obj=-1)
    #     xi_var = m.addVars(inst['xi_dim'], lb=-1, ub=1, name="xi")
    #     num_cons = 3 if inst['loans'] else 2
    #     v_index = [(k, l) for k in range(num_ss) for l in range(num_cons)]
    #     v_var = m.addVars(v_index, name="z", vtype="B")
    #
    #     # violation constraint
    #     bigM = self.get_bigM()
    #     for k in range(num_ss):
    #         m.addConstr(sum(v_var[k, l] for l in range(num_cons)) == 1)
    #         # objective constraint
    #         rev_vector = self.rev_fun(xi_var)
    #         rhs = 0
    #         for i in range(inst['n_items']):
    #             rhs += rev_vector[i] * (x_var[i] + inst['k'] * y_var[k][i])
    #         if inst['loans']:
    #             rhs += inst['l'] * (x0_var + inst['m'] * y0_var[k])  # prediction of y0 is already included in pred_out
    #         # try to find a objective that is less than z
    #         # so difference between rhs and z needs to be as big as possible
    #         m.addConstr(zeta_var + bigM * (v_var[k, 0] - 1) <= -rhs - obj)
    #
    #         # budget constraints
    #         cost_vector = self.cost_fun(xi_var)
    #         rhs = sum(cost_vector[i] * (x_var[i] + y_var[k][i]) for i in range(inst['n_items'])) - inst['budget']
    #         if inst['loans']:
    #             rhs += - x0_var - y0_var[k]
    #         m.addConstr(zeta_var + bigM * (v_var[k, 1] - 1) <= rhs)
    #
    #         if inst['loans']:
    #             m.addConstr(zeta_var + bigM * (v_var[k, 2] - 1)
    #                         <= sum(cost_vector[i] * x_var[i] for i in range(inst['n_items']))
    #                         - inst['budget'] - x0_var)
    #     # solve
    #     m.optimize()
    #     zeta_sol = zeta_var.X
    #     xi_sol = np.array([var.X for i, var in xi_var.items()])
    #     return zeta_sol