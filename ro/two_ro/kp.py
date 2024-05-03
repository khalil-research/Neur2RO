from multiprocessing import Pool, Manager

import gurobipy as gp
import numpy as np

from .two_ro import TwoStageRO


class Knapsack(TwoStageRO):

    def __init__(self):
        pass

    def solve_second_stage(self, x, xi, instance, gap=0.02, time_limit=600, threads=1, verbose=1):
        """ Solves the second-stage problem for a given first-stage, uncertainty pair. """
        # first-stage objective
        # self.inst = prob 

        x_obj = instance['f'] - instance['p_bar']

        # second-stage objective
        y_obj = np.multiply(instance['p_hat'], xi) - instance['f']
        r_obj = - np.multiply(instance['p_hat'], xi)
        
        m = gp.Model()
        
        # first-stage variables
        x_var = m.addVars(instance['n_items'], name="x", vtype="B", obj=x_obj)
        for i in range(instance['n_items']):
            x_var[i].lb = x[i]
            x_var[i].ub = x[i]

        # second-stage constraints
        y_var = m.addVars(instance['n_items'], name="y", vtype="B", obj=y_obj)
        r_var = m.addVars(instance['n_items'], name="r", vtype="B", obj=r_obj)

        # store variables in model
        m._x = x_var
        m._y = y_var
        m._r = r_var
        
        # constraints
        lhs = 0
        for i in range(instance['n_items']):
            lhs += instance['c'][i] * y_var[i] + instance['t'][i] * r_var[i]
        m.addConstr(lhs <= instance['C'])

        # constraints
        for i in range(instance['n_items']):
            m.addConstr(y_var[i] <= x_var[i])
            m.addConstr(r_var[i] <= y_var[i])
        m.setParam("OutputFlag", 0)
        m.optimize()
        
        # compute first-stage objective
        fs_obj = 0
        for i in range(instance['n_items']):
            fs_obj += x_var[i].x * x_var[i].obj
            
        # compute second-stage objective
        ss_obj = 0
        for i in range(instance['n_items']):
            ss_obj += y_var[i].x * y_var[i].obj
            ss_obj += r_var[i].x * r_var[i].obj
        
        return fs_obj, ss_obj, m


    def get_first_stage_obj(self, x, instance):
        """ Gets first-stage objective. """
        x_obj = instance['f'] - instance['p_bar']
        return np.dot(x_obj, x)


    def evaluate_first_stage_solution(self, x, instance):
        """ Evaluates the first-stage solution by solving the pricing problem.  Note that this can only be done easily 
            for problems without constraint uncertainty.  
        """
        def solve_pricing_problem(x, xi, instance):
            """ Solves pricing problem, i.e., gets y,r corresponding to minimization problem. """
            _, ss_obj, m_price = self.solve_second_stage(x, xi, instance)
            y = list(map(lambda x: x.x, m_price._y.values()))
            r = list(map(lambda x: x.x, m_price._r.values()))
            return ss_obj, y, r

        def get_pricing_constr(xi_var, z_var, y, r, instance):
            """ Computes constraints based on pricing problem """
            rhs = 0
            for i in range(instance['n_items']):
                t1 = instance['p_hat'][i] * xi_var[i] * y[i]
                t2 = - instance['f'][i] * y[i]
                t3 = - instance['p_hat'][i] * xi_var[i] * r[i]
                rhs += t1 + t2 + t3
            return z_var <= rhs

        # self.inst = prob 

        # get first-stage cost
        fs_obj = self.get_first_stage_obj(x, instance)
        
        # get budget 
        budget = instance['max_budget']

        # initialize model pricing problem
        m_eval = gp.Model()
        m_eval.setParam("OutputFlag", 0)
        m_eval.setObjective(0, sense=gp.GRB.MAXIMIZE)
        
        # variables
        xi_var = m_eval.addVars(instance['n_items'], name='xi', vtype='C', lb=0, ub=1)
        z_var = m_eval.addVar(name='z', obj=1, lb=-gp.GRB.INFINITY, ub=1e10, vtype='C')
        
        # constraints
        lhs = 0
        for i in range(instance['n_items']):
            lhs += xi_var[i]
        m_eval.addConstr(lhs <= budget)
        
        # pricing problem
        print(f"  Solving Pricing Problem with budget Gamma={budget}")
        pp_iter = 0
        while True:
            if (pp_iter + 1) % 100 == 0: 
                print(f"    Iteration: {pp_iter+1} ")
            
            if pp_iter > 10000:
                 return None, None, None, None

            m_eval.optimize()

            xi = list(map(lambda x: x.x, xi_var.values()))
            z = z_var.x

            ss_obj, y, r = solve_pricing_problem(x, xi, instance)
            
            if ss_obj <= z - 1e-3:
                constr = get_pricing_constr(xi_var, z_var, y, r, instance)
                m_eval.addConstr(constr)
            else:
                print("    Done!")
                break
                
            pp_iter += 1
                
        fs_obj, ss_obj, m_final = self.solve_second_stage(x, xi, instance)
        obj = - (fs_obj + ss_obj)
        
        return obj, xi, y, r


    def sample_new_inst(self, cfg, seed):
            """ Samples new instance using procedure from paper, does not use instances from paper. """
            np.random.seed(seed)

            inst = {}

            # ## SAMPLE EVERYTHING
            # n_items = [20, 30, 40, 50, 60, 70, 80]
            # correlation = ['UN', 'WC', 'ASC', 'SC']
            # delta = [0.1, 0.5, 1.0]
            # h = [40,80]
            # budget_factor = [0.1, 0.15, 0.20]

            # cfg.n_items = int(np.random.choice(n_items))
            # cfg.correlation = np.random.choice(correlation)
            # cfg.delta = np.random.choice(delta)
            # cfg.h = np.random.choice(h)
            # cfg.budget_factor = np.random.choice(budget_factor)

            n_items = int(np.random.choice(cfg.n_items))
            correlation = np.random.choice(cfg.correlation)
            delta = np.random.choice(cfg.delta)
            h = np.random.choice(cfg.h)
            budget_factor = np.random.choice(cfg.budget_factor)

            R = cfg.R 
            H = cfg.H

            c = np.random.uniform(1, R, size=n_items)
            C = h/(H+1) * np.sum(c)

            # sample profits 
            # Uncorrelated
            p_bar = np.zeros(n_items)
            if correlation == 'UN': 
                p_bar = np.random.uniform(1, R, size=n_items)

            # Weakly correlated
            elif correlation == 'WC':
                for i in range(n_items):
                    low  = c[i] - R/20
                    high = c[i] + R/20
                    p_bar[i] = np.random.uniform(low, high)
            # Almost strongly correlated
            elif correlation == 'ASC':
                for i in range(n_items):
                    low  = c[i] + R/10 - R/1000
                    high = c[i] + R/10 + R/1000
                    p_bar[i] = np.random.uniform(low, high)
            # Strongly correlated
            elif correlation == 'SC':
                for i in range(n_items):
                    p_bar[i] = c[i] + R/10

            # sample degradation
            low =  (1 - delta)/2
            high = (1 + delta)/2
            p_hat = np.random.uniform(low, high, size=n_items)
            p_hat = np.multiply(p_bar, p_hat)

            # sample rejection penalty
            low =  1.1
            high = 1.5
            f = np.random.uniform(low, high, size=n_items)
            f = np.multiply(p_bar, f)

            # sample repair capacity (UN, uncorrelated)
            t = np.zeros(n_items)
            # Uncorrelated
            if correlation == 'UN': 
                for i in range(n_items):
                    t[i] = np.random.uniform(1, c[i])
            # Weakly correlated
            elif correlation == 'WC':
                for i in range(n_items):
                    low  = 0.5 * p_hat[i] - cfg.R/40
                    high = 0.5 * p_hat[i] + cfg.R/40
                    t[i] = np.random.uniform(low, high)
            # Almost strongly correlated
            elif correlation == 'ASC':
                for i in range(n_items):
                    low  = 0.5 * p_hat[i] + c[i]/10 - R/1000
                    high = 0.5 * p_hat[i] + c[i]/10 + R/1000
                    t[i] = np.random.uniform(low, high)
            # Strongly correlated
            elif correlation == 'SC':
                for i in range(n_items):
                    t[i] = 0.5 * p_hat[i] + c[i]/10

            inst['n_items'] = n_items
            inst['correlation'] = correlation
            inst['delta'] = delta
            inst['h'] = h
            inst['budget_factor'] = budget_factor

            inst['c'] = c
            inst['C'] = C
            inst['p_bar'] = p_bar
            inst['p_hat'] = p_hat
            inst['f'] = f
            inst['t'] = t

            return inst

            
    def read_paper_inst(self, inst, inst_dir):
        """ Reads instances data from paper. """
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

        fp_inst = inst_dir + inst_name
        with open(fp_inst, 'r') as f:
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