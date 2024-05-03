from abc import ABC, abstractmethod
import time
import copy
import pickle as pkl
import numpy as np


class CCGAlgorithm(ABC):
    @abstractmethod
    def master_problem_build(self):
        pass

    @abstractmethod
    def master_problem_update(self):
        pass

    @abstractmethod
    def sub_problem(self):
        pass

    @abstractmethod
    def sub_problem_build(self):
        pass

    def algorithm(self, time_limit=30*60, x_init=None):
        # Initialize
        it = 0
        start_time = time.time()
        # initialization for saving
        inc_obj_t = dict()
        inc_obj_n = dict()
        inc_zeta_t = dict()
        inc_zeta_n = dict()

        tot_nodes = 0
        mp_time = 0
        sp_time = 0
        # initialization of lower and upper bounds
        zeta = np.inf
        inc_zeta_t[0] = zeta
        inc_zeta_n[0] = zeta

        # initialize master problem
        obj_i, x_i, y_i = (self.upper_bound, [], [])
        start_mp = time.time()
        obj, x, y, model = self.master_problem_build(x_init=x_init)
        mp_time += time.time() - start_mp

        # initialize sub problem
        start_sp = time.time()
        m_sub = self.sub_problem_build(x)
        sp_time += time.time() - start_sp

        # CCG algorithm
        print_header = True
        ub = np.inf
        while time.time() - start_time < time_limit:
            # SUB PROBLEM
            start_sp = time.time()
            obj_adv, xi = self.sub_problem(m_sub, x)
            sp_time += time.time() - start_sp

            inc_obj_t[time.time() - start_time] = obj_i
            inc_obj_n[tot_nodes] = obj_i
            inc_zeta_t[time.time() - start_time] = zeta
            inc_zeta_n[tot_nodes] = zeta
            self.print_results(it, obj, ub, start_time, print_header=print_header)
            print_header = False

            # check if robust
            if obj_adv < obj + 1e-04:
                obj_i, x_i, y_i = (copy.deepcopy(obj), copy.deepcopy(x), copy.deepcopy(y))
                self.print_results(it, obj_i, ub, start_time, print_header=print_header)
                break
            ub = min([ub, obj_adv])
            # update master problem
            start_mp = time.time()
            obj, x, y, model = self.master_problem_update(xi, model)
            mp_time += time.time() - start_mp

            it += 1

        # termination results
        inc_obj_t[time.time() - start_time] = obj_i
        inc_obj_n[tot_nodes] = obj_i
        inc_zeta_t[time.time() - start_time] = zeta
        inc_zeta_n[tot_nodes] = zeta
        self.print_results(it, obj_i, ub, start_time, print_header=print_header)

        results = {"obj": obj_i, "x": x_i, "y": y_i,
                   "inc_obj_t": inc_obj_t,  "inc_obj_n": inc_obj_n,
                   "inc_viol_t": inc_zeta_t, "inc_viol_n": inc_zeta_n,
                   "runtime": time.time() - start_time, "tot_nodes": tot_nodes, "mp_time": mp_time, "sp_time": sp_time}

        pkl.dump(results, open(self.results_path, 'wb'), protocol=pkl.HIGHEST_PROTOCOL)
        return obj_i

    def get_gap(self, lb, ub):
        # from SCIP https://scipopt.org/doc-7.0.1/html/PARAMETERS.php:  | primal - dual | / MIN( | dual |, | primal |)
        return abs(ub - lb) / min([abs(lb), abs(ub)])

    def print_results(self, it, lb, ub, start_time, print_header=True):
        if print_header:
            print(*["{!s:^8}|".format(item) for item in ["I", "It.", "LB", "UB", "Time"]])
            print("  ------------------------------------------------------------------- ")

        inst = self.inst['seed']
        if ub is not None:
            ub_str = '{:.4}'.format(ub)
        else:
            ub_str = ""
        lb_str = '{:.4}'.format(lb)

        time_ = time.time() - start_time
        time_str = '{:.2f}s'.format(time_)
        print(*["{!s:^8}|".format(item) for item in [inst, it, lb_str, ub_str, time_str]])
