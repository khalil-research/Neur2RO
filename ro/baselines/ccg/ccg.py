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
        self.start_time = time.time()
        self.time_limit = time_limit

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
        sols = self.master_problem_build(x_init=x_init)
        obj, x, y, model = sols
        mp_time += time.time() - start_mp

        # initialize sub problem
        start_sp = time.time()
        m_sub = self.sub_problem_build(x)
        sp_time += time.time() - start_sp

        # CCG algorithm
        print_header = True
        ub = self.upper_bound
        while self.time_left() > 0:
            # SUB PROBLEM
            start_sp = time.time()
            sols = self.sub_problem(m_sub, x)
            if sols is None:
                break
            obj_adv, xi = sols
            ub = min([ub, obj_adv])
            sp_time += time.time() - start_sp

            inc_obj_t[self.curr_time()] = obj_i
            inc_obj_n[tot_nodes] = obj_i
            inc_zeta_t[self.curr_time()] = zeta
            inc_zeta_n[tot_nodes] = zeta
            self.print_results(it, obj, ub, print_header=print_header)
            print_header = False

            # check if robust
            if ub - obj < 1e-04:
                obj_i, x_i, y_i = (copy.deepcopy(obj), copy.deepcopy(x), copy.deepcopy(y))
                break
            # update master problem
            start_mp = time.time()
            sols = self.master_problem_update(xi, model)
            if sols is None:
                break
            obj, x, y, model = sols
            mp_time += time.time() - start_mp

            it += 1
        self.print_results(it, obj, ub, print_header=print_header)
        # termination results
        inc_obj_t[self.curr_time()] = obj_i
        inc_obj_n[tot_nodes] = obj_i
        inc_zeta_t[self.curr_time()] = zeta
        inc_zeta_n[tot_nodes] = zeta

        results = {"obj": obj_i, "x": x_i, "y": y_i,
                   "inc_obj_t": inc_obj_t,  "inc_obj_n": inc_obj_n,
                   "inc_viol_t": inc_zeta_t, "inc_viol_n": inc_zeta_n,
                   "runtime": self.curr_time(), "tot_nodes": tot_nodes, "mp_time": mp_time, "sp_time": sp_time}

        pkl.dump(results, open(self.results_path, 'wb'), protocol=pkl.HIGHEST_PROTOCOL)
        return obj_i

    def time_left(self):
        return max([0, self.time_limit - (time.time() - self.start_time)])

    def curr_time(self):
        return time.time() - self.start_time

    def get_gap(self, lb, ub):
        # from SCIP https://scipopt.org/doc-7.0.1/html/PARAMETERS.php:  | primal - dual | / MIN( | dual |, | primal |)
        return abs(ub - lb) / min([abs(lb), abs(ub)])

    def print_results(self, it, lb, ub, print_header=True):
        if print_header:
            print(*["{!s:^8}|".format(item) for item in ["I", "It.", "LB", "UB", "Time"]])
            print("  ------------------------------------------------------------------- ")

        if ub is not None:
            ub_str = '{:.4}'.format(ub)
        else:
            ub_str = ""
        lb_str = '{:.4}'.format(lb)

        time_str = '{:.2f}s'.format(self.curr_time())
        print(*["{!s:^8}|".format(item) for item in [self.inst_seed, it, lb_str, ub_str, time_str]])
