from abc import ABC, abstractmethod
import time
import copy
import numpy as np
import pickle as pkl


class KAdaptability(ABC):
    @abstractmethod
    def master_problem_build(self):
        pass

    @abstractmethod
    def master_problem_update(self):
        pass

    @abstractmethod
    def sub_problem(self):
        pass

    def algorithm(self, K, time_limit=30*60, x_eval=None):
        # Initialize
        it = 0
        self.start_time = time.time()
        self.time_limit = time_limit

        # initialization for saving
        inc_obj_t = dict()
        inc_obj_n = dict()
        inc_lb_t = dict()
        inc_lb_n = dict()
        inc_gap_t = dict()
        inc_gap_n = dict()
        inc_x_t = dict()
        inc_x_n = dict()

        prune_count = 0
        tot_nodes = 0
        mp_time = 0
        sp_time = 0
        # initialization of lower and upper bounds
        obj_i, x_i, y_i = (self.upper_bound, [], [])
        lb = self.lower_bound
        inc_lb_t[0] = lb
        inc_lb_n[0] = lb
        gap = np.inf
        inc_gap_t[0] = gap
        inc_gap_n[0] = gap

        inc_x_t[0] = x_i
        inc_x_n[0] = x_i

        # initialize N_set
        placement_i, scen_all = self.init_k_adapt(K)
        N_set = [placement_i]
        lb_active = [lb]
        # K-branch and bound algorithm
        xi_new, k_new = None, None
        new_xi_num = 0
        new_model = True
        model = None

        print_header = True
        while N_set and gap > 1e-5 and self.time_left() > 0:
            # MASTER PROBLEM
            if new_model:
                del model
                tot_nodes += 1
                # take new node
                new_pass = self.rng.randint(len(N_set))
                placement = N_set.pop(new_pass)
                lb_active.pop(new_pass)
                tau = {k: scen_all[placement[k]] for k in range(K)}
                # master problem
                start_mp = time.time()
                sols = self.master_problem_build(K, tau, x_eval)
                mp_time += time.time() - start_mp
                if sols is None:
                    break
                obj, x, y, model = sols
            else:
                # make new tau from k_new
                tot_nodes += 1
                # master problem
                start_mp = time.time()
                sols = self.master_problem_update(K, k_new, xi, model)
                mp_time += time.time() - start_mp
                if sols is None:
                    break
                obj, x, y, model = sols

                placement[k_new].append(new_xi_num)
                tau = {k: scen_all[placement[k]] for k in range(K)}

            # UPDATE LB
            if lb_active:
                min_lb = min(lb_active)
                if lb < min_lb:
                    lb = min_lb
                    inc_lb_t[time.time() - self.start_time] = lb
                    inc_lb_n[tot_nodes] = lb
                    try:
                        gap = self.get_gap(lb, obj_i)
                    except ZeroDivisionError:
                        gap = np.inf
                    inc_gap_t[time.time() - self.start_time] = gap
                    inc_gap_n[tot_nodes] = gap
                    self.print_results(it, K, None, lb, gap, print_header=print_header)
                    print_header = False

            # prune if obj higher than current robust obj
            if obj - obj_i > -1e-8:
                prune_count += 1
                new_model = True
                continue

            # SUB PROBLEM
            start_sp = time.time()
            sols = self.sub_problem(K, x, y, obj, placement)
            sp_time += time.time() - start_sp
            if sols is None:
                break
            zeta, xi = sols

            # check if robust
            if zeta < 1e-04:
                obj_i, x_i, y_i = (copy.deepcopy(obj), copy.deepcopy(x), copy.deepcopy(y))
                tau_i = copy.deepcopy(tau)
                inc_obj_t[self.curr_time()] = obj_i
                inc_obj_n[tot_nodes] = obj_i
                gap = self.get_gap(lb, obj_i)
                inc_gap_t[self.curr_time()] = gap
                inc_gap_n[tot_nodes] = gap
                inc_x_t[self.curr_time()] = x_i
                inc_x_n[tot_nodes] = x_i

                self.print_results(it, K, obj_i, lb, gap, print_header=print_header)

                print_header = False
                prune_count += 1
                new_model = True
                continue
            else:
                new_model = False
                new_xi_num += 1
                scen_all = np.vstack([scen_all, xi])

            full_list = [k for k in range(K) if len(tau[k]) > 0]
            if len(full_list) == 0:
                K_set = [0]
                k_new = 0
            elif len(full_list) == K:
                K_set = np.arange(K)
                k_new = self.rng.randint(K)
            else:
                K_prime = min(K, full_list[-1] + 2)
                K_set = np.arange(K_prime)
                k_new = K_set[-1]

            for k in K_set:
                if k == k_new:
                    continue
                # add to node set
                placement_tmp = copy.deepcopy(placement)
                placement_tmp[k].append(new_xi_num)
                N_set.append(placement_tmp)
                lb_active.append(obj)
            it += 1

            # UPDATE LB
            if lb_active:
                min_lb = min(lb_active)
                if lb < min_lb:
                    lb = min_lb
                    inc_lb_t[self.curr_time()] = lb
                    inc_lb_n[tot_nodes] = lb
                    try:
                        gap = self.get_gap(lb, obj_i)
                    except ZeroDivisionError:
                        gap = np.inf
                    inc_gap_t[self.curr_time()] = gap
                    inc_gap_n[tot_nodes] = gap
                    self.print_results(it, K, None, lb, gap, print_header=print_header)
                    print_header = False

        # termination results
        inc_obj_t[self.curr_time()] = obj_i
        inc_obj_n[tot_nodes] = obj_i
        inc_lb_t[self.curr_time()] = lb
        inc_lb_n[tot_nodes] = lb
        if not N_set:
            gap = 0
        inc_gap_t[self.curr_time()] = gap
        inc_gap_n[tot_nodes] = gap
        inc_x_t[self.curr_time()] = x_i
        inc_x_n[tot_nodes] = x_i

        self.print_results(it, K, obj_i, lb, gap, print_header=print_header)

        results = {"obj": obj_i, "lb": lb, "gap": gap,
                   "x": x_i, "y": y_i, "scenarios": tau_i,
                   "inc_obj_t": inc_obj_t,  "inc_obj_n": inc_obj_n,
                   "inc_lb_t": inc_lb_t, "inc_lb_n": inc_lb_n,
                   "inc_gap_t": inc_gap_t, "inc_gap_n": inc_gap_n,
                   "inc_x_t": inc_x_t, "inc_x_n": inc_x_n,
                   "runtime": self.curr_time(), "tot_nodes": tot_nodes, "mp_time": mp_time, "sp_time": sp_time}
        pkl.dump(results, open(self.results_path, 'wb'), protocol=pkl.HIGHEST_PROTOCOL)
        return results

    def time_left(self):
        return max([0, self.time_limit - (time.time() - self.start_time)])

    def curr_time(self):
        return time.time() - self.start_time

    # Initialize uncertainty set with init_scen
    def init_k_adapt(self, K):
        tau = {k: [] for k in range(K)}
        tau[0].append(self.init_scen)

        placement = {k: [] for k in range(K)}
        placement[0].append(0)

        scen_all = self.init_scen.reshape([1, -1])
        return tau, placement, scen_all

    def get_gap(self, lb, ub):
        # from SCIP https://scipopt.org/doc-7.0.1/html/PARAMETERS.php:  | primal - dual | / MIN( | dual |, | primal |)
        return abs(ub - lb) / min([abs(lb), abs(ub)])

    def print_results(self, it, K, ub, lb, gap, print_header=True):
        if print_header:
            print(*["{!s:^10}|".format(item) for item in ["I", "K", "It.", "UB", "LB", "GAP", "Time"]])
            print("  ------------------------------------------------------------------- ")

        if ub is not None:
            ub_str = '{:.4}'.format(float(ub))
        else:
            ub_str = ""
        lb_str = '{:.4}'.format(float(lb))
        gap_str = '{:.2%}'.format(float(gap))

        time_str = '{:.2f}s'.format(self.curr_time())
        print(*["{!s:^8}|".format(item) for item in [self.inst_seed, K, it, ub_str, lb_str, gap_str, time_str]])

