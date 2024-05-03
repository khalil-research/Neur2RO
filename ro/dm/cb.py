import os
import pickle as pkl
import time
from multiprocessing import Manager, Pool

import gurobipy as gp
import numpy as np

from ro.two_ro.cb import CapitalBudgeting
from ro.utils.cb import get_path
from .dm import DataManager


class CapitalBudgetingDataManager(DataManager):

    def __init__(self, cfg, problem):
        """ Constructor for Capital Budgeting Data Manager.  """
        super(CapitalBudgetingDataManager, self).__init__(cfg, problem)

    # TODO: Figure out where to move these
    # def generate_test_instances(self, args):
    #     """
    #     Generate a capital budgeting instance from ....
    #     """
    #     print("Generating instance...")
    #     self.seed_used = self.cfg.seed
    #     self.rng.seed(self.cfg.seed)

    #     for s in range(1, args.n_inst_per_size+1):
    #         inst = dict()
    #         inst = self._get_test_data(self.cfg, inst, args, s)
    #         inst = self._get_sample_data(self.cfg, inst)
    #         self._write_test_inst(inst)


    # def _write_test_inst(self, inst):
    #     # different:
    #     os.makedirs(self.inst_dir, exist_ok=True)
    #     test_dir = self.inst_dir + f"CB_n{inst['n_items']}_xi{inst['xi_dim']}_s{inst['seed']}"
    #     f = open(test_dir, "w+")
    #     f.write(str(inst['seed']) + " ")
    #     f.write(str(inst['n_items']) + " ")
    #     f.write(str(inst['xi_dim']) + " ")
    #     f.write(str(inst['budget']) + " ")
    #     f.write(str(inst['max_loan']) + "\n")
    #     f.write(" ".join(inst['c_bar'].astype(str)) + "\n")
    #     f.write(" ".join(inst['r_bar'].astype(str)) + "\n")
    #     for phi_vector in inst['phi'].values():
    #         f.write(" ".join(phi_vector.astype(str)) + "\n")
    #     for psi_vector in inst['psi'].values():
    #         f.write(" ".join(psi_vector.astype(str)) + "\n")
    #     f.close()


    def get_problem_data(self):
        """ Stores generic problem information. """
        prob = {}

        # fixed problem data
        prob['n_items'] = self.cfg.n_items
        prob['k'] = self.cfg.k
        prob['loans'] = self.cfg.loans
        prob['l'] = self.cfg.l
        prob['m'] = self.cfg.m
        prob['xi_dim'] = self.cfg.xi_dim

        # data generation data.
        prob['time_limit'] = self.cfg.time_limit
        prob['mip_gap'] = self.cfg.mip_gap
        prob['verbose'] = self.cfg.verbose
        prob['threads'] = self.cfg.threads
        prob['tr_split'] = self.cfg.tr_split
        prob['n_samples_inst'] = self.cfg.n_samples_inst
        prob['n_samples_fs'] = self.cfg.n_samples_fs
        prob['n_samples_per_fs'] = self.cfg.n_samples_per_fs

        # general problem data
        prob['seed'] = self.cfg.seed
        prob['data_path'] = self.cfg.data_path

        return prob


    def sample_procs(self, two_ro):
        """ Samples the set of instances + decisions, i.e., everything needed to run 
            based on single instance based on the cfg in blo/params.py. 
            For KP, we can either do single-instance or general sampling. 
        """
        procs_to_run = []
        for i in range(self.cfg.n_samples_inst):
            instance = two_ro.sample_new_inst(self.cfg, seed = self.cfg.seed + i)

            for j in range(self.cfg.n_samples_fs):
                x = self.sample_x(instance)

                for j in range(self.cfg.n_samples_per_fs):
                    xi = self.sample_xi(instance, x)

                    procs_to_run.append((instance, i, x, xi))

        return procs_to_run


    def sample_instances(self, two_ro):
        """ Samples the set of instances or single instance based on the cfg in blo/params.py. 
            For KP, we can either do single-instance or general sampling. 
        """
        # otherwise sample instances based on paper procedure
        instances = []
        for k in range(self.cfg.n_samples_inst):
            inst = two_ro.sample_new_inst(self.cfg, seed = self.cfg.seed + k + 1000)
            instances.append(inst)

        return instances


    def solve_second_stage(self, x, xi, instance, two_ro, inst_id, mp_time, mp_count, n_samples):
        """ Obtains the cost of the suboptimal first stage solution and stores all relavant
            information that is used to construct features.  """
        time_ = time.time()

        # solve problem without slack
        fs_obj, ss_obj, unc_cons, ss_unc_cons, _ = two_ro.solve_second_stage(
            x,
            xi,
            instance,
            gap = self.cfg.mip_gap,
            time_limit = self.cfg.time_limit,
            verbose = self.cfg.verbose,
            threads = self.cfg.threads)

        # solve problem with slack
        slk_fs_obj, slk_ss_obj, slk_unc_cons, slk_ss_unc_cons, _ = two_ro.solve_second_stage_slack(
            x,
            xi,
            instance,
            gap = self.cfg.mip_gap,
            time_limit = self.cfg.time_limit,
            verbose = self.cfg.verbose,
            threads = self.cfg.threads)

        # get c(xi)^T x for label
        min_budget_cost = two_ro.get_constr_label(x, xi, instance)

        res = {
            # variable/problem information
            'x': x,
            'xi': xi,
            'instance': instance,

            # without slack labels
            'fs_obj': fs_obj,
            'ss_obj': ss_obj,
            'fs_plus_ss_obj' : fs_obj + ss_obj,
            'unc_cons': unc_cons,
            'ss_unc_cons': ss_unc_cons,

            # with slack labels
            'slk_fs_obj': slk_fs_obj,
            'slk_ss_obj': slk_ss_obj,
            'slk_fs_plus_ss_obj' : slk_fs_obj + slk_ss_obj,
            'slk_unc_cons': slk_unc_cons,
            'slk_ss_unc_cons': slk_ss_unc_cons,

            # constraint budget labels
            'min_budget_cost' : min_budget_cost,

            # time 
            'time': time.time() - time_
        }

        self.update_mp_status(mp_count, mp_time, n_samples)

        return res


    def sample_x(self, instance):
        """ Samples first-stage decision. 
            For KP this is a binary vector number of 0's sampled based on a random uniform dist.  
        """
        while True:
            prob = np.random.uniform(0, 1)
            x = np.random.choice([0, 1], p=[prob, 1 - prob], size=instance['n_items'])
            if instance['loans']:
                x0 = np.random.uniform(0, instance['max_loan'])

            # check
            if not instance['loans'] \
                    and np.multiply(instance['c_bar'], x).sum() > instance['budget']:
                continue
            elif instance['loans'] \
                    and (np.multiply(instance['c_bar'], x).sum() > instance['budget'] + x0
                    or np.multiply(instance['c_bar'], x).sum() > inst['budget'] + x0 + instance['max_loan']):
                instance
            if instance['loans']:
                x = np.append(x, x0)
            return x

    def sample_xi(self, instance, x):
        """ Samples random uncertianty decision. 
            For KP this is a continous vector that sums to a number between [1, budget]
        """
        xi = np.random.uniform(-1, 1, size=instance['xi_dim'])
        return xi
