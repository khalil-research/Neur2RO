import os
import pickle as pkl
import time
from multiprocessing import Manager, Pool

import gurobipy as gp
import numpy as np

from ro.two_ro.kp import Knapsack
from ro.utils.kp import get_path
from .dm import DataManager




class KnapsackDataManager(DataManager):

    def __init__(self, cfg, problem):
        """ Constructor for Knapsack Data Manager.  """
        super(KnapsackDataManager, self).__init__(cfg, problem)


    def get_problem_data(self):
        """ Generates an problem based on the cfg in blo/params.py. """
        prob = {}
        
        # problem type
        prob['data_type'] = self.cfg.data_type

        # problem data
        prob['n_items'] = self.cfg.n_items
        prob['correlation'] = self.cfg.correlation
        prob['h'] = self.cfg.h
        prob['delta'] = self.cfg.delta
        prob['budget_factor'] = self.cfg.budget_factor
        #inst['max_budget'] = inst['n_items'] * inst['budget_factor']
        prob['H'] = self.cfg.H
        prob['R'] = self.cfg.R

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

        # store cfg
        prob['cfg'] = self.cfg

        return prob


    def sample_procs(self, two_ro):
        """ Samples the set of instances + decisions, i.e., everything needed to run 
            based on single instance based on the cfg in blo/params.py. 
            For KP, we can either do single-instance or general sampling. 
        """
        procs_to_run = []

        # read instances from paper data if evaluating/training on single instance
        if self.cfg.data_type == "instance":
            # check that number of samples is equal to 1 for this data type
            assert(self.cfg.n_samples_inst == 1)

            # get instance
            inst_dir =  self.cfg.data_path + "kp/instances/"
            instance = two_ro.read_paper_inst(self.prob, inst_dir)

            for j in range(self.cfg.n_samples_fs):
                x = self.sample_x(instance)

                for j in range(self.cfg.n_samples_fs):
                    xi = self.sample_xi(instance, x)

                    procs_to_run.append((instance, 1, x, xi))

        elif self.cfg.data_type == "general":
            # otherwise sample instances based on paper procedure
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
        # read instances from paper data if evaluating/training on single instance
        if self.cfg.data_type == "instance":
            # check that number of samples is equal to 1 for this data type
            assert(self.cfg.n_samples_inst == 1)

            # get instance
            inst_dir =  self.cfg.data_path + "kp/instances/"
            instance = two_ro.read_paper_inst(self.prob, inst_dir)
            return [instance]

        elif self.cfg.data_type == "general":
            # otherwise sample instances based on paper procedure
            instances = []
            for k in range(self.cfg.n_samples_inst):
                inst = two_ro.sample_new_inst(self.cfg, seed = self.cfg.seed + k)
                instances.append(inst)
            return instances

        else:
            raise Exception(f"data_type = {self.cfg.data_type} is not valid for KP.")


    def sample_x(self, instance):
        """ Samples first-stage decision. 
            For KP this is a binary vector number of 0's sampled based on a random uniform dist.  
        """
        prob = np.random.uniform(0, 1)
        x = np.random.choice([0,1], p=[prob, 1-prob], size=instance['n_items'])
        return x


    def sample_xi(self, instance, x, zero_unused_prob=0.0):
        """ Samples random uncertainty. 
            For KP this is a continous vector that sums to a number between [1, budget]
        """
        budget_max = instance['n_items'] * instance['budget_factor']
        budget = np.random.uniform(1/budget_max, budget_max)
        xi = np.random.uniform(0, 1, size=instance['n_items'])
        
        # zero out indicies where x is not contained in
        # default set to zero, but may be useful for later
        # zero_unused_prob = 0.0
        if zero_unused_prob > 0:
            p = np.random.uniform(0,1)
            if p < zero_unused_prob and x.sum() > 0:
                xi = np.multiply(x, xi) 
        xi = budget * xi / xi.sum()
            
        return xi


    def solve_second_stage(self, x, xi, instance, two_ro, inst_id, mp_time, mp_count, n_samples):
        """ Obtains the cost of the suboptimal first stage solution.  """
        time_ = time.time()

        fs_obj, ss_obj, _ = two_ro.solve_second_stage(
            x, 
            xi,
            instance,
            gap = self.cfg.mip_gap,
            time_limit = self.cfg.time_limit,
            verbose = self.cfg.verbose,
            threads = self.cfg.threads)

        res = {
            'x' : x,
            'xi' : xi,
            'instance' : instance,
            'inst_id' : inst_id,
            'concat_x_xi' : list(x) + list(xi),
            'fs_obj' : fs_obj,
            'ss_obj' : ss_obj,
            'time' : time.time() - time_
        }

        self.update_mp_status(mp_count, mp_time, n_samples)

        return res

