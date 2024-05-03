import os
import pickle as pkl
import time
from multiprocessing import Manager, Pool

import gurobipy as gp
import numpy as np

from ro.two_ro import factory_two_ro
from ro.utils import factory_get_path

from abc import ABC, abstractmethod


class DataManager(ABC):


    #---------------------------------------------------------------#
    #                           Constructor                         #
    #---------------------------------------------------------------#

    def __init__(self, cfg, problem):
        """ Constructor.  Initializes random seeds, config, and paths. """
        self.cfg = cfg
        self.problem_name = problem

        get_path = factory_get_path(self.problem_name)

        self.problem_path = get_path(self.cfg.data_path, self.cfg, "problem")
        self.ml_data_path = get_path(self.cfg.data_path, self.cfg, "ml_data")   
        self.inst_dir = get_path(self.cfg.data_path, self.cfg, "test_instances/")


    #---------------------------------------------------------------#
    #               Abstract methods to be implemented.             #
    #---------------------------------------------------------------#

    @abstractmethod
    def get_problem_data(self):
        """ Generates an problem based on the cfg in blo/params.py. """
        pass

    @abstractmethod
    def sample_instances(self,):
        """ Samples the set of instances or instances based on the cfg in blo/params.py. """
        pass

    @abstractmethod
    def sample_xi(self,):
        """ Samples uncertainty.  Problem specific and based on uncertainty set. """

    @abstractmethod
    def sample_xi(self,):
        """ Samples uncertainty.  Problem specific and based on uncertainty set. """

    @abstractmethod
    def solve_second_stage(self,):
        """ Solves second-stage.  Problem specific and should store appropraite features.  """


    #---------------------------------------------------------------#
    #               Base class methods that are shared.             #
    #---------------------------------------------------------------#

    def init_problem(self):
        """ Initializes problem and saves to .pkl file.  """
        print("Generating problem...")
        self.prob = self.get_problem_data()

        print(f"  Saved problem to: {self.problem_path}")
        pkl.dump(self.prob, open(self.problem_path, 'wb'))


    def load_problem(self):
        """ Loads instance file. """
        self.prob = pkl.load(open(self.problem_path, 'rb'))


    def generate_dataset(self, n_procs, debug):
        """ Generate dataset for training ml models. """
        # seed data collection
        np.random.seed(self.cfg.seed)

        # load problem
        self.load_problem()

        print(f"Generating dataset... ")
        print(f"  problem:   {self.problem_name} ")
        print(f"  n_procs:   {n_procs}")
        print(f"  debug:     {debug} ")

        data = []
        total_time = time.time()

        two_ro = factory_two_ro(self.problem_name)

        # collect random x, xi samples to compute second-stage cost for
        print(f"Collecting instances for dataset...")

        procs_to_run = self.sample_procs(two_ro)

        # solve all second-stage problems in debugging mode
        #   If implementing data collection for this first time, this will solve instances without multiprocessing, 
        #   which, in some cases is useful for debugging.  
        if debug:
            print("Running in debugging mode, i.e., without multiprocessing, only using 1 cpu...")
            mp_count = Manager().Value('i', 0)
            mp_time = time.time()
            for instance, inst_id, x, xi in procs_to_run:
                res = self.solve_second_stage(x, xi, instance, two_ro, inst_id, mp_time, mp_count, len(procs_to_run))
            print("Successfully ran all data collection.  Exiting!")
            exit()

        # solve all second-stage problems in debugging mode
        print("Solving second-stage problems for each instance...")
        data = []

        print(f"  Running {'{:,}'.format(len(procs_to_run))} with {n_procs} cpus... ")
        mp_count = Manager().Value('i', 0)
        mp_time = time.time()

        pool = Pool(n_procs)
        for instance, inst_id, x, xi in procs_to_run:
            res = pool.apply_async(self.solve_second_stage, args=(x, xi, instance, two_ro, inst_id, mp_time, mp_count, len(procs_to_run)))
            data.append(res)
        data = list(map(lambda x: x.get(), data))

        pool.close()
        pool.join()        

        print("  Done.")

        total_time = time.time() - total_time

        # get train/validation split, then store
        tr_data, val_data = self.get_train_validation_processes(data)

        ml_data = {
            "tr_data": tr_data,
            "val_data": val_data,
            "data": tr_data + val_data,
            "total_time": total_time,
            # "tr_time" : tr_time,
            # "val_time" : val_time,
        }

        print("Total Time:         ", total_time)
        print("Train dataset size: ", len(tr_data))
        print("Valid dataset size: ", len(val_data))

        pkl.dump(ml_data, open(self.ml_data_path, 'wb'), protocol=pkl.HIGHEST_PROTOCOL)

        total_time = time.time() - total_time


    def generate_dataset_by_inst(self, n_procs, debug):
        """ Generate dataset for training ml models. """
        # seed data collection
        np.random.seed(self.cfg.seed)

        # load problem
        self.load_problem()

        print(f"Generating dataset (by inst) ... ")
        print(f"  problem:   {self.problem_name} ")
        print(f"  n_procs:   {n_procs}")
        print(f"  debug:     {debug} ")

        data = []
        total_time = time.time()

        two_ro = factory_two_ro(self.problem_name)

        # collect random x, xi samples to compute second-stage cost for
        print(f"Collecting instances for dataset ...")

        # get instances
        instances = self.sample_instances(two_ro)

        # split instances to train/validation
        tr_instances, val_instances = self.get_train_validation_instances(instances)

        # get processes
        tr_procs_to_run = self.sample_processes_to_run(tr_instances)
        val_procs_to_run = self.sample_processes_to_run(val_instances)
    
        # solve all second-stage problems in debugging mode
        #   If implementing data collection for this first time, this will solve instances without multiprocessing, 
        #   which, in some cases is useful for debugging.  
        if debug:
            print("Running in debugging mode, i.e., without multiprocessing, only using 1 cpu...")
            mp_count = Manager().Value('i', 0)
            mp_time = time.time()
            for instance, inst_id, x, xi in tr_procs_to_run:
                res = self.solve_second_stage(x, xi, instance, two_ro, inst_id, mp_time, mp_count, len(tr_procs_to_run))
            print("Successfully ran all data collection.  Exiting!")
            exit()

        # solve all second-stage problems in debugging mode
        print("Solving second-stage problems for each training instance...")
        tr_time = time.time()
        tr_data = []

        print(f"  Running {'{:,}'.format(len(tr_procs_to_run))} with {n_procs} cpus... ")
        mp_count = Manager().Value('i', 0)
        mp_time = time.time()

        pool = Pool(n_procs)
        for instance, inst_id, x, xi in tr_procs_to_run:
            res = pool.apply_async(self.solve_second_stage, args=(x, xi, instance, two_ro, inst_id, mp_time, mp_count, len(tr_procs_to_run)))
            tr_data.append(res)
        tr_data = list(map(lambda x: x.get(), tr_data))

        pool.close()
        pool.join()        

        tr_time = time.time() - tr_time

        print("  Done.")

        print("Solving second-stage problems for each validation instance...")

        # Solve all suboptimal LPs for validation instances
        val_time = time.time()
        val_data = []

        print(f"  Running {'{:,}'.format(len(val_procs_to_run))} with {n_procs} cpus... ")
        mp_count = Manager().Value('i', 0)
        mp_time = time.time()

        pool = Pool(n_procs)
        
        for instance, inst_id, x, xi in val_procs_to_run:
            res = pool.apply_async(self.solve_second_stage, args=(x, xi, instance, two_ro, inst_id, mp_time, mp_count, len(val_procs_to_run)))
            val_data.append(res)

        val_data = list(map(lambda x: x.get(), val_data))

        pool.close()
        pool.join()        

        val_time = time.time() - val_time

        print("  Done.")

        total_time = time.time() - total_time

        # get train/validation split, then store
        ml_data = {
            "tr_data": tr_data,
            "val_data": val_data,
            "data": tr_data + val_data,
            "total_time": total_time,
            "tr_time" : tr_time,
            "val_time" : val_time,
        }

        print("Total Time:         ", total_time)
        print("Train dataset size: ", len(tr_data))
        print("Valid dataset size: ", len(val_data))

        pkl.dump(ml_data, open(self.ml_data_path, 'wb'), protocol=pkl.HIGHEST_PROTOCOL)

        total_time = time.time() - total_time


    def sample_processes_to_run(self, instances):
        """ Gets all processes to run by sampling x and xi for each instances.  """
        procs_to_run = []

        for inst_id, instance in enumerate(instances):

            # for each first-stage decision
            for i in range(self.cfg.n_samples_fs):

                # sample decisions
                x = self.sample_x(instance)

                # for each uncertainty for that decision
                for j in range(self.cfg.n_samples_per_fs):
                    xi = self.sample_xi(instance, x)

                    procs_to_run.append((instance, inst_id, x, xi))

        return procs_to_run


    def get_train_validation_instances(self, instances):
        """ Gets train/validation splits for the data.  
            Returns either train/validation instances if generalizing over multiple instances
            Otherwise returns the same instance for both. 
        """
        if len(instances) == 1:
            return instances, instances

        # get permutation of all indicies
        perm = np.random.permutation(len(instances))

        # get indicies for train/validation instances
        split_idx = int(self.cfg.tr_split * (len(instances)))
        tr_idx = perm[:split_idx].tolist()
        val_idx = perm[split_idx:].tolist()

        # get train/validation instances
        tr_instances = [instances[i] for i in tr_idx]
        val_instances = [instances[i] for i in val_idx]

        return tr_instances, val_instances


    def get_train_validation_processes(self, data):
        """ Gets train/validation splits for the data.  """
        np.random.seed(self.cfg.seed)
        perm = np.random.permutation(len(data))

        split_idx = int(self.cfg.tr_split * (len(data)))
        tr_idx = perm[:split_idx].tolist()
        val_idx = perm[split_idx:].tolist()

        tr_data = [data[i] for i in tr_idx]
        val_data = [data[i] for i in val_idx]

        return tr_data, val_data


    def update_mp_status(self, mp_count, mp_time, n_samples):
        """ Printing/status for multiprocessing. """
        mp_count.value += 1
        count = mp_count.value

        if count % 1000 == 0:
            if count == 1000:
                print("    Sample              | Percent   | Time      | ETA          ")
                print("    -----------------------------------------------------------")

            n_samples_str = '{:,}'.format(n_samples)

            pct = count / n_samples
            pct_str = '{:.2%}'.format(pct)
            
            count_str = '{:,}'.format(count)

            time_ = time.time() - mp_time
            time_str = '{:.2f}'.format(time_)

            eta_ = (time_/count) *  (n_samples - count)
            eta_str = '{:.2f}'.format(eta_)

            print(f"    {count_str} / {n_samples_str}   |   " \
                   f"{pct_str}   |   " \
                   f"{time_str}s   |   " \
                   f"{eta_str}s"
                   )