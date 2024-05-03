import pickle

import numpy as np
import torch
import torch.nn as nn

import ro.params as params

from .consts import DataManagerModes
from .consts import LearningModelTypes
from .consts import LossPenaltyTypes

from .kp import get_path as get_path_kp


def factory_get_path(problem):
    """ Gets the get_path function for each problem type. """
    if 'kp' in problem:
        from .kp import get_path
        return get_path

    elif 'cb' in problem:
        from .cb import get_path

    # add new problems here

    else:
        raise Exception(f"ro.utils not defined for problem class {problem}")

    return get_path


def factory_load_problem(cfg, problem):
    """ Loads problem file from the cfg. """
    get_path = factory_get_path(problem)
    prob_fp = get_path(cfg.data_path, cfg, "problem")
    prob = pickle.load(open(inst_fp, 'rb'))
    return prob


