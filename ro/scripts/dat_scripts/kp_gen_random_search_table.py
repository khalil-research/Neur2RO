import argparse
import hashlib
import itertools

import numpy as np

problem_types = {
    "kp": ["kp"],
}

class ContinuousValueSampler(object):
    """ A class to sample uniformly at random in the range of [lb,ub].
        Additionally includes a probability of sampling zero if needed.  """

    def __init__(self, lb, ub, prob_zero=0.0):
        self.lb = lb
        self.ub = ub
        self.prob_zero = prob_zero

    def sample(self):
        if np.random.rand() < self.prob_zero:
            return 0
        return np.round(np.random.uniform(self.lb, self.ub), 5)


class DiscreteSampler(object):
    """ A class to sample uniformly at random in the range of [lb,ub]. """
    def __init__(self, choices):
        self.choices = choices
        self.n_choices = len(choices)

    def sample(self):
        choice = np.random.choice(list(range(self.n_choices)))
        return self.choices[choice]
        #return np.random.choice(self.choices)


def get_kp_tiny_set_config(problem, model_type):
    """ Defines params space for NN-E. """
    LR_LB, LR_UB = 1e-5, 1e-1
    L1_LB, L1_UB = 1e-5, 1e-1
    L2_LB, L2_UB = 1e-5, 1e-1
    L1_ZERO, L2_ZERO = 0.25, 0.25

    # combintaions of embedding dimensions for tiny_set_nets
    x_embed_dims = itertools.product([8, 16, 32, 64], [4,8,16,32,64])
    x_post_agg_dims = itertools.product([8, 16, 32, 64], [4,8,16,32,64])
    xi_embed_dims = itertools.product([8, 16, 32, 64], [4,8,16,32,64])
    xi_post_agg_dims = itertools.product([8, 16, 32, 64], [4,8,16,32,64])
    tiny_dims = [[2], [4], [8], [16], [32]]

    # itertools products to list of lists
    x_embed_dims = [list(item) for item in x_embed_dims]
    x_post_agg_dims = [list(item) for item in x_embed_dims]
    xi_embed_dims = [list(item) for item in x_embed_dims]
    xi_post_agg_dims = [list(item) for item in x_embed_dims]

    config = {
        # general parameters
        "batch_size": DiscreteSampler([32, 64, 128, 256]),
        "lr": ContinuousValueSampler(LR_LB, LR_UB),
        "wt_lasso": ContinuousValueSampler(L1_LB, L1_UB, L1_ZERO),
        "wt_ridge": ContinuousValueSampler(L2_LB, L2_UB, L2_ZERO),
        "n_epochs": DiscreteSampler([2000]),
        "loss_fn": DiscreteSampler(["MSELoss"]),
        "dropout": ContinuousValueSampler(0.0, 0.5),
        "optimizer": DiscreteSampler(['Adam', 'Adagrad', 'RMSprop']),

        # parameters specific to tiny_set_net
        "agg_type": DiscreteSampler(["mean", "sum"]),
        "x_embed_dims": DiscreteSampler(x_embed_dims),
        "x_post_agg_dims": DiscreteSampler(x_post_agg_dims),
        "xi_embed_dims": DiscreteSampler(xi_embed_dims),
        "xi_post_agg_dims": DiscreteSampler(xi_post_agg_dims),
        "tiny_dims": DiscreteSampler(tiny_dims),

    }

    return config


def get_config(problem, model_type):
    """ Gets the config for the given model_type and problem. """
    if "kp" in problem:
        if model_type == "nn_tiny_set":
            return get_kp_tiny_set_config(problem, model_type)
        else:
            raise Exception(f"Config not defined for model_type [{model_type}]")
    else:
        raise Exception(f"Config not defined for problem [{problem}]")


def sample_config(problem, model_type, config):
    """ Samples a confiuration for NN-E. """
    config_cmd = f"python -m ro.scripts.kp_train_tiny_set --problem {problem} --model_type {model_type}"
    for param_name, param_sampler in config.items():
        param_val = param_sampler.sample()
        if type(param_val) == list:
            param_val = list(map(lambda x: str(x), param_val))
            param_str = " ".join(param_val)
            config_cmd += f" --{param_name} {param_str}"
        else:
            config_cmd += f" --{param_name} {param_val}"

    return config_cmd


def main(args):
    cmds = []

    for problem in args.problems:
        for ptypes in problem_types[problem]:
            for model_type in args.model_type:
                config = get_config(problem, model_type)
                for i in range(args.n_configs):
                    p_hash = int(hashlib.md5(b'{ptypes}').hexdigest(), 16)
                    np.random.seed((args.seed + i + p_hash) % (2 ** 32 - 1))
                    cmds.append(sample_config(ptypes, model_type, config))

    cmds += "\n"
    # config_cmd += "\n"

    # write to text file
    textfile = open(args.file_name, "w")
    for i, cmd in enumerate(cmds[:-1]):
        textfile.write(f"{i + args.start_idx} {cmd}\n")
    textfile.write(f"{i + 2} {cmds[-1]}")
    textfile.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generates a list of configs to run for random search.')
    parser.add_argument('--problems', type=str, nargs='+', default=["kp"])
    parser.add_argument('--model_type', type=str, default=["nn_tiny_set"])
    parser.add_argument('--n_configs', type=int, default=100)
    parser.add_argument('--file_name', type=str, default='table.dat')
    parser.add_argument('--start_idx', type=int, default=1)
    parser.add_argument('--use_problem_for_rng', type=int, default=1)
    parser.add_argument('--seed', type=int, default=1234)

    args = parser.parse_args()

    main(args)
