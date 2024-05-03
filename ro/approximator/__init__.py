import ro.params as params


def factory_approximator(args, cfg, net, inst_params):
        from .cb import CapitalBudgetingApproximator
        return CapitalBudgetingApproximator(args, cfg, net, inst_params)

    elif "kp" in args.problem:
        from .kp import KnapsackApproximator
        return KnapsackApproximator(args, cfg, net, inst_params)

    else:
        raise ValueError("Invalid problem type!")
