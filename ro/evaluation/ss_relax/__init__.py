import ro.params as params


def factory_eval(problem):
    cfg = getattr(params, problem)
    if "cb" in problem:
        from .cb import CapitalBudgetingRelax
        return CapitalBudgetingRelax(cfg)
    else:
        raise ValueError("Invalid problem type!")
