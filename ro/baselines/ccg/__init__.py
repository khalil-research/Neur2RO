import ro.params as params


def factory_bm(problem):
    cfg = getattr(params, problem)
    if "cb" in problem:
        from .cb import CapitalBudgetingCCG
        return CapitalBudgetingCCG(cfg)
    else:
        raise ValueError("Invalid problem type!")
