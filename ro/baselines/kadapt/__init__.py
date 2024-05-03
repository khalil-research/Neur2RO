import ro.params as params


def factory_bm(problem):
    cfg = getattr(params, problem)

    if "cb" in problem:
        from .cb import CapitalBudgetingKAdapt
        return CapitalBudgetingKAdapt(cfg)
    else:
        raise ValueError("Invalid problem type!")
