import ro.params as params


def factory_eval(problem):
    cfg = getattr(params, problem)

    if "cb" in problem:
        from ro.baselines.kadapt.cb import CapitalBudgetingKAdapt
        return CapitalBudgetingKAdapt(cfg)
    else:
        raise ValueError("Invalid problem type!")
