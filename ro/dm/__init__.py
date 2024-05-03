import ro.params as params


def factory_dm(problem):
    cfg = getattr(params, problem)

    if "kp" in problem:
        print("Loading Knapsack data manager...")
        from .kp import KnapsackDataManager
        return KnapsackDataManager(cfg, problem)

    elif "cb" in problem:
        print("Loading Capital Budgeting data manager...")
        from .cb import CapitalBudgetingDataManager
        return CapitalBudgetingDataManager(cfg, problem)

    else:
        raise ValueError("Invalid problem type!")
