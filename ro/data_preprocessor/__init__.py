import ro.params as params


def factory_dp(cfg, model_type, predict_feas, problem, device):
    cfg = getattr(params, problem)

    if "kp" in problem:
        print("Loading Knapsack data preprocessor...")
        from .kp import KnapsackDataPreprocessor
        return KnapsackDataPreprocessor(cfg, model_type, predict_feas, device)

    elif "cb" in problem:
        print("Loading Captial Budgeting data preprocessor...")
        from .cb import CapitalBudgetingDataPreprocessor
        return CapitalBudgetingDataPreprocessor(cfg, model_type, predict_feas, device)

    else:
        raise ValueError("Invalid problem type!")