import ro.params as params


def factory_two_ro(problem):
    """ Initializes TwoRO class.  """
    if "kp" in problem:
        print("Loading Knapsack data manager...")
        from .kp import Knapsack
        return Knapsack()

    if "cb" in problem:
        print("Loading Capital Budgeting data manager...")
        from .cb import CapitalBudgeting
        return CapitalBudgeting()

    # add new problems here
    
    else:
        raise ValueError("Invalid problem type!")
