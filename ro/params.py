from types import SimpleNamespace


# ---------------------#
#   Knapsack Problem   #
# ---------------------#

kp = SimpleNamespace(
    # type of data (general or instance)
    data_type = "general",

    # variable parameters between isntances
    n_items = [20, 30, 40, 50, 60, 70, 80],
    correlation = ["UN", "WC", "ASC", "SC"],
    h = [40, 80],
    delta = [0.1, 0.5, 1.0],
    budget_factor = [0.1, 0.15, 0.20],

    # fixed parameters between isntances
    R = 1000,
    H = 100,

    # data generation parameters
    time_limit = 30,            # for data generation only
    mip_gap = 0.01,             # for data generation only
    verbose = 0,                # for data generation only
    threads = 1,                # for data generation only
    tr_split=0.80,              # train/test split size
    
    n_samples_inst = 500,       # number of instances to samples
    n_samples_fs = 10,          # number of first-stage decisions samples per problem
    n_samples_per_fs = 50,      # number of uncertainty samples per first-stage decision

    # generic parameters
    seed = 7,
    data_path = './data/',

)


# --------------------------- #
#   Capital Budgeting Problem #
# --------------------------- #

cb = SimpleNamespace(
    # problem parameters
    n_items=[10, 20, 30, 40, 50],
    k=.8,
    loans=0,
    l=.12,    # default but not needed for this instance
    m=1.2,     # default but also not needed
    xi_dim=4,

    # data generation parameters
    obj_label="fs_plus_ss_obj",     # label for objective prediction
    feas_label="min_budget_cost",   # label for feasibility prediction
    
    time_limit=30,  # for data generation only
    mip_gap=0.01,   # for data generation only
    verbose=0,      # for data generation only
    threads=1,      # for data generation only

    tr_split=0.80,  # train/test split size

    n_samples_inst = 500,       # number of instances to samples
    n_samples_fs = 10,          # number of first-stage decisions samples per problem
    n_samples_per_fs = 50,      # number of uncertainty samples per first-stage decision

    # generic parameters
    seed=1,
    inst_seed=range(1, 101),
    data_path='./data/',
)

