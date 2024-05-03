from pathlib import Path
import numpy as np


def get_path(data_path, cfg, ptype="inst", suffix=".pkl", as_str=False, ml=True):
    _dir = "cb"
    if cfg.loans:
        _dir += "_loans"
    p = Path(data_path) / _dir
    p.mkdir(parents=True, exist_ok=True)

    if ml:
        if len(cfg.n_items) == 1:
            p = p / f"{ptype}_" \
                    f"n-{cfg.n_items[0]}_" \
                    f"ol-{cfg.obj_label}_" \
                    f"fl-{cfg.feas_label}_" \
                    f"nsi{cfg.n_samples_inst}_" \
                    f"nsf{cfg.n_samples_fs}_" \
                    f"nsu{cfg.n_samples_per_fs}_" \
                    f"sd{cfg.seed}{suffix}"
        else:
            p = p / f"{ptype}_" \
                    f"n-all_" \
                    f"ol-{cfg.obj_label}_" \
                    f"fl-{cfg.feas_label}_" \
                    f"nsi{cfg.n_samples_inst}_" \
                    f"nsf{cfg.n_samples_fs}_" \
                    f"nsu{cfg.n_samples_per_fs}_" \
                    f"sd{cfg.seed}{suffix}"
    else:
        p = p / f"{ptype}_sd{cfg.seed}{suffix}"

    if as_str:
        return str(p)
    return p


def read_test_instance(inst_dir, cfg, inst_seed, n_items):
    """ Load instances data from paper. """
    inst_name = f"CB_n{n_items}"
    inst_name += f"_xi{cfg.xi_dim}"
    inst_name += f"_s{inst_seed}"

    inst = {}
    inst['inst_name'] = inst_name
    inst['k'] = cfg.k
    inst['loans'] = cfg.loans
    inst['l'] = cfg.l
    inst['m'] = cfg.m

    fp_inst = inst_dir + inst_name
    with open(fp_inst, 'r') as f:
        f_lines = f.readlines()

    # get metainfo, only C is required
    inst_info = list(map(lambda x: float(x), f_lines[0].replace('\n', '').split(' ')))
    inst['seed'] = int(inst_info[0])
    inst['n_items'] = int(inst_info[1])
    inst['xi_dim'] = int(inst_info[2])
    inst['budget'] = int(inst_info[3])
    inst['max_loan'] = int(inst_info[4])

    # store inst data
    c_bar = np.array(list(map(lambda x: float(x), f_lines[1].replace('\n', '').split(' '))))
    inst['c_bar'] = c_bar
    r_bar = np.array(list(map(lambda x: float(x), f_lines[2].replace('\n', '').split(' '))))
    inst['r_bar'] = r_bar

    phi = dict()
    for i in range(inst['n_items']):
        phi[i] = np.array(list(map(lambda x: float(x), f_lines[3+i].replace('\n', '').split(' '))))
    inst['phi'] = phi

    psi = dict()
    for i in range(inst['n_items']):
        psi[i] = np.array(list(map(lambda x: float(x), f_lines[3 + inst['n_items'] + i].replace('\n', '').split(' '))))
    inst['psi'] = psi
    return inst