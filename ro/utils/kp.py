from pathlib import Path


def lst_to_str(x):
    return "-".join(x)


def get_path(data_path, cfg, ptype="inst", suffix=".pkl", as_str=False):
    p = Path(data_path) / "kp"
    p.mkdir(parents=True, exist_ok=True)

    if cfg.data_type == "instance":
        p = p / f"{ptype}_" \
                f"I{cfg.n_items}_" \
                f"C{cfg.correlation}_" \
                f"h{cfg.h}_" \
                f"d{cfg.delta}" \
                f"R{cfg.R}_" \
                f"H{cfg.H}_" \
                f"nsi{cfg.n_samples_inst}_" \
                f"nsf{cfg.n_samples_fs}_" \
                f"nsu{cfg.n_samples_per_fs}_" \
                f"sd{cfg.seed}{suffix}"

    elif cfg.data_type == "general":
        p = p / f"{ptype}_" \
            f"n-all_" \
            f"c-{lst_to_str(cfg.correlation)}_" \
            f"nsi{cfg.n_samples_inst}_" \
            f"nsf{cfg.n_samples_fs}_" \
            f"nsu{cfg.n_samples_per_fs}_" \
            f"sd{cfg.seed}{suffix}"

    else:
        raise Exception("cfg.data_type must be one of \{general, instance\}")

    if as_str:
        return str(p)
    return p
