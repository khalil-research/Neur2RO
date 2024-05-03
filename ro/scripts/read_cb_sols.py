import pandas as pd
import pickle as pkl


def main():
    df = pd.DataFrame(columns=["ccg_obj", "ccg_time", "K2_obj", "K2_time", "K5_obj", "K5_time",
                               "K10_obj", "K10_time"])
    for n in [10, 20, 30, 40, 50, 60, 70]:
        for s in range(1, 19):
            inst_name = f"CB_n{n}_xi4_s{s}"
            try:
                with open(f"./data/cb/ccg_res/res_{inst_name}.pkl", "rb") as handle:
                    res = pkl.load(handle)
                ccg_obj = res['obj']
                ccg_time = res['runtime']
            except FileNotFoundError:
                ccg_obj = None
                ccg_time = None

            K_obj = dict()
            K_time = dict()

            for K in [2, 5, 10]:
                try:
                    with open(f"./data/cb/k_adapt_res/res_{inst_name}_K{K}.pkl", "rb") as handle:
                        res = pkl.load(handle)
                    K_obj[K] = res['obj']
                    K_time[K] = res['runtime']
                except FileNotFoundError:
                    K_obj[K] = None
                    K_time[K] = None

            df.loc[inst_name] = [ccg_obj, ccg_time, K_obj[2], K_time[2], K_obj[5], K_time[5], K_obj[10], K_time[10]]
    # save
    df.to_csv("./data/cb/ResultsCCG_KAdapt.csv")

main()
