import argparse




def main(args):

    emun_opt_type = ["sampling", "adversarial"]
    emun_obj_type = ["argmax", "max"]
    enum_n_items = [20,30,40,50,60,70,80]
    enum_correlation = ['UN', 'WC', 'ASC', 'SC']
    enum_delta = [0.1, 0.5, 1]
    enum_h = [40, 80]
    enum_budget_factor = [0.1, 0.15, 0.20]

    cmd_list = []
    for opt_type in emun_opt_type:
        for obj_type in emun_obj_type:
            for n_items in  enum_n_items:
                for correlation in enum_correlation:
                    for delta in enum_delta:
                        for h in enum_h:
                            for budget_factor in enum_budget_factor:
                                cmd = "python -m ro.scripts.appendix_scripts.kp_eval_net "
                                cmd += f"--problem {args.problem} "
                                cmd += f"--model_type {args.model_type} "
                                cmd += f"--mp_gap {args.mp_gap} "
                                cmd += f"--n_uncertainty_samples {args.n_uncertainty_samples} "
                                cmd += f"--adversarial_gap {args.adversarial_gap} "
                                cmd += f"--max_n_models {args.max_n_models} "
                                cmd += f"--n_items {n_items} "
                                cmd += f"--correlation {correlation} "
                                cmd += f"--delta {delta} "
                                cmd += f"--h {h} "
                                cmd += f"--budget_factor {budget_factor} "
                                cmd += f"--opt_type {opt_type} "
                                cmd += f"--obj_type {obj_type} "
                                cmd_list.append(cmd)


    # write to text file
    textfile = open(args.file_name, "w")
    for i, cmd in enumerate(cmd_list[:-1]):
        textfile.write(f"{i + 1} {cmd}\n")
    # textfile.write(f"{i + 2} {cmd_list[-1]}")
    textfile.write(f"{i + 2} {cmd_list[-1]}\n")
    textfile.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Writes commands to evaluate ml approach on all knapsack instances from paper.')

    parser.add_argument('--problem', type=str, default="kp")
    parser.add_argument('--model_type', type=str, default='set_encoder')

    # algorithm choices 
    parser.add_argument('--n_uncertainty_samples', type=int, default=100000, help='...')
    parser.add_argument('--mp_gap', type=float, default=1e-4, help='...')
    parser.add_argument('--adversarial_gap', type=float, default=1e-4, help='...')
    parser.add_argument('--max_n_models', type=int, default=100, help='...')
    parser.add_argument('--seed', type=int, default=1, help='...')

    # file name
    parser.add_argument('--file_name', type=str, default='table.dat')

    args = parser.parse_args()

    main(args)