import argparse



def get_run_cmds(args):
    """ Generate list of commands for running ml-ccg. """
    enum_n_items = [20,30,40,50,60,70,80]
    enum_correlation = ['UN', 'WC', 'ASC', 'SC']
    enum_delta = [0.1, 0.5, 1]
    enum_h = [40, 80]
    enum_budget_factor = [0.1, 0.15, 0.20]

    cmd_list = []
    for n_items in  enum_n_items:
        for correlation in enum_correlation:
            for delta in enum_delta:
                for h in enum_h:
                    for budget_factor in enum_budget_factor:
                        
                        # inst/model params
                        if args.opt_type == "adversarial":
                            cmd = "python -m ro.scripts.05_run_ml_ccg "
                        elif args.opt_type == "pga":
                            cmd = "python -m ro.scripts.05_run_ml_ccg_pga "

                        if args.corr_specific:
                            cmd += f"--problem {args.problem}_{correlation.lower()} "
                        else:
                            cmd += f"--problem {args.problem} "

                        cmd += f"--model_type {args.model_type} "

                        # opt params
                        cmd += f"--mp_gap {args.mp_gap} "
                        cmd += f"--adversarial_gap {args.adversarial_gap} "
                        cmd += f"--mp_time {args.mp_time} "
                        cmd += f"--adversarial_time {args.adversarial_time} "
                        cmd += f"--mp_inc_time {args.mp_inc_time} "
                        cmd += f"--adversarial_inc_time {args.adversarial_inc_time} "
                        cmd += f"--mp_focus {args.mp_focus} "
                        cmd += f"--adversarial_focus {args.adversarial_focus} "

                        # kp specific params
                        cmd += f"--kp_n_items {n_items} "
                        cmd += f"--kp_correlation {correlation} "
                        cmd += f"--kp_delta {delta} "
                        cmd += f"--kp_h {h} "
                        cmd += f"--kp_budget_factor {budget_factor} "

                        if args.opt_type == "pga":
                            cmd += f"--pga_samples {args.pga_samples} "
                            cmd += f"--pga_epochs {args.pga_epochs} "
                            cmd += f"--n_procs {args.pga_n_procs} "

                        cmd += f"--opt_type {args.opt_type} "

                        cmd_list.append(cmd)

    return cmd_list

    
def get_eval_cmds(args):
    """ Generate list of commands for evaluating ml-ccg. """
    enum_n_items = [20,30,40,50,60,70,80]
    enum_correlation = ['UN', 'WC', 'ASC', 'SC']
    enum_delta = [0.1, 0.5, 1]
    enum_h = [40, 80]
    enum_budget_factor = [0.1, 0.15, 0.20]

    cmd_list = []
    for n_items in  enum_n_items:
        for correlation in enum_correlation:
            for delta in enum_delta:
                for h in enum_h:
                    for budget_factor in enum_budget_factor:
                        
                        # inst/model params
                        cmd = "python -m ro.scripts.06_eval_ml_ccg_obj "
                        
                        if args.corr_specific:
                            cmd += f"--problem {args.problem}_{correlation.lower()} "
                        else:
                            cmd += f"--problem {args.problem} "

                        cmd += f"--model_type {args.model_type} "

                        # kp specific params
                        cmd += f"--kp_n_items {n_items} "
                        cmd += f"--kp_correlation {correlation} "
                        cmd += f"--kp_delta {delta} "
                        cmd += f"--kp_h {h} "
                        cmd += f"--kp_budget_factor {budget_factor} "

                        cmd += f"--opt_type {args.opt_type} "

                        cmd_list.append(cmd)
                        
    return cmd_list

def main(args):

    if args.cmd_type == "run":
        cmd_list = get_run_cmds(args)

    elif args.cmd_type == "eval":
        cmd_list = get_eval_cmds(args)

    # write to text file
    textfile = open(args.file_name, "w")
    for i, cmd in enumerate(cmd_list[:-1]):
        textfile.write(f"{i + 1} {cmd}\n")
    textfile.write(f"{i + 2} {cmd_list[-1]}\n")
    textfile.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Writes commands to evaluate ml approach on all knapsack instances from paper.')

    parser.add_argument('--problem', type=str, default="kp")
    parser.add_argument('--model_type', type=str, default='set_encoder')

    parser.add_argument('--cmd_type', type=str, default='run', choices=['run', 'eval'])


    # ml-opt choices
    parser.add_argument('--mp_gap', type=float, default=1e-4, help='...')
    parser.add_argument('--adversarial_gap', type=float, default=1e-4, help='...')
    parser.add_argument('--mp_time', type=float, default=10800, help='...')
    parser.add_argument('--adversarial_time', type=float, default=10800, help='...')
    parser.add_argument('--mp_inc_time', type=float, default=180, help='...')
    parser.add_argument('--adversarial_inc_time', type=float, default=180, help='...')
    parser.add_argument('--mp_focus', type=int, default=0, help='MIPFocus for master')
    parser.add_argument('--adversarial_focus', type=int, default=0, help='MIPFocus for adversarial')

    # # algorithm choices 
    # parser.add_argument('--n_uncertainty_samples', type=int, default=100000, help='...')
    # parser.add_argument('--mp_gap', type=float, default=1e-3, help='...')
    # parser.add_argument('--adversarial_gap', type=float, default=1e-3, help='...')
    # parser.add_argument('--use_lp_relax', type=int, default=0, help='...')
    # parser.add_argument('--max_n_models', type=int, default=100, help='...')
    # parser.add_argument('--seed', type=int, default=1, help='...')

    # optimize type (pga or adversarial)
    parser.add_argument('--opt_type', type=str, default="adversarial", help='...')

    # pga specific parameters
    parser.add_argument('--pga_samples', type=int, default=20, help='...')
    parser.add_argument('--pga_epochs', type=int, default=100, help='...')
    parser.add_argument('--pga_n_procs', type=int, default=4, help='...')

    parser.add_argument('--corr_specific', type=int, default=0, help='...')

    # file name
    parser.add_argument('--file_name', type=str, default='table.dat')

    args = parser.parse_args()

    main(args)