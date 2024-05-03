import argparse

import ro.params as ro_params


# def main(args):

#     #emun_opt_type = ["sampling", "adversarial"]
#     #emun_obj_type = ["argmax", "max"]

#     cfg = getattr(ro_params, args.problem)


#     emun_opt_type = ["adversarial"]
#     emun_obj_type = ["argmax"]
#     enum_n_items = cfg.n_items # [10, 20, 30, 40, 50]
#     enum_inst_seed = list(range(1,51))

#     cmd_list = []

#     if args.eval_ssa and (args.eval_ml or args.eval_k_adapt):
#         raise Exception("Cannot evaluate ssa with either ml or k_adapt!")

#     # # commands for ml-based algorithm
#     # if args.eval_ml:
#     #     for opt_type in emun_opt_type:
#     #         for obj_type in emun_obj_type:
#     #             for n_items in  enum_n_items:
#     #                 for inst_seed in enum_inst_seed:
#     #                     cmd = "python -m ro.scripts.cb_eval_tiny_set "
#     #                     cmd += f"--problem {args.problem} "
#     #                     cmd += f"--model_type {args.model_type} "
#     #                     # todo: add these into pipeline
#     #                     # cmd += f"--n_uncertainty_samples {args.n_uncertainty_samples} "
#     #                     cmd += f"--mp_gap {args.mp_gap} "
#     #                     cmd += f"--adversarial_gap {args.adversarial_gap} "
#     #                     cmd += f"--mp_time {args.mp_time} "
#     #                     cmd += f"--adversarial_time {args.adversarial_time} "
#     #                     cmd += f"--mp_inc_time {args.mp_inc_time} "
#     #                     cmd += f"--adversarial_inc_time {args.adversarial_inc_time} "
#     #                     cmd += f"--mp_focus {args.mp_focus} "
#     #                     cmd += f"--adversarial_focus {args.adversarial_focus} "
#     #                     cmd += f"--use_lp_relax {args.use_lp_relax} "
#     #                     cmd += f"--max_n_models {args.max_n_models} "
#     #                     cmd += f"--n_items {n_items} "
#     #                     cmd += f"--inst_seed {inst_seed} "
#     #                     cmd += f"--opt_type {opt_type} "
#     #                     cmd += f"--obj_type {obj_type} "
#     #                     cmd += f"--use_exact_cons {args.use_exact_cons} "
#     #                     cmd += f"--use_item_specific {args.use_item_specific} "
#     #                     cmd += f"--time_limit {args.time_limit} "
#     #                     cmd += f"--verbose {args.verbose} "
#     #                     cmd_list.append(cmd)

#     # commands for k-adaptability baseline
#     # if args.eval_k_adapt:
#     #     for n_items in  enum_n_items:
#     #         for inst_seed in enum_inst_seed:
#     #             for k in args.K:
#     #                 cmd = "python -m ro.scripts.run_kadapt "
#     #                 cmd += f"--problem {args.problem} "
#     #                 cmd += f"--K {k} "
#     #                 cmd += f"--n_items {n_items} "
#     #                 cmd += f"--inst_seed {inst_seed} "
#     #                 cmd += f"--time_limit {args.time_limit} "
#     #                 cmd_list.append(cmd)

#     # commands for evaluation
#     # if args.eval_ssa:
#     #     for n_items in  enum_n_items:
#     #         for inst_seed in enum_inst_seed:
#     #             cmd = "python -m ro.scripts.cb_eval_ssa "
#     #             cmd += f"--problem {args.problem} "
#     #             cmd += f"--n_items {n_items} "
#     #             cmd += f"--inst_seed {inst_seed} "
#     #             cmd += f"--n_sample_scenarios {args.n_sample_scenarios} "
#     #             cmd += f"--n_procs {args.n_procs} "
#     #             cmd_list.append(cmd)

#     # write to text file
#     textfile = open(args.file_name, "w")
#     for i, cmd in enumerate(cmd_list[:-1]):
#         textfile.write(f"{i + 1} {cmd}\n")
#     textfile.write(f"{i + 2} {cmd_list[-1]}")
#     textfile.close()



def get_run_ml_cmds(args, cfg):
    """ Commands for ml-based CCG. """
    enum_n_items = cfg.n_items
    enum_inst_seed = list(range(1,51))

    cmd_list = []
    for n_items in  enum_n_items:
        for inst_seed in enum_inst_seed:
            # inst/model params
            if args.opt_type == "adversarial":
                cmd = "python -m ro.scripts.05_run_ml_ccg "
            elif args.opt_type == "pga":
                cmd = "python -m ro.scripts.05_run_ml_ccg_pga "
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

            # cb specific params
            cmd += f"--cb_n_items {n_items} "
            cmd += f"--cb_inst_seed {inst_seed} "

            # cb specific choices
            cmd += f"--cb_use_exact_cons {args.use_exact_cons} "
            # cmd += f"--cb_use_item_specific {args.use_item_specific} "
            cmd += f"--use_lp_relax {args.use_lp_relax} "

            # opt type
            if args.opt_type == "pga":
                cmd += f"--pga_samples {args.pga_samples} "
                cmd += f"--pga_epochs {args.pga_epochs} "
                cmd += f"--n_procs {args.pga_n_procs} "

            cmd += f"--opt_type {args.opt_type} "

            cmd_list.append(cmd)
                        
    return cmd_list



def get_run_k_adapt_cmds(args, cfg):
    """ Commands for k-adaptibility baselines. """
    enum_n_items = cfg.n_items
    enum_inst_seed = list(range(1,51))

    cmd_list = []
    for n_items in  enum_n_items:
        for inst_seed in enum_inst_seed:
            for k in args.baseline_k:
                cmd = "python -m ro.scripts.run_kadapt "
                cmd += f"--problem {args.problem} "
                cmd += f"--K {k} "
                cmd += f"--n_items {n_items} "
                cmd += f"--inst_seed {inst_seed} "
                cmd += f"--time_limit {args.time_limit} "
                cmd_list.append(cmd)
                        
    return cmd_list


def get_eval_cmds(args, cfg):
    """ Commands for evaluation. """
    enum_n_items = cfg.n_items
    enum_inst_seed = list(range(1,51))

    cmd_list = []
    for n_items in  enum_n_items:
        for inst_seed in enum_inst_seed:
                        
            # inst/model params
            cmd = "python -m ro.scripts.06_eval_ml_ccg_cons "
            cmd += f"--problem {args.problem} "
            cmd += f"--model_type {args.model_type} "

            # kp specific params
            cmd += f"--cb_n_items {n_items} "
            cmd += f"--cb_inst_seed {inst_seed} "

            cmd += f"--opt_type {args.opt_type} "

            cmd += f"--ssa_n_sample_scenarios {args.ssa_n_sample_scenarios} "
            cmd += f"--n_procs {args.n_procs} "

            cmd_list.append(cmd)

    return cmd_list


def main(args):
    """ Write commands to table.dat file.  """

    cfg = getattr(ro_params, args.problem)

    if args.cmd_type == "run_ml":
        cmd_list = get_run_ml_cmds(args, cfg)
    elif args.cmd_type == "run_k_adapt":
        cmd_list = get_run_k_adapt_cmds(args, cfg)
    elif args.cmd_type == "eval":
        cmd_list = get_eval_cmds(args, cfg)

    # write to text file
    textfile = open(args.file_name, "w")
    for i, cmd in enumerate(cmd_list[:-1]):
        textfile.write(f"{i + 1} {cmd}\n")
    textfile.write(f"{i + 2} {cmd_list[-1]}\n")
    textfile.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Writes commands to evaluate ml + k-adapt approach on all CB test instances.')

    parser.add_argument('--problem', type=str, default="cb")
    parser.add_argument('--model_type', type=str, default='set_encoder')

    # type of command to run
    parser.add_argument('--cmd_type', type=str, default='run_ml', choices=['run_ml', 'run_k_adapt', 'eval'])

    # config choices (model targets, etc.)
    # parser.add_argument('--use_item_specific', type=int, default=0, help='Item specific model')
    parser.add_argument('--use_lp_relax', type=int, default=0, help='...')
    parser.add_argument('--n_uncertainty_samples', type=int, default=100000, help='...')
    parser.add_argument('--use_exact_cons', type=int, default=1, help='...')

    # 
    # parser.add_argument('--max_n_models', type=int, default=100, help='...')
    # parser.add_argument('--seed', type=int, default=1, help='...')

    # optimize type (pga or adversarial)
    parser.add_argument('--opt_type', type=str, default="adversarial", help='...')


    # ml-opt choices
    parser.add_argument('--mp_gap', type=float, default=1e-4, help='...')
    parser.add_argument('--adversarial_gap', type=float, default=1e-4, help='...')
    parser.add_argument('--mp_time', type=float, default=10800, help='...')
    parser.add_argument('--adversarial_time', type=float, default=10800, help='...')
    parser.add_argument('--mp_inc_time', type=float, default=180, help='...')
    parser.add_argument('--adversarial_inc_time', type=float, default=180, help='...')
    parser.add_argument('--mp_focus', type=int, default=1, help='MIPFocus for master')
    parser.add_argument('--adversarial_focus', type=int, default=1, help='MIPFocus for adversarial')

    # debugging
    parser.add_argument('--verbose', type=int, default=1, help='Verbose output for debugging gurobi model')

    # time limit
    parser.add_argument('--time_limit', type=int, default=3 * 3600, help='...')

    # baseline choices k-adaptability
    # parser.add_argument('--eval_k_adapt', type=int, default=1, help='...')
    parser.add_argument('--baseline_k', nargs="+", type=int, default=[2, 5, 10], help='...')

    # final evaluation choices
    parser.add_argument('--eval_type', type=str, default="ssa", help='...')
    parser.add_argument('--ssa_n_sample_scenarios', type=int, default=10000, help='...')

    #
    parser.add_argument('--n_procs', type=int, default=1, help='...')

    # pga specific parameters
    parser.add_argument('--pga_samples', type=int, default=20, help='...')
    parser.add_argument('--pga_epochs', type=int, default=100, help='...')
    parser.add_argument('--pga_n_procs', type=int, default=4, help='...')

    # file name
    parser.add_argument('--file_name', type=str, default='table.dat')

    args = parser.parse_args()

    main(args)