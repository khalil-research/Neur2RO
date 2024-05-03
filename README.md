# Neur2RO
Implementation of "Neur2RO: Neural Two-Stage Robust Optimization" ([https://arxiv.org/pdf/2205.12006.pdf](https://arxiv.org/pdf/2310.04345.pdf)).  

This repository contains all the implementations required to reproduce the experiments from the paper, as well as the trained models and results.  In addition, the repository is modular enough to support the implementation of new two-stage robust optimization (2RO) problems with only a few files needed.  Below, we include an example of running the pipeline for the two-stage robust knapsack problem and details of the files needed for adding new 2RO problems.  


## Example:  Two-Stage Robust Knapsack Problem

This section includes commands to reproduce results for the two-stage robust knapsack problem.

Initialize the data folder and problem.
```
python -m ro.scripts.00_init_directories --problem kp
python -m ro.scripts.01_init_problem --problem kp 
```



To run data collection, execute the following command.  When implementing new problems or debugging, set `--debug 1` to prevent multiprocessing, which will make debugging easier.  Note that the data collection in the repository does not exactly reproduce the dataset in the paper.  However, previous datasets are available at [ADD].  
```
python -m ro.scripts.02_generate_dataset --problem kp --n_procs [N] --debug 0
```

Train and save the model
```
python -m ro.scripts.03_train_model --problem kp --x_embed_dims 32 16 --x_post_agg_dims 64 8 --xi_embed_dims 32 16 --xi_post_agg_dims 64 8 --value_dims 8  --n_epochs 500
python -m ro.scripts.04_get_best_model --problem kp
```

Produce a list of commands to run ml-based ccg and evaluate solutions for all instances.  Note that these commands will require quite a long time to run sequentially, so utilizing parallel resources is recommended.  
```
python -m ro.scripts.dat_scripts.kp_gen_eval_table --cmd_type run
python -m ro.scripts.dat_scripts.kp_gen_eval_table --cmd_type eval
```


## Adding New 2RO Problems

To add a new 2RO problem, a few files will need to be added/edited. Specifically, for adding a problem `p`, the following files need to be added:
- `ro/params.py`: This file contains all information related to the distribution of instances, such as the number of items in the knapsack problem, and data collection parameters, such as the number of samples.  Problem 'p' should also be added here and `kp` or `cb` can be used as an example.  
- `ro/utils/p.py`: Implements `get_path`, i.e., file naming for storing data. The path should ideally depend on the parameters defined in `ro/params.py` to make iterating over different sets of instances/parameters more streamlined.  
- `ro/two_ro/p.py`: Implements solving the inner-most optimization problem and sampling instances.
- `ro/dm/p.py`: Implements problem generation (a `.pkl` file containing useful parameters) and data collection.  For data collection, one must also implement the sampling of first-stage decisions and uncertainty. In addition, `solve_second_stage,` which calls `two_ro.solve_second_stage` and collects info for ML features/labels, should be done here.
- `ro/data_preprocessor/p.py`: This implements converting the raw features from `ro/dm/p.py` into features for pytorch. 
- `ro/approximator/p.py`: This implements solving the CCG approximation, specifically, the main and adversarial problems, as well as adding models.  
- `ro/*/__init__.py`: These files will need to be modified to include imports and factory functions for the respective `ro/*/p.py` files.
- `ro/scripts/*`:  In this section, only `05_run_ml_ccg.py`, `06_eval_ml_ccg_obj.py`, `06_eval_ml_ccg_cons.py` should need to be modified.  For `05_run_ml_ccg.py`, only getting only the parameters for the instance, as well as reading/sampling the evaluation instance need to be implemented.  For `06_eval_ml_ccg_obj.py` or `06_eval_ml_ccg_cons.py`, only one of the files needs to be modified depending on whether the 2RO problem has only objective or constraint uncertainty, respectively.  In either case, solutions from `05_run_ml_ccg.py` should be read and possibly compared to appropriate baselines.
- `ro/scripts/dat_scripts/*`:  If evaluating a lot of instances, then it may be useful to implement scripts that can generate a set of commands to run.  See the examples for the knapsack and capital budget for reference.  

## Adding New Machine Learning Models
If one wants to add a new machine learning model for specific problem `p` (or even all problems), it is recommended that the following files be modified/added.
- `ro/data_preprocessor/p.py`: Convert raw features from `ro/dm/p.py` to machine learning ready features/datasets.
- `ro/models/new_model.py`: Implements the machine learning model.
- `ro/scripts/03_train_model.py`:  The training pipeline for the new model.  Alternatively, it may be easier to create a separate file in this case.
- `ro/approximator/p.py`:  Here, only modifications to how the model's machine learning models are added to Gurobi should be changed.



## Reference

Please cite our work if you find our code/paper useful to your work.

```
  @article{dumouchelle2023neur2ro,
    title={Neur2RO: Neural Two-Stage Robust Optimization},
    author={Dumouchelle, Justin and Julien, Esther and Kurtz, Jannis and Khalil, Elias B},
    journal={arXiv preprint arXiv:2310.04345},
    year={2023}
}
```

## Benchmark Instances

### Knapsack
- Reference: Arslan, A. N., & Detienne, B. (2022). Decomposition-based approaches for a class of two-stage robust binary optimization problems. INFORMS journal on computing, 34(2), 857-871.
- Link to instances: https://github.com/borisdetienne/RobustDecomposition/tree/main/TwoStageRobustKnapsack
- These instances should be downloaded, unzipped, and moved to `data/kp/eval_instances/`.  The results from branch-and-price (`ResultsForBranchAndPrice.csv`) should downloaded and be moved to `data/kp/baseline_results/`.
  
### Capital Budgeting
- Reference: Subramanyam, A., Gounaris, C. E., & Wiesemann, W. (2020). K-adaptability in two-stage mixed-integer robust optimization. Mathematical Programming Computation, 12, 193-224.
- These instances have been generated according to the procedure in the paper and are already available in `data/cb/eval_instances/`.


For any questions, please contact justin.dumouchelle@mail.utoronto.ca
