'''
    This script generates a set of runs based on a grid search approach.
    There is a run for each possible combination of parameters.
'''
from itertools import product
import os
import json
import datetime as dt

'''
    Considering that by default head_num * head_dim = 256
    which is close to the attention hidden dimension of 200, I aim
    to keep the product of head_num and head_dim constant.

    I create a constant value to store the product of the two variables
    and vary only one of them and calculate the other accordingly.
'''
HEAD_NUM_TIMES_HEAD_DIM = 256

param_grid = {
    # ========================== FIXED PARAMETERS ============================

    # DEBUG SETTING
    "DEBUG": [False],

    # PATHS
    "DATASPLIT": ["ebnerd_demo"],
    "DATA_PATH": ["../data"],
    "DOCUMENT_EMBEDDINGS": ["Ekstra_Bladet_contrastive_vector/contrastive_vector.parquet"],

    # PARAMETERS DERIVED FROM TRIAL AND ERROR ON THE TESTING MACHINE
    "BS_TRAIN": [128],
    "BS_TEST": [128],
    "BATCH_SIZE_TEST_WO_B": [128],
    "BATCH_SIZE_TEST_W_B": [8],
    "FRACTION_TEST": [0.1],
    "N_CHUNKS_TEST": [1],
    "CHUNKS_DONE": [0],

    # I USE ALL THE TRAINING DATA
    "TRAIN_FRACTION": [1],

    # I KEEP THE OVERALL LEARNING CAPACITY OF THE ATTENTION LAYER CONSTANT
    "ATTENTION_HIDDEN_DIM": [200],

    # I KEEP THE TITLE SIZE CONSTANT
    "TITLE_SIZE": [768],

    # I KEEP SOME OF THE MACHINE LEARNING PARAMETERS CONSTANT TO REDUCE THE TOTAL NUMBER OF ITERATIONS
    "LEARNING_RATE": [1e-4],
    "NEWSENCODER_L2_REGULARIZATION": [1e-4],
    "NEWSENCODER_UNITS_PER_LAYER": [[512, 512, 512]],
    "OPTIMIZER": ["adam"],
    "LOSS": ["cross_entropy_loss"],

    # I KEEP THE DROPOUT CONSTANT
    "DROPOUT": [0.2],

    # SEED FOR THE PSEUDO-RANDOM NUMBER GENERATION
    "SEED": [123],

    # ========================== CHANGING PARAMETERS ============================ 
    "epochs": [5, 10],
    "history_size": [10, 15, 20],
    "npratio": [3, 5],
    "head_num": [16, 32],
    "nrms_loader": ["NRMSDataLoaderPretransform", "NRMSDataLoader"],
}

keys = param_grid.keys()
values = param_grid.values()

configs = [
    dict(zip(keys, combo))
    for combo in product(*values)
]

# PATHS
EXPERIMENTS_DIR = "../experiments"
DT_NOW = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
RUNS_DIR = os.path.join(EXPERIMENTS_DIR, f"runs-{DT_NOW}")

# I initialize the manifest as an empty dictionary
manifest = {}

for i in range(len(configs)):
    run_id = f"run_{i+1}"

    # I create the folder containing the run files
    run_folder = os.path.join(RUNS_DIR, run_id)
    os.makedirs(run_folder)

    # I calculate the parameter head_dim
    head_dim = HEAD_NUM_TIMES_HEAD_DIM // configs[i]["head_num"]

    '''
        Each configuration file has the following structure:

        "run_id": run_id,
        "parameters": {
            ...
        }

        In the parameters I add the head_dim parameter.
    '''
    parameters = configs[i]
    parameters["head_dim"] = head_dim

    config = {
        "run_id": run_id,
        "parameters": parameters
    }

    # I create the JSON file containing the run ID and all the parameters of the run
    config_path = os.path.join(run_folder, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    # I create the entry of the manifest dictionary
    manifest[run_id] = {
        "status": "PENDING",
        "run_folder": run_folder
    }

# I create the manifest file in the parent folder of the runs folders
MANIFEST_PATH = os.path.join(RUNS_DIR, "manifest.json")
with open(MANIFEST_PATH, "w") as f:
    json.dump(manifest, f, indent=2)
