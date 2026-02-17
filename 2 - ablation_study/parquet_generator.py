'''
    This file runs through the runs folder, gathers the run parameters and
    results and uses the gathered data to fill a polars dataframe which
    will be stored permanently as a parquet file.
'''
from pathlib import Path
import json
import polars as pl

RUNS_DIRECTORY = Path(r"..\experiments\runs-ablation")

# I initialize an empty dataframe with a schema
df_schema = {
        # ==================== RUN PARAMETERS =========================
        # Fixed parameters
        "run id": pl.String,
        "debug": pl.Boolean,
        "datasplit": pl.String,
        "data path": pl.String,
        "document embeddings": pl.String,
        "bs train": pl.Int16,
        "bs test": pl.Int16,
        "batch size test wo b": pl.Int16,
        "batch size test w b": pl.Int8,
        "fraction test": pl.Float64,
        "n chunks test": pl.Int16,
        "chunks done": pl.Int16,
        "train fraction": pl.Int16,
        "attention hidden dimension": pl.Int16,
        "title size": pl.Int16,
        "learning rate": pl.Float64,
        "newsencoder l2 regularization": pl.Float64,
        "newsencoder units per layer": pl.List,
        "optimizer": pl.String,
        "loss": pl.String,
        "dropout": pl.Float64,
        "seed": pl.Int8,
        "epochs": pl.Int8,
        "history size": pl.Int8,
        "npratio": pl.Int8,
        "head num": pl.Int8,
        "head dim": pl.Int8,
        "nrms loader": pl.String,

        # =================== USED MODEL =============================
        "used model": pl.String,

        # ==================== CALCULATED METRICS ====================
        "auc": pl.Float64,
        "mrr": pl.Float64,
        "ndcg@5": pl.Float64,
        "ndcg@10": pl.Float64
}
df = pl.DataFrame(schema=df_schema)

for item in RUNS_DIRECTORY.iterdir():
    if item.is_dir():
        # Files paths
        config_path = item / "config.json"
        metrics_path = item / "metrics.json"

        with open(config_path, "r") as config_file:
            with open(metrics_path, "r") as metrics_file:
                parameters = json.load(config_file)
                internal_parameter_handle = parameters["parameters"]
                metrics = json.load(metrics_file)

                # USED MODEL
                pe = internal_parameter_handle["use_pe"]
                rln = internal_parameter_handle["use_rln"]
                td = internal_parameter_handle["use_td"]

                used_model = None

                match (pe, rln, td):
                    case (False, False, False):
                        used_model = "Baseline"
                    case (False, False, True):
                        used_model = "TD"
                    case (False, True, False):
                        used_model = "RLN"
                    case (False, True, True):
                        used_model = "RLN+TD"
                    case (True, False, False):
                        used_model = "PE"
                    case (True, False, True):
                        used_model = "PE+TD"
                    case (True, True, False):
                        used_model = "PE+RLN"
                    case (True, True, True):
                        used_model = "PE+RLN+TD"

                # Now I can create the new tuple of the dataframe
                new_row = pl.DataFrame(
                    data={
                        # ==================== RUN PARAMETERS =========================
                        "run id": [parameters["run_id"]],
                        "debug": [internal_parameter_handle["DEBUG"]],
                        "datasplit": [internal_parameter_handle["DATASPLIT"]],
                        "data path": [internal_parameter_handle["DATA_PATH"]],
                        "document embeddings": [internal_parameter_handle["DOCUMENT_EMBEDDINGS"]],
                        "bs train": [internal_parameter_handle["BS_TRAIN"]],
                        "bs test": [internal_parameter_handle["BS_TEST"]],
                        "batch size test wo b": [internal_parameter_handle["BATCH_SIZE_TEST_WO_B"]],
                        "batch size test w b": [internal_parameter_handle["BATCH_SIZE_TEST_W_B"]],
                        "fraction test": [internal_parameter_handle["FRACTION_TEST"]],
                        "n chunks test": [internal_parameter_handle["N_CHUNKS_TEST"]],
                        "chunks done": [internal_parameter_handle["CHUNKS_DONE"]],
                        "train fraction": [internal_parameter_handle["TRAIN_FRACTION"]],
                        "attention hidden dimension": [internal_parameter_handle["ATTENTION_HIDDEN_DIM"]],
                        "title size": [internal_parameter_handle["TITLE_SIZE"]],
                        "learning rate": [internal_parameter_handle["LEARNING_RATE"]],
                        "newsencoder l2 regularization": [internal_parameter_handle["NEWSENCODER_L2_REGULARIZATION"]],
                        "newsencoder units per layer": [internal_parameter_handle["NEWSENCODER_UNITS_PER_LAYER"]],
                        "optimizer": [internal_parameter_handle["OPTIMIZER"]],
                        "loss": [internal_parameter_handle["LOSS"]],
                        "dropout": [internal_parameter_handle["DROPOUT"]],
                        "seed": [internal_parameter_handle["SEED"]],
                        "epochs": [internal_parameter_handle["epochs"]],
                        "history size": [internal_parameter_handle["history_size"]],
                        "npratio": [internal_parameter_handle["npratio"]],
                        "head num": [internal_parameter_handle["head_num"]],
                        "nrms loader": [internal_parameter_handle["nrms_loader"]],
                        "head dim": [internal_parameter_handle["head_dim"]],

                        # ==================== USED MODEL ============================
                        "used model": used_model,

                        # ==================== CALCULATED METRICS ====================
                        "auc": [metrics["auc"]],
                        "mrr": [metrics["mrr"]],
                        "ndcg@5": [metrics["ndcg@5"]],
                        "ndcg@10": [metrics["ndcg@10"]]
                    },
                    schema=df_schema
                )

                df = pl.concat([df, new_row])

# I store the dataframe permanently in a CSV file
df.write_parquet("./results.parquet")
