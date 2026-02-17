from pathlib import Path
import tensorflow as tf
import datetime as dt
import polars as pl
import numpy as np
import gc
import os
import json
import argparse

from ebrec.utils._constants import *

from ebrec.utils._behaviors import (
    create_binary_labels_column,
    sampling_strategy_wu2019,
    add_prediction_scores,
    ebnerd_from_path,
)
from ebrec.evaluation import MetricEvaluator, AucScore, NdcgScore, MrrScore

from ebrec.utils._python import (
    write_submission_file,
    rank_predictions_by_score,
    write_json_file,
)

from ebrec.models.newsrec.model_config import (
    hparams_nrms_docvec,
    hparams_to_dict,
)

from nrms_docvec_td import NRMSDocVecTd

from new_dataloader import NewNRMSDataLoaderPretransform, create_article_id_to_value_mapping

os.environ["TOKENIZERS_PARALLELISM"] = "false"

NEW_DATA_PATH = Path("../new_data").resolve()

DOCVEC_COL = "contrastive_vector"

'''
    This script runs the single experiment as a separated subprocess
    from runs_executor.py. This is to prevent RAM saturation because
    after each subprocess terminates, memory is released. This grants
    that all the experiments are independent from each other and also,
    if the single experiments don't saturate the RAM, the whole
    process won't.
'''

''' ===================== PHASE 1: PARAMETERS RETRIEVAL ====================================== '''
# I retrieve the absolute path of the run folder from the arguments of the command line
parser = argparse.ArgumentParser(
    description="Argument parser that retrieves the absolute path of the run folder from the command line"
)

parser.add_argument(
    "--absolute_path_run_folder",
    type=str,
    help="Absolute path of the run folder of the single experiment"
)

args = parser.parse_args()
absolute_path_run_folder = Path(args.absolute_path_run_folder)

# I retrieve the parameters from the run folder
parameters_path = absolute_path_run_folder.joinpath("config.json")

with open(parameters_path, "r") as parameters_file:
    parameters = json.load(parameters_file)
    # Handle to the internal parameters object
    parameters = parameters["parameters"]

    # ================= RESULTS DIRECTORY ======================================
    results_dir = absolute_path_run_folder

    # ==================== FIXED PARAMETERS ====================================
    seed = parameters["SEED"]
    datasplit =  parameters["DATASPLIT"]
    debug = parameters["DEBUG"]
    data_path = parameters["DATA_PATH"]
    document_embeddings = parameters["DOCUMENT_EMBEDDINGS"]
    bs_train = parameters["BS_TRAIN"]
    bs_test = parameters["BS_TEST"]
    batch_size_test_wo_b = parameters["BATCH_SIZE_TEST_WO_B"]
    batch_size_test_w_b = parameters["BATCH_SIZE_TEST_W_B"]
    fraction_test = parameters["FRACTION_TEST"]
    n_chunks_test = parameters["N_CHUNKS_TEST"]
    chunks_done = parameters["CHUNKS_DONE"]
    train_fraction = parameters["TRAIN_FRACTION"]
    attention_hidden_dim = parameters["ATTENTION_HIDDEN_DIM"]
    title_size = parameters["TITLE_SIZE"]
    learning_rate = parameters["LEARNING_RATE"]
    newsencoder_l2_regularization = parameters["NEWSENCODER_L2_REGULARIZATION"]
    newsencoder_units_per_layer = parameters["NEWSENCODER_UNITS_PER_LAYER"]
    dropout = parameters["DROPOUT"]
    optimizer = parameters["OPTIMIZER"]
    loss = parameters["LOSS"]

    epochs = parameters["epochs"]
    history_size = parameters["history_size"]
    npratio = parameters["npratio"]
    head_num = parameters["head_num"]
    nrms_loader = parameters["nrms_loader"]
    head_dim = parameters["head_dim"]

    # =========== PREPROCESSING RELATED PARAMETERS OF THE MODEL ==============
    user_feat_dim = 2 # Premium propensity & Reading engagement
    item_feat_dim = 1 # Premium propensity

    # Path of articles.parquet
    ARTICLES_PATH = Path(datasplit).joinpath("articles.parquet")

    ''' ===================== PHASE 2: MODEL TRAINING ====================================== '''
    # Reproduce the original training procedure:
    # - load docvec artifacts
    # - build train+validation, sample, negative sampling (Wu2019), binary labels
    # - split last day as validation
    # - train with callbacks + val_auc early stopping + load best weights
    tf.random.set_seed(seed)

    PATH = Path(data_path).expanduser()

    # To keep the folders tidy I put the deliverables folder inside the results folder
    results_dir = Path(results_dir).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    deliverables_folder = results_dir / "deliverables"
    deliverables_folder.mkdir(parents=True, exist_ok=True)

    # ------------------ Choose DataLoader ----------------------
    NRMSLoader_training = NewNRMSDataLoaderPretransform

    # ------------------ Set hparams --------------------------------------
    hparams = hparams_nrms_docvec

    hparams.user_feat_dim = user_feat_dim
    hparams.item_feat_dim = item_feat_dim

    hparams.history_size = history_size
    hparams.head_num = head_num
    hparams.head_dim = head_dim
    hparams.attention_hidden_dim = attention_hidden_dim
    hparams.newsencoder_units_per_layer = newsencoder_units_per_layer
    hparams.optimizer = optimizer
    hparams.loss = loss
    hparams.dropout = dropout
    hparams.learning_rate = learning_rate
    hparams.newsencoder_l2_regularization = newsencoder_l2_regularization

    ''' ------------------ LOAD ARTICLE DOCVEC MAPPING ------------------ '''
    DOC_VEC_PATH = PATH.joinpath("artifacts").joinpath(document_embeddings)
    df_articles = pl.read_parquet(DOC_VEC_PATH)

    ''' ------------------ BEGIN OF ADDITIONAL ARTICLES PREPROCESSING --------------- '''
    item_feature_cols = []

    ''' Premiumness integration '''
    df_premiumness = pl.read_parquet(NEW_DATA_PATH / "premium_articles.parquet")
    df_articles = df_articles.join(
        other=df_premiumness,
        on=DEFAULT_ARTICLE_ID_COL,
        how="inner"
    )

    # I add the column to the list
    item_feature_cols.append("premium")

    # I create a dictionary to associate article IDs to features
    item_feature_dict = None

    if item_feature_cols:
        df_item = df_articles.select([DEFAULT_ARTICLE_ID_COL] + item_feature_cols)
        # cast a float32 e fill null
        for c in item_feature_cols:
            df_item = df_item.with_columns(pl.col(c).cast(pl.Float32).fill_null(0.0))
        item_feature_dict = {
            int(r[0]): np.array(r[1:], dtype=np.float32)
            for r in df_item.iter_rows()
        }
    ''' ------------------ END OF ADDITIONAL ARTICLES PREPROCESSING ------------------ '''

    article_mapping = create_article_id_to_value_mapping(
        df=df_articles, value_col=DOCVEC_COL
    )

    # Persist run config (helpful for reproducibility)
    write_json_file(hparams_to_dict(hparams), deliverables_folder / "NRMSDocVecAblation_hparams.json")
    write_json_file(parameters, deliverables_folder / "NRMSDocVecAblation_run_parameters.json")

    # ------------------ Build train/val dataset (same as original) ----------------------
    '''
        DEFAULT SELECTION OF COLUMNS
        
        These columns only belong to the behaviors dataframe because by default
        the ebnerd_from_path method retrieves the columns "user_id" and "article_id_fixed"
        from the history dataframe and makes an inner join with the behaviors dataframe on the
        "user_id" column.
        
        During the preprocessing we may need to retrieve other columns from other dataframes.
        We do so by retrieving an additional dataframe containing the desired information 
        and making an inner join with the default dataframe using the "user_id" column.
    '''

    COLUMNS = [
        DEFAULT_IMPRESSION_TIMESTAMP_COL, # behaviors["impression_time"]
        DEFAULT_HISTORY_ARTICLE_ID_COL, # behaviors["article_id_fixed"]
        DEFAULT_INVIEW_ARTICLES_COL, # behaviors["article_ids_inview"]
        DEFAULT_CLICKED_ARTICLES_COL, # behaviors["article_ids_clicked"]
        DEFAULT_IMPRESSION_ID_COL, # behaviors["impression_id"]
        DEFAULT_USER_COL, # behaviors["user_id"]
    ]

    train_fraction_eff = train_fraction if not debug else 0.0001

    ''' DEFAULT DATAFRAMES RETRIEVAL '''
    df = (
        pl.concat(
            [
                ebnerd_from_path(
                    PATH.joinpath(datasplit, "train"),
                    history_size=history_size,
                    padding=0,
                ),
                ebnerd_from_path(
                    PATH.joinpath(datasplit, "validation"),
                    history_size=history_size,
                    padding=0,
                ),
            ]
        )
        .sample(fraction=train_fraction_eff, shuffle=True, seed=seed)
        .select(COLUMNS)
        .pipe(
            sampling_strategy_wu2019,
            npratio=npratio,
            shuffle=True,
            with_replacement=True,
            seed=seed,
        )
        .pipe(create_binary_labels_column)
    )

    df_eval = (
        ebnerd_from_path(
            PATH.joinpath(datasplit, "validation"),
            history_size=history_size,
            padding=0,
        )
        .select(COLUMNS)
        .pipe(create_binary_labels_column)  # produces DEFAULT_LABELS_COL aligned with inview
    )

    ''' BEGINNING OF ADDITIONAL PREPROCESSING OPERATIONS ON THE USERS '''
    user_feature_cols = []

    ''' Premiumness categorization data '''
    # I retrieve the association between user and premium propensity
    df_premium_propensity = pl.read_parquet(NEW_DATA_PATH / "premium_propensity.parquet")

    # I add the premium propensity to the main dataframe
    df = df.join(
        other=df_premium_propensity,
        on=DEFAULT_USER_COL,
        how="inner"
    )

    # I add the premium propensity to the evaluation dataframe
    df_eval = df_eval.join(
        other=df_premium_propensity,
        on=DEFAULT_USER_COL,
        how="inner"
    )

    # I append the column to the list
    user_feature_cols.append("premium_propensity")

    ''' Reading engagement data '''
    # I retrieve the association between user and reading engagement
    df_reading_engagement = pl.read_parquet(NEW_DATA_PATH / "reading_engagement.parquet")

    # I add the reading engagement to the main dataframe
    df = df.join(
        other=df_reading_engagement,
        on=DEFAULT_USER_COL,
        how="inner"
    )

    # I add the reading engagement to the evaluation dataframe
    df_eval = df_eval.join(
        other=df_reading_engagement,
        on=DEFAULT_USER_COL,
        how="inner"
    )

    # I append the column to the list
    user_feature_cols.append("reading_engagement")

    ''' END OF ADDITIONAL PREPROCESSING OPERATIONS ON THE USERS '''

    # TRAIN-TEST SPLIT
    last_dt = df[DEFAULT_IMPRESSION_TIMESTAMP_COL].dt.date().max() - dt.timedelta(days=1)
    df_train = df.filter(pl.col(DEFAULT_IMPRESSION_TIMESTAMP_COL).dt.date() < last_dt)
    df_validation = df.filter(pl.col(DEFAULT_IMPRESSION_TIMESTAMP_COL).dt.date() >= last_dt)

    train_dataloader = NRMSLoader_training(
        behaviors=df_train,
        article_dict=article_mapping,
        unknown_representation="zeros",
        history_column=DEFAULT_HISTORY_ARTICLE_ID_COL,
        eval_mode=False,
        batch_size=int(bs_train),
        kwargs=dict(
            user_feature_cols=user_feature_cols,
            item_feature_dict=item_feature_dict,
        ),
    )
    val_dataloader = NRMSLoader_training(
        behaviors=df_validation,
        article_dict=article_mapping,
        unknown_representation="zeros",
        history_column=DEFAULT_HISTORY_ARTICLE_ID_COL,
        eval_mode=False,
        batch_size=int(bs_train),
        kwargs=dict(
            user_feature_cols=user_feature_cols,
            item_feature_dict=item_feature_dict,
        ),
    )

    # ------------------ Callbacks -----------------------------------
    weights_dir = deliverables_folder / "weights"
    weights_dir.parent.mkdir(parents=True, exist_ok=True)

    log_dir = deliverables_folder / "runs"
    log_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=4,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(weights_dir),  # TF accepts str path
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_auc",
            mode="max",
            factor=0.2,
            patience=2,
            min_lr=1e-6,
        ),
    ]

    model = NRMSDocVecTd(hparams=hparams, seed=42)
    model.model.compile(
        optimizer=model.model.optimizer,
        loss=model.model.loss,
        metrics=["AUC"],
    )

    print("\nModel training:")
    hist = model.model.fit(
        train_dataloader,
        validation_data=val_dataloader,
        epochs=int(epochs),
        callbacks=callbacks,
    )

    # Load best weights (as original)
    model.model.load_weights(str(weights_dir))

    ''' ===================== PHASE 3: PREDICTION ====================================== '''
    # Evaluate on the *true* validation split (datasplit/validation) with real clicks/inview
    # so we can compute AUC/MRR/NDCG@k.
    #
    # IMPORTANT: predictions must be score lists, not ranks.

    eval_dataloader = NRMSLoader_training(
        behaviors=df_eval,
        article_dict=article_mapping,
        unknown_representation="zeros",
        history_column=DEFAULT_HISTORY_ARTICLE_ID_COL,
        eval_mode=True,
        batch_size=int(bs_test),
        kwargs=dict(
            user_feature_cols=user_feature_cols,
            item_feature_dict=item_feature_dict,
        ),
    )

    print("\nResults prediction:")
    scores = model.scorer.predict(eval_dataloader)

    # Add scores to df
    scores = np.asarray(scores)  # (num_impr, MAX_INVIEW)
    lens = df_eval[DEFAULT_INVIEW_ARTICLES_COL].list.len().to_list()
    scores_trimmed = [scores[i, :lens[i]].tolist() for i in range(len(lens))]
    df_pred = add_prediction_scores(df_eval, scores_trimmed)

    # y_true and y_pred as list-of-lists for MetricEvaluator
    y_true = df_pred[DEFAULT_LABELS_COL].to_list()
    y_pred = df_pred["scores"].to_list()

    ''' ===================== PHASE 4: METRICS EVALUATION ====================================== '''
    metrics_evaluator = MetricEvaluator(
        labels=y_true,
        predictions=y_pred,
        metric_functions=[
            AucScore(),
            MrrScore(),
            NdcgScore(k=5),
            NdcgScore(k=10),
        ],
    )

    print("\nMetrics evaluation:")
    metrics = metrics_evaluator.evaluate().evaluations
    print(f"Metrics: {metrics}")

    ''' ===================== PHASE 5: PERMANENT STORAGE ====================================== '''
    # Save metrics and also keep a ranked version like the official predictions file
    write_json_file(metrics, results_dir / "metrics.json")

    # Save ranked_scores parquet in the same format used by the benchmark
    df_pred_ranked = df_pred.with_columns(
        pl.col("scores")
        .map_elements(lambda x: list(rank_predictions_by_score(x)))
        .alias("ranked_scores")
    )
    df_pred_ranked.select(DEFAULT_IMPRESSION_ID_COL, "ranked_scores").write_parquet(
        deliverables_folder / "test_predictions.parquet"
    )

    # Save submission-style files (optional but matches original tooling)
    write_submission_file(
        impression_ids=df_pred_ranked[DEFAULT_IMPRESSION_ID_COL],
        prediction_scores=df_pred_ranked["ranked_scores"],
        path=deliverables_folder / "predictions.txt",
        filename_zip=f"{model}-{seed}-{datasplit}.zip",
    )

    ''' ===================== PHASE 6: MEMORY CLEANUP ============================ '''
    del df, df_train, df_validation, df_eval, df_pred, df_pred_ranked
    del train_dataloader, val_dataloader, eval_dataloader
    del scores, model, hist

    tf.keras.backend.clear_session()
    gc.collect()
