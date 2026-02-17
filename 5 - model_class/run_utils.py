'''
    This file contains parameters and functions to be used in the
    run.py script. I am making another script for this code to
    make the file run.py more readable, understandable and straight to the point.
'''


from pathlib import Path
import polars as pl
import numpy as np

from ebrec.utils._constants import *

from ebrec.utils._behaviors import (
    create_binary_labels_column,
    sampling_strategy_wu2019,
    ebnerd_from_path,
)

from ebrec.models.newsrec.model_config import hparams_nrms_docvec

# ===================== PATHS ===============================
NEW_DATA_PATH = Path("../new_data").resolve()
CONFIG_PATH = Path("./config").resolve()

TRAINING_DATA_PATH = Path("../data/ebnerd_demo").resolve()
TESTING_DATA_PATH = Path("../data/ebnerd_testset").resolve()

DOC_VEC_PATH = Path("./contrastive_vector.parquet").resolve()

TRAIN_PATH = TRAINING_DATA_PATH.joinpath("train").resolve()
VALIDATION_PATH = TRAINING_DATA_PATH.joinpath("validation").resolve()
TEST_PATH = TESTING_DATA_PATH.joinpath("test").resolve()

ARTIFACTS_FOLDER = Path("./artifacts").resolve()

# Constant parameters taken from the config.json file and from the single_run.py file
DOCVEC_COL = "contrastive_vector"
RANDOM_SEED = 123
BS_TRAIN = 128
BS_TEST = 128
BATCH_SIZE_TEST_WO_B = 128
BATCH_SIZE_TEST_W_B = 8
ATTENTION_HIDDEN_DIM = 200
TITLE_SIZE = 768
LEARNING_RATE = 0.0001
NEWSENCODER_L2_REGULARIZATION = 0.0001
NEWSENCODER_UNITS_PER_LAYER = [512, 512, 512]
DROPOUT = 0.2
OPTIMIZER = "adam"
LOSS = "cross_entropy_loss"
EPOCHS = 1
HISTORY_SIZE = 20
NPRATIO = 5
HEAD_NUM = 16
HEAD_DIM = 16
NRMS_LOADER = "NRMSDataLoaderPretransform"
TRAIN_FRACTION = 0.001

# =========== PREPROCESSING RELATED PARAMETERS OF THE MODEL ==============
ITEM_FEAT_COLS = ["premium"]
USER_FEAT_COLS = ["premium_propensity", "reading_engagement"]
USER_FEAT_DIM = 2  # Premium propensity & Reading engagement
ITEM_FEAT_DIM = 1  # Premium propensity

# Default columns
COLUMNS = [
        DEFAULT_IMPRESSION_TIMESTAMP_COL, # behaviors["impression_time"]
        DEFAULT_HISTORY_ARTICLE_ID_COL, # behaviors["article_id_fixed"]
        DEFAULT_INVIEW_ARTICLES_COL, # behaviors["article_ids_inview"]
        DEFAULT_CLICKED_ARTICLES_COL, # behaviors["article_ids_clicked"]
        DEFAULT_IMPRESSION_ID_COL, # behaviors["impression_id"]
        DEFAULT_USER_COL, # behaviors["user_id"]
]

''' HYPERPARAMETERS GENERATION '''

def make_hparams():
    # Hyperparameter object initialization
    hparams = hparams_nrms_docvec

    # Values assignment
    hparams.user_feat_dim = USER_FEAT_DIM
    hparams.item_feat_dim = ITEM_FEAT_DIM
    hparams.history_size = HISTORY_SIZE
    hparams.head_num = HEAD_NUM
    hparams.head_dim = HEAD_DIM
    hparams.attention_hidden_dim = ATTENTION_HIDDEN_DIM
    hparams.newsencoder_units_per_layer = NEWSENCODER_UNITS_PER_LAYER
    hparams.optimizer = OPTIMIZER
    hparams.loss = LOSS
    hparams.dropout = DROPOUT
    hparams.learning_rate = LEARNING_RATE
    hparams.newsencoder_l2_regularization = NEWSENCODER_L2_REGULARIZATION

    return hparams


def get_train_and_evaluation_df():
    # TRAINING SET
    df_train = (
        pl.concat(
            [
                ebnerd_from_path(
                    path=TRAIN_PATH,
                    history_size=HISTORY_SIZE,
                    padding=0,
                ),
                ebnerd_from_path(
                    path=VALIDATION_PATH,
                    history_size=HISTORY_SIZE,
                    padding=0,
                ),
            ]
        )
        .sample(fraction=1, shuffle=True, seed=RANDOM_SEED)  # I use the whole training set
        .select(COLUMNS)
        .pipe(
            sampling_strategy_wu2019,
            npratio=NPRATIO,
            shuffle=True,
            with_replacement=True,
            seed=RANDOM_SEED,
        )
        .pipe(create_binary_labels_column)
    )

    # VALIDATION SET FOR TRAINING (same sampling strategy as training)
    df_val_fit = (
        ebnerd_from_path(
            path=VALIDATION_PATH,
            history_size=HISTORY_SIZE,
            padding=0,
        )
        .select(COLUMNS)
        .pipe(
            sampling_strategy_wu2019,
            npratio=NPRATIO,
            shuffle=True,
            with_replacement=True,
            seed=RANDOM_SEED,
        )
        .pipe(create_binary_labels_column)
    )

    ''' BEGINNING OF ADDITIONAL PREPROCESSING OPERATIONS ON THE USERS '''

    ''' Premiumness categorization data '''
    # I retrieve the association between user and premium propensity
    df_premium_propensity = pl.read_parquet(NEW_DATA_PATH / "premium_propensity.parquet")

    # I add the premium propensity to the main dataframe
    df_train = df_train.join(
        other=df_premium_propensity,
        on=DEFAULT_USER_COL,
        how="inner"
    )

    # I add the premium propensity to the validation dataframe for the training
    df_val_fit = df_val_fit.join(
        other=df_premium_propensity,
        on=DEFAULT_USER_COL,
        how="inner"
    )

    ''' Reading engagement data '''
    # I retrieve the association between user and reading engagement
    df_reading_engagement = pl.read_parquet(NEW_DATA_PATH / "reading_engagement.parquet")

    # I add the reading engagement to the main dataframe
    df_train = df_train.join(
        other=df_reading_engagement,
        on=DEFAULT_USER_COL,
        how="inner"
    )

    # I add the reading engagement to the validation dataframe for the training
    df_val_fit = df_val_fit.join(
        other=df_reading_engagement,
        on=DEFAULT_USER_COL,
        how="inner"
    )

    return df_train, df_val_fit


def get_test_df():
    # MAIN DATAFRAME
    df_test = ((
        pl.concat(
            [
                ebnerd_from_path(
                    path=TEST_PATH,
                    history_size=HISTORY_SIZE,
                    padding=0,
                )
            ]
        )
        .sample(fraction=TRAIN_FRACTION, shuffle=True, seed=RANDOM_SEED)
    ).with_columns(
        pl.lit([]).cast(pl.List(pl.Int32)).alias(DEFAULT_CLICKED_ARTICLES_COL)
    )
    .select(COLUMNS)
    .with_columns(
        # I add the dummy premium propensity column
        pl.lit(0).cast(pl.Int32).alias("premium_propensity")
    )
    .pipe(create_binary_labels_column))

    # I add the reading engagement data
    df_reading_engagement_testset = pl.read_parquet(NEW_DATA_PATH / "reading_engagement_TESTSET.parquet")

    df_test = df_test.join(
        other=df_reading_engagement_testset,
        on=DEFAULT_USER_COL,
        how="inner"
    )

    return df_test


def get_articles_df():
    df_articles = pl.read_parquet(DOC_VEC_PATH)

    ''' Premiumness integration '''
    df_premiumness = pl.read_parquet(NEW_DATA_PATH / "premium_articles.parquet")
    df_articles = df_articles.join(
        other=df_premiumness,
        on=DEFAULT_ARTICLE_ID_COL,
        how="inner"
    )

    return df_articles


def get_item_feature_dict(df_articles: pl.DataFrame):
    # I create a dictionary to associate article IDs to features
    df_item = (
        df_articles
        .select([DEFAULT_ARTICLE_ID_COL, "premium"])
        .with_columns(
            pl.col("premium").cast(pl.Float32).fill_null(0.0)
        )
    )

    item_feature_dict = {
        int(article_id): np.array([premium], dtype=np.float32)
        for article_id, premium in df_item.iter_rows()
    }

    return item_feature_dict
