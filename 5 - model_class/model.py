'''
    This file contains a class implementation of the single_run.py scripts.

    The class name and the methods name are compliant to the example provided
    in the official repository of Codabench.

    The class Model has 3 methods: __init__(), fit(X, y) and predict(X).
'''
import tensorflow as tf
import numpy as np
import polars as pl
import os
from pathlib import Path

from ebrec.utils._constants import (
    DEFAULT_HISTORY_ARTICLE_ID_COL,
    DEFAULT_INVIEW_ARTICLES_COL,
    DEFAULT_LABELS_COL,
    DEFAULT_IMPRESSION_ID_COL
)

from ebrec.utils._behaviors import (
    add_prediction_scores,
)

from ebrec.utils._python import (
    write_submission_file,
    rank_predictions_by_score
)

from nrms_docvec_td import NRMSDocVecTd

from new_dataloader import NewNRMSDataLoaderPretransform

os.environ["TOKENIZERS_PARALLELISM"] = "false"

DELIVERABLES_FOLDER = Path("./deliverables")

class Model:
    ''' MANDATORY METHODS '''
    def __init__(
            self,
            hparams,
            item_feature_dict,
            article_mapping,
            artifacts_folder,
            user_feat_cols,
            seed=123,
            bs_train=128,
            bs_test=128,
            epochs=10,
    ):
        # Model parameters
        self.dataloader = NewNRMSDataLoaderPretransform
        self.hparams = hparams
        self.item_feature_dict = item_feature_dict
        self.article_mapping = article_mapping
        self.user_feat_cols = user_feat_cols

        # Folders
        self.artifacts_folder = artifacts_folder

        # Machine learning parameters
        self.bs_train = bs_train
        self.bs_test = bs_test
        self.epochs = epochs

        self.model = NRMSDocVecTd(hparams=self.hparams, seed=42)
        self.model.model.compile(
            optimizer=self.model.model.optimizer,
            loss=self.model.model.loss,
            metrics=["AUC"],
        )

        self.seed = seed
        tf.random.set_seed(self.seed)

    def fit(self, X, y=None): # In this case X is the training set and y is the evaluation set
        # ------------------ Callbacks -----------------------------------
        weights_dir = self.artifacts_folder / "weights"
        weights_dir.parent.mkdir(parents=True, exist_ok=True)

        log_dir = self.artifacts_folder / "runs"
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

        train_dataloader = self.dataloader(
            behaviors=X,
            article_dict=self.article_mapping,
            unknown_representation="zeros",
            history_column=DEFAULT_HISTORY_ARTICLE_ID_COL,
            eval_mode=False,
            batch_size=int(self.bs_train),
            kwargs=dict(
                user_feature_cols=self.user_feat_cols,
                item_feature_dict=self.item_feature_dict,
            ),
        )
        val_dataloader = self.dataloader(
            behaviors=y,
            article_dict=self.article_mapping,
            unknown_representation="zeros",
            history_column=DEFAULT_HISTORY_ARTICLE_ID_COL,
            eval_mode=False,
            batch_size=int(self.bs_train),
            kwargs=dict(
                user_feature_cols=self.user_feat_cols,
                item_feature_dict=self.item_feature_dict,
            ),
        )

        print("\nModel training:")
        self.model.model.fit(
            train_dataloader,
            validation_data=val_dataloader,
            epochs=int(self.epochs),
            callbacks=callbacks,
        )

        # Load best weights (as original)
        self.model.model.load_weights(str(weights_dir))

    def predict(self, X): # X is the training set
        eval_dataloader = self.dataloader(
            behaviors=X,
            article_dict=self.article_mapping,
            unknown_representation="zeros",
            history_column=DEFAULT_HISTORY_ARTICLE_ID_COL,
            eval_mode=True,
            batch_size=int(self.bs_test),
            kwargs=dict(
                user_feature_cols=self.user_feat_cols,
                item_feature_dict=self.item_feature_dict,
            ),
        )

        print("\nResults prediction:")
        scores = self.model.scorer.predict(eval_dataloader)

        # Add scores to df
        scores = np.asarray(scores)  # (num_impr, MAX_INVIEW)
        lens = X[DEFAULT_INVIEW_ARTICLES_COL].list.len().to_list()
        scores_trimmed = [scores[i, :lens[i]].tolist() for i in range(len(lens))]
        df_pred = add_prediction_scores(X, scores_trimmed)

        y_true = df_pred[DEFAULT_LABELS_COL].to_list()
        y_pred = df_pred["scores"].to_list()

        ''' PERMANENT RESULT STORAGE '''
        # Save ranked_scores parquet in the same format used by the benchmark
        df_pred_ranked = df_pred.with_columns(
            pl.col("scores")
            .map_elements(lambda x: list(rank_predictions_by_score(x)))
            .alias("ranked_scores")
        )
        df_pred_ranked.select(DEFAULT_IMPRESSION_ID_COL, "ranked_scores").write_parquet(
            DELIVERABLES_FOLDER / "test_predictions.parquet"
        )

        # Save submission-style files (optional but matches original tooling)
        write_submission_file(
            impression_ids=df_pred_ranked[DEFAULT_IMPRESSION_ID_COL],
            prediction_scores=df_pred_ranked["ranked_scores"],
            path=DELIVERABLES_FOLDER / "predictions.txt",
            filename_zip=f"{self}-{self.seed}-prediction_results.zip",
        )

        return y_true, y_pred

    ''' ADDITIONAL METHOD '''
    def __str__(self):
        return "submission_model"
