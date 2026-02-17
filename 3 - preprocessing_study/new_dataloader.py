from dataclasses import dataclass, field
import tensorflow as tf
import polars as pl
import numpy as np

from ebrec.utils._articles_behaviors import map_list_article_id_to_value
from ebrec.utils._python import (
    repeat_by_list_values_from_matrix,
    create_lookup_objects,
    create_lookup_dict
)

from ebrec.utils._constants import (
    DEFAULT_INVIEW_ARTICLES_COL,
    DEFAULT_LABELS_COL,
    DEFAULT_USER_COL,
)

from ebrec.utils._constants import DEFAULT_ARTICLE_ID_COL


def create_article_id_to_value_mapping(
    df: pl.DataFrame,
    value_col: str,
    article_col: str = DEFAULT_ARTICLE_ID_COL,
):
    return create_lookup_dict(
        df.select(article_col, value_col), key=article_col, value=value_col
    )

""" Pad list-of-lists to same length using pad_value. Returns padded lists and max length. """
def _pad_2d_int_lists_to_len(lists, pad_value: int, target_len: int) -> list[list[int]]:
    # normalize -> list[list[int]] of ints (come avevamo già fatto)
    normalized = []
    for row in lists:
        if row is None:
            row = []
        if not isinstance(row, (list, tuple)):
            row = [row]

        out_row = []
        for v in row:
            if isinstance(v, (list, tuple)) and len(v) == 1:
                v = v[0]
            try:
                out_row.append(int(v))
            except Exception:
                out_row.append(pad_value)

        normalized.append(out_row)

    # pad/truncate to target_len
    padded = [r[:target_len] + [pad_value] * max(0, target_len - len(r)) for r in normalized]
    return padded


@dataclass
class NewsrecDataLoader(tf.keras.utils.Sequence):
    """
    A DataLoader for news recommendation.
    """

    behaviors: pl.DataFrame
    history_column: str
    article_dict: dict[int, any]
    unknown_representation: str
    eval_mode: bool = False
    batch_size: int = 32
    inview_col: str = DEFAULT_INVIEW_ARTICLES_COL
    labels_col: str = DEFAULT_LABELS_COL
    user_col: str = DEFAULT_USER_COL
    kwargs: dict | None = None

    def __post_init__(self):
        """
        Post-initialization method. Loads the data and sets additional attributes.
        """
        self.lookup_article_index, self.lookup_article_matrix = create_lookup_objects(
            self.article_dict, unknown_representation=self.unknown_representation
        )
        self.unknown_index = [0]
        self.X, self.y = self.load_data()
        if self.kwargs is not None:
            self.set_kwargs(self.kwargs)

        # Consider the item features
        if self.item_feature_dict is not None:
            self.lookup_itemfeat_index, self.lookup_itemfeat_matrix = create_lookup_objects(
                self.item_feature_dict, unknown_representation="zeros"
            )
        else:
            self.lookup_itemfeat_index = None
            self.lookup_itemfeat_matrix = None

    def __len__(self) -> int:
        return int(np.ceil(len(self.X) / float(self.batch_size)))

    def __getitem__(self):
        raise ValueError("Function '__getitem__' needs to be implemented.")

    def load_data(self) -> tuple[pl.DataFrame, pl.DataFrame]:
        X = self.behaviors.drop(self.labels_col).with_columns(
            pl.col(self.inview_col).list.len().alias("n_samples")
        )
        y = self.behaviors[self.labels_col]
        return X, y

    def set_kwargs(self, kwargs: dict):
        for key, value in kwargs.items():
            setattr(self, key, value)

@dataclass
class NewNRMSDataLoaderPretransform(NewsrecDataLoader):
    """
    In the __post_init__ pre-transform the entire DataFrame. This is useful for
    when data can fit in memory, as it will be much faster ones training.
    Note, it might not be as scaleable.
    """

    # Additional fields
    user_feature_cols: list[str] = field(default_factory=list)
    item_feature_dict: dict[int, np.ndarray] | None = None

    def __post_init__(self):
        super().__post_init__()
        self.X = self.X.with_columns([
            pl.col(self.history_column).alias("history_doc_idx"),
            pl.col(self.inview_col).alias("inview_doc_idx"),
        ])

        # compute global max candidates (only needed for eval_mode listwise)
        self.max_inview = int(
            self.X[self.inview_col].list.len().max()
        )

        # mappa docvec su history_doc_idx / inview_doc_idx
        self.X = self.X.pipe(
            map_list_article_id_to_value,
            behaviors_column="history_doc_idx",
            mapping=self.lookup_article_index,
            fill_nulls=self.unknown_index,
            drop_nulls=False,
        ).pipe(
            map_list_article_id_to_value,
            behaviors_column="inview_doc_idx",
            mapping=self.lookup_article_index,
            fill_nulls=self.unknown_index,
            drop_nulls=False,
        )

        # se item features attive, crea colonne e mappa
        if self.lookup_itemfeat_index is not None:
            self.X = self.X.with_columns([
                pl.col(self.history_column).alias("history_item_idx"),
                pl.col(self.inview_col).alias("inview_item_idx"),
            ]).pipe(
                map_list_article_id_to_value,
                behaviors_column="history_item_idx",
                mapping=self.lookup_itemfeat_index,
                fill_nulls=self.unknown_index,
                drop_nulls=False,
            ).pipe(
                map_list_article_id_to_value,
                behaviors_column="inview_item_idx",
                mapping=self.lookup_itemfeat_index,
                fill_nulls=self.unknown_index,
                drop_nulls=False,
            )

    def __getitem__(self, idx) -> tuple[tuple[np.ndarray], np.ndarray]:
        """
        his_input_title:    (samples, history_size, document_dimension)
        pred_input_title:   (samples, npratio, document_dimension)
        batch_y:            (samples, npratio)
        """
        batch_X = self.X[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size : (idx + 1) * self.batch_size]
        # =>
        if self.eval_mode:
            batch_y = np.zeros((len(batch_X), 1), dtype=np.float32)

            his_input_title = self.lookup_article_matrix[
                batch_X["history_doc_idx"].to_list()
            ]

            inview_lists = batch_X["inview_doc_idx"].to_list()
            inview_lists_padded = _pad_2d_int_lists_to_len(
                inview_lists,
                pad_value=int(self.unknown_index[0]),
                target_len=int(self.max_inview),
            )

            idx = np.asarray(inview_lists_padded, dtype=np.int64)
            pred_input_title = self.lookup_article_matrix[idx]

            # squeeze safe (se lookup_article_matrix ha shape (n,1,D))
            if pred_input_title.ndim == 4:
                pred_input_title = np.squeeze(pred_input_title, axis=2)

        else:
            batch_y = np.array(batch_y.to_list())
            his_input_title = self.lookup_article_matrix[batch_X["history_doc_idx"].to_list()]
            pred_input_title = self.lookup_article_matrix[batch_X["inview_doc_idx"].to_list()]
            pred_input_title = np.squeeze(pred_input_title, axis=2)

        # Squeeze safe: sometimes lookup returns an extra singleton dim
        if his_input_title.ndim == 4:
            his_input_title = np.squeeze(his_input_title, axis=2)
        if pred_input_title.ndim == 4:
            pred_input_title = np.squeeze(pred_input_title, axis=2)

        # Only squeeze (B,1,D) in TRAIN mode (pointwise) — in eval we want (B,N,D)
        if (not self.eval_mode) and pred_input_title.ndim == 3 and pred_input_title.shape[1] == 1:
            pred_input_title = np.squeeze(pred_input_title, axis=1)

        # Consider the additional features
        if self.user_feature_cols:
            user_features = batch_X.select(self.user_feature_cols).to_numpy().astype(np.float32)
        else:
            user_features = np.zeros((len(batch_X), 0), dtype=np.float32)

        if self.lookup_itemfeat_matrix is not None:
            # With item features
            if self.eval_mode:
                his_item_features = self.lookup_itemfeat_matrix[
                    batch_X["history_item_idx"].to_list()
                ]

                inview_item_lists = batch_X["inview_item_idx"].to_list()
                inview_item_lists_padded = _pad_2d_int_lists_to_len(
                    inview_item_lists,
                    pad_value=int(self.unknown_index[0]),
                    target_len=int(self.max_inview),
                )

                idx_item = np.asarray(inview_item_lists_padded, dtype=np.int64)
                pred_item_features = self.lookup_itemfeat_matrix[idx_item]

                # squeeze safe (se lookup_article_matrix ha shape (n,1,D))
                if pred_item_features.ndim == 4:
                    pred_item_features = np.squeeze(pred_item_features, axis=2)

            else:
                his_item_features = self.lookup_itemfeat_matrix[
                    batch_X["history_item_idx"].to_list()
                ]
                pred_item_features = self.lookup_itemfeat_matrix[
                    batch_X["inview_item_idx"].to_list()
                ]
                pred_item_features = np.squeeze(pred_item_features, axis=2)

            if his_item_features.ndim == 4:
                his_item_features = np.squeeze(his_item_features, axis=2)

        else:
            # Without item features
            # Usa shapes coerenti con docvec: eval -> (B,H,D) e (B,N,D)
            if self.eval_mode:
                B = len(batch_X)
                H = his_input_title.shape[1]
                N = pred_input_title.shape[1]
                his_item_features = np.zeros((B, H, 0), dtype=np.float32)
                pred_item_features = np.zeros((B, N, 0), dtype=np.float32)
            else:
                B = len(batch_X)
                H = his_input_title.shape[1]  # in train dovrebbe essere history_size
                # in train pointwise i candidati tipicamente sono (B, 1, D) dopo squeeze
                pred_item_features = np.zeros((B, 1, 0), dtype=np.float32)
                his_item_features = np.zeros((B, H, 0), dtype=np.float32)

        # Final checks
        assert his_input_title.shape[0] == batch_y.shape[0]
        assert pred_input_title.shape[0] == batch_y.shape[0]
        assert user_features.shape[0] == batch_y.shape[0]
        assert his_item_features.shape[0] == batch_y.shape[0]
        assert pred_item_features.shape[0] == batch_y.shape[0]

        return (his_input_title, pred_input_title, user_features, his_item_features, pred_item_features), batch_y

