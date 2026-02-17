import polars as pl
import numpy as np
from pathlib import Path

from ebrec.utils._constants import DEFAULT_USER_COL

'''
    In this script I divide the dataframe of the users according to their label and analyze
    each label separately.
'''

def normalized_gini_index(df_col: pl.Series) -> float:
    unique_values = df_col.unique().to_numpy()
    H = len(unique_values)
    total_values = df_col.count()
    counts = [df_col.filter(df_col == val).count() for val in unique_values]
    frequencies = np.array([c / total_values for c in counts])
    squared_frequencies = np.power(frequencies, 2)

    gini_index = 1 - np.sum(squared_frequencies, axis=0)
    n_gini_index = H * gini_index / (H - 1)

    return n_gini_index


def normalized_entropy_index(df_col: pl.Series) -> float:
    unique_values = df_col.unique().to_numpy()
    H = len(unique_values)
    total_values = df_col.count()
    counts = np.array([df_col.filter(df_col == val).count() for val in unique_values])
    frequencies = counts / total_values

    entropy_index = -np.sum(frequencies * np.log2(frequencies))
    n_entropy_index = entropy_index / np.log2(H)

    return n_entropy_index

def mode_rate(df_col: pl.Series) -> float:
    mode = df_col.mode()
    total_values = df_col.count()
    return df_col.filter(df_col == mode).count() / total_values


DF_PATH = Path('./labeled_users_data.parquet').resolve()
STORAGE_PATH = Path('./clusters_data').resolve()

# Dictionary containing the association between feature and feature type
features_dict = {
    'mean_positive_score': "numerical",
    'mean_neutral_score': "numerical",
    'mean_negative_score': "numerical",
    'positive_sentiment_articles_rate': "numerical",
    'neutral_sentiment_articles_rate': "numerical",
    'negative_sentiment_articles_rate': "numerical",
    'favourite_device': "categorical",
    '0': "numerical",
    '1': "numerical",
    '2': "numerical",
    '3': "numerical",
    '4': "numerical",
    '5': "numerical",
    '6': "numerical",
    '7': "numerical",
    '8': "numerical",
    '9': "numerical",
    '10': "numerical",
    '11': "numerical",
    '12': "numerical",
    '13': "numerical",
    '14': "numerical",
    '15': "numerical",
    '16': "numerical",
    '17': "numerical",
    '18': "numerical",
    '19': "numerical",
    '20': "numerical",
    '21': "numerical",
    '22': "numerical",
    '23': "numerical",
    'median_next_read_time': "numerical",
    'premium_propensity': "categorical",
    'reading_engagement': "categorical",
    'is_subscriber': "binary"
}

# Schema of the dataframe containing the statistical indices of the dataset
indices_schema = {
    # INFO
    "feature": pl.String,
    "feature_type": pl.String,

    # Indices of the numerical features
    "mean": pl.Float64,
    "median": pl.Float64,
    "std": pl.Float64,
    "q25": pl.Float64,
    "q75": pl.Float64,
    "skewness_index": pl.Float64,
    "kurtosis_index": pl.Float64,

    # Indices of the categorical/binary features
    "mode": pl.Float64,
    "mode_rate": pl.Float64,
    "n_unique": pl.Int16,
    "normalized_gini_index": pl.Float64,
    "normalized_entropy_index": pl.Float64
}


# Function to create a dataframe containing the indices of the features in a cluster
def summary_creator(df: pl.DataFrame) -> pl.DataFrame:
    rows = []
    for feature in features_dict.keys():
        row = dict()
        row["feature"] = feature
        row["feature_type"] = features_dict[feature]

        expression = pl.col(feature) # For the built-in functions
        series = df[feature] # For the custom functions

        # Computation of the indices for the numerical features
        if features_dict[feature] == "numerical":
            row["mean"] = df.select(expression.mean()).item()
            row["median"] = df.select(expression.median()).item()
            row["std"] = df.select(expression.std()).item()
            row["q25"] = df.select(expression.quantile(0.25)).item()
            row["q75"] = df.select(expression.quantile(0.75)).item()
            row["skewness_index"] = df.select(expression.skew()).item()
            row["kurtosis_index"] = df.select(expression.kurtosis()).item()

        # Computation of the indices for the categorical/binary data
        else:
            row["mode"] = df.select(expression.mode()).item()
            row["mode_rate"] = mode_rate(series)
            row["n_unique"] = df.select(expression.n_unique()).item()
            row["normalized_gini_index"] = normalized_gini_index(series)
            row["normalized_entropy_index"] = normalized_entropy_index(series)

        rows.append(row)

    return pl.DataFrame(rows, schema=indices_schema)


# I retrieve the labels list
labels_list = pl.scan_parquet(DF_PATH).select("label").unique().collect().to_series().sort().to_list()

# I create a dictionary associating to a key a dataframe
clusters = {
    i: pl.scan_parquet(DF_PATH).filter(pl.col("label") == i).drop(["label", DEFAULT_USER_COL]).collect()
    for i in labels_list
}

for k,v in clusters.items():
    # I generate the summary dataframe
    summary_df = summary_creator(v)

    summary_df_directory = STORAGE_PATH / f"cluster_{k}"
    summary_df_directory.mkdir(parents=True, exist_ok=True)
    summary_df_path = summary_df_directory / f"summary_cluster_{k}.parquet"
    summary_df.write_parquet(summary_df_path)
    break # For the moment I generate only the summary of one cluster

global_df = pl.scan_parquet(DF_PATH).drop(["label", DEFAULT_USER_COL]).collect()

# I generate the global summary dataframe
global_summary_df = summary_creator(global_df)

global_summary_df_directory = STORAGE_PATH / "global"
global_summary_df_directory.mkdir(parents=True, exist_ok=True)
global_summary_df_path = global_summary_df_directory / "global_summary.parquet"
global_summary_df.write_parquet(global_summary_df_path)
