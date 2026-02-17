import polars as pl
from pathlib import Path

'''
    This script analyzes the indices of one cluster by comparing them with the
    global indices.
    
    I create a dictionary of discriminating features separating them from the descriptive features.
    
        Descriptive features: Features of the cluster that describe the global dataset because they
        are similar to their global counterpart.
        
        Discriminating features: Features of the cluster that differ from their global counterpart and therefore
        they explain why this cluster is different from the others.
    
    To distinguish the features I use two criteria:
    
        1) Relative difference: |stat_cluster - stat_global| / stat_global > PERCENTAGE_THRESHOLD
        2) Statistical relevance of the change: z = |mean_cluster - mean_global| / std_global > Z_SCORE_THRESHOLD
        
    I use both criteria for the numerical features but only the relative difference for the categorical/binary features
    because, in such case, the z score does not have a semantic meaning.
    
    The features that satisfy the criteria will have a non-null value in the comparison dataframe.
    The feature that do not satisfy the criteria will have a null value in the comparison dataframe.
'''

# Paths
FOLDER_PATH = Path("./clusters_data")
CLUSTER_PATH = FOLDER_PATH / "cluster_0" / "summary_cluster_0.parquet"
GLOBAL_PATH = FOLDER_PATH / "global" / "global_summary.parquet"

# Constants
PERCENTAGE_THRESHOLD = 0.1
Z_SCORE_THRESHOLD = 1.0

# I retrieve the data
df_cluster = pl.read_parquet(CLUSTER_PATH).fill_nan(0)
df_global = pl.read_parquet(GLOBAL_PATH).fill_nan(0)

df_cluster_rows = df_cluster.to_dicts()
df_global_rows = df_global.to_dicts()

discriminating_features = dict()

# I generate the dictionary of relevant features
for i in range(len(df_cluster_rows)):
    row_cluster = df_cluster_rows[i]
    row_global = df_global_rows[i]

    # For the numerical features I use an AND logic
    if row_cluster["feature_type"] == "numerical":
        # I calculate the relative difference of the means
        means_relative_difference = abs(row_cluster["mean"] - row_global["mean"]) / row_global["mean"]
        if means_relative_difference > PERCENTAGE_THRESHOLD:
            # I calculate the statistical relevance of the change
            statistical_relevance = abs(row_cluster["mean"] - row_global["mean"]) / row_global["std"]
            if statistical_relevance > Z_SCORE_THRESHOLD:
                # I memorize the data
                discriminating_features[row_cluster["feature"]] = {
                    "feature_type": row_cluster["feature_type"],
                    "means_relative_difference": means_relative_difference,
                    "statistical_relevance": statistical_relevance
                }
    else:
        # For the categorical features I use a composite logic
        mode_rate_relative_difference = abs(row_cluster["mode_rate"] - row_global["mode_rate"]) / row_global["mode_rate"]
        n_gini_index_relative_difference = abs(row_cluster["normalized_gini_index"] - row_global["normalized_gini_index"]) / row_global["normalized_gini_index"]
        n_entropy_index_relative_difference = abs(row_cluster["normalized_entropy_index"] - row_global["normalized_entropy_index"]) / row_global["normalized_entropy_index"]

        if ((mode_rate_relative_difference > PERCENTAGE_THRESHOLD) or
                ((n_gini_index_relative_difference > PERCENTAGE_THRESHOLD) and
                 (n_entropy_index_relative_difference > PERCENTAGE_THRESHOLD))):
            # I memorize the data
            discriminating_features[row_cluster["feature"]] = {
                "feature_type": row_cluster["feature_type"],
                "mode_rate_relative_difference": mode_rate_relative_difference,
                "n_gini_index_relative_difference": n_gini_index_relative_difference,
                "n_entropy_index_relative_difference": n_entropy_index_relative_difference
            }

print("\nDiscriminating features:")
for k,v in discriminating_features.items():
    print(f"\t{k}: {v}\n")

print("Insert a breakpoint here to look at the dataframes")
