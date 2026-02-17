import polars as pl

# I generate the dataframe starting from the parquet file
df = pl.read_parquet("./results.parquet")

'''
    I encode the loss because it is a string and I need it to be 1 or 2.
    cross_entropy_loss = 1
    log_loss = 2
'''
df = df.with_columns(
    pl.when(pl.col("loss") == "cross_entropy_loss")
      .then(1)
      .otherwise(2)
      .alias("encoded loss")
)

# I keep only the changing columns because the others don't carry information
df = df[["run id", "encoded loss", "npratio", "used model", "auc", "mrr", "ndcg@5", "ndcg@10"]]

''' I CALCULATE THE CORRELATION BETWEEN THE TARGET METRICS AND THE ATTRIBUTES FOR ALL THE MODELS '''
# AUC correlations
auc_correlations = df.select(
    pl.corr("auc", "encoded loss").alias("encoded loss"),
    pl.corr("auc", "npratio").alias("npratio"),
)

print("AUC CORRELATIONS:")
auc_correlations_list = sorted(
    [(field, auc_correlations[field][0]) for field in auc_correlations.columns],
    key=lambda x: x[1], reverse=True
)
for i in range(len(auc_correlations_list)):
    print(f"\t{i+1} {auc_correlations_list[i][0]}: {auc_correlations_list[i][1]}")

# MRR correlations
mrr_correlations = df.select(
    pl.corr("mrr", "encoded loss").alias("encoded loss"),
    pl.corr("mrr", "npratio").alias("npratio"),
)

print("\nMRR CORRELATIONS:")
mrr_correlations_list = sorted(
    [(field, mrr_correlations[field][0]) for field in mrr_correlations.columns],
    key=lambda x: x[1], reverse=True
)
for i in range(len(mrr_correlations_list)):
    print(f"\t{i+1} {mrr_correlations_list[i][0]}: {mrr_correlations_list[i][1]}")

# NDCG@5 correlations
ndcg5_correlations = df.select(
    pl.corr("ndcg@5", "encoded loss").alias("encoded loss"),
    pl.corr("ndcg@5", "npratio").alias("npratio"),
)

print("\nNDCG@5 CORRELATIONS:")
ndcg5_correlations_list = sorted(
    [(field, ndcg5_correlations[field][0]) for field in ndcg5_correlations.columns],
    key=lambda x: x[1], reverse=True
)
for i in range(len(ndcg5_correlations_list)):
    print(f"\t{i+1} {ndcg5_correlations_list[i][0]}: {ndcg5_correlations_list[i][1]}")

# NDCG@10 correlations
ndcg10_correlations = df.select(
    pl.corr("ndcg@10", "encoded loss").alias("encoded loss"),
    pl.corr("ndcg@10", "npratio").alias("npratio"),
)

print("\nNDCG@10 CORRELATIONS:")
ndcg10_correlations_list = sorted(
    [(field, ndcg10_correlations[field][0]) for field in ndcg10_correlations.columns],
    key=lambda x: x[1], reverse=True
)
for i in range(len(ndcg10_correlations_list)):
    print(f"\t{i+1} {ndcg10_correlations_list[i][0]}: {ndcg10_correlations_list[i][1]}")

''' I GENERATE WEIGHTS FOR THE METRICS TO COMPUTE AN OVERALL SCORE '''
# I keep the sum of the weights to 1 so that the overall score is in the range [0, 1]
weights = {
    "w_auc": 0.5,
    "w_mrr": 0.3,
    "w_dcg@5": 0.1,
    "w_ndcg@10": 0.1
}

df = df.with_columns(
    (
        weights["w_auc"] * pl.col("auc") +
        weights["w_mrr"] * pl.col("mrr") +
        weights["w_dcg@5"] * pl.col("ndcg@5") +
        weights["w_ndcg@10"] * pl.col("ndcg@10")
    ).alias("overall score")
)

''' I RETRIEVE THE WINNING COMBINATIONS FOR EACH MODEL '''
winning_combinations = (df.select(["used model", "encoded loss", "npratio", "auc", "mrr", "ndcg@5", "ndcg@10", "overall score"])
                        .group_by("used model")
                        .map_groups(lambda g: g.filter(pl.col("overall score") == pl.col("overall score").max()))
                        .sort("overall score", descending=True))

print("Winning combinations:")
print(winning_combinations)

''' I RETURN THE WINNING COMBINATION AS THE ONE WITH THE HIGHEST OVERALL SCORE '''
winning_combination = winning_combinations.head(1)

print("\n-----------------------------------------------------------------------------")
print("WINNING COMBINATION:")
for field in winning_combination.columns:
    print(f"\t{field}: {winning_combination[field][0]}")
