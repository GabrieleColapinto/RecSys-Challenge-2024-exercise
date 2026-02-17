import polars as pl

# I generate the dataframe starting from the parquet file
df = pl.read_parquet("./results.parquet")

# I keep only the changing columns because the others don't carry information
df = df[["run id", "used model", "auc", "mrr", "ndcg@5", "ndcg@10"]]

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

''' I RETRIEVE THE RESULTS FOR EACH MODEL '''
winning_combinations = df.sort("overall score", descending=True)

print("Results:")
print(winning_combinations)

''' I RETURN THE WINNING COMBINATION AS THE ONE WITH THE HIGHEST OVERALL SCORE '''
winning_combination = winning_combinations.head(1)

print("\n-----------------------------------------------------------------------------")
print("WINNING COMBINATION:")
for field in winning_combination.columns:
    print(f"\t{field}: {winning_combination[field][0]}")
