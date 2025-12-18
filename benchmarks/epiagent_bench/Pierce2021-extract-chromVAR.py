#%%
import pandas as pd
from pathlib import Path

df = pd.read_excel("./Pierce2021_supp_mat_perturbation_diffdevscores.xlsx")

# Calculate average DiffDevScore for each unique sgRNA and sort in decreasing order
avg_scores = df.groupby('sgRNA')['DiffDevScore'].mean().sort_values(ascending=False)

# Print the ranking
print("Gene Ranking by Average DiffDevScore (decreasing order):")
print("=" * 60)
for rank, (sgRNA, score) in enumerate(avg_scores.items(), 1):
    print(f"{rank}. {sgRNA}: {score:.6f}")

output_file = "Pierce2021_gene_ranking.csv"
ranking_df = pd.DataFrame({
    'gene': avg_scores.index,
    'diffdevscore': avg_scores.values
})
ranking_df.to_csv(output_file, index=False)
print(f"\nRanking saved to {output_file}")
# %%
