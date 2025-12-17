#%%
import pandas as pd
from pathlib import Path

df = pd.read_excel("Pierce2021_perturbation_diffdevscores.xlsx")

# Calculate average DiffDevScore for each unique sgRNA and sort in decreasing order
avg_scores = df.groupby('sgRNA')['DiffDevScore'].mean().sort_values(ascending=False)
avg_scores

# Print the ranking
print("Gene Ranking by Average DiffDevScore (decreasing order):")
print("=" * 60)
for rank, (sgRNA, score) in enumerate(avg_scores.items(), 1):
    print(f"{rank}. {sgRNA}: {score:.6f}")

# Save ranking to .txt file (just gene names separated by commas)
output_file = "Pierce2021_gene_ranking.txt"
gene_names = ','.join(avg_scores.index)
with open(output_file, 'w') as f:
    f.write(gene_names)
print(f"\nRanking saved to {output_file}")
# %%
