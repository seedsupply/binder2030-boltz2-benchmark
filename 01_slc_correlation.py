import pandas as pd
import numpy as np
from scipy import stats

df = pd.read_csv('slc_merged_results.csv')

results = []
for uniprot, group in df.groupby('UniProt'):
    if len(group) < 3:
        continue
    r, p = stats.pearsonr(group['pKd'], group['boltz2_pKd'])
    results.append({
        'UniProt': uniprot,
        'Target': group['Target'].iloc[0],
        'n': len(group),
        'pearson_r': round(r, 3),
        'p_value': round(p, 4)
    })

df_corr = pd.DataFrame(results).sort_values('pearson_r', ascending=False)

# サマリー統計
print("=== SLC Boltz-2 Correlation Summary ===")
print(f"Total targets: {len(df_corr)}")
print(f"Total pairs: {len(df)}")
print(f"Mean Pearson r: {df_corr['pearson_r'].mean():.3f}")
print(f"Median Pearson r: {df_corr['pearson_r'].median():.3f}")
print(f"Range: {df_corr['pearson_r'].min():.3f} to {df_corr['pearson_r'].max():.3f}")
print()

# カテゴリ分類
print("=== Performance Categories ===")
print(f"Strong (r >= 0.5):      {(df_corr['pearson_r'] >= 0.5).sum()} ({(df_corr['pearson_r'] >= 0.5).mean()*100:.1f}%)")
print(f"Moderate (0.2-0.5):     {((df_corr['pearson_r'] >= 0.2) & (df_corr['pearson_r'] < 0.5)).sum()} ({((df_corr['pearson_r'] >= 0.2) & (df_corr['pearson_r'] < 0.5)).mean()*100:.1f}%)")
print(f"Weak (0-0.2):           {((df_corr['pearson_r'] >= 0) & (df_corr['pearson_r'] < 0.2)).sum()} ({((df_corr['pearson_r'] >= 0) & (df_corr['pearson_r'] < 0.2)).mean()*100:.1f}%)")
print(f"Inverse (r < 0):        {(df_corr['pearson_r'] < 0).sum()} ({(df_corr['pearson_r'] < 0).mean()*100:.1f}%)")
print()

print("=== Top 10 Targets ===")
print(df_corr.head(10).to_string(index=False))
print()
print("=== Bottom 10 Targets ===")
print(df_corr.tail(10).to_string(index=False))

df_corr.to_csv('slc_correlation_results.csv', index=False)
print("\nSaved: slc_correlation_results.csv")
