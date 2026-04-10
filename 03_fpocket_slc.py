import pandas as pd
import numpy as np
from scipy import stats

df_fp = pd.read_csv('fpocket_slc_results.csv')
df_corr = pd.read_csv('slc_correlation_annotated2.csv')

df = pd.merge(df_corr, df_fp[['UniProt','drug_score','pocket_volume','n_pockets']], on='UniProt')
print(f"Merged: {len(df)} targets")

# === 1. drug_score vs Pearson_r の相関 ===
r, p = stats.pearsonr(df['drug_score'], df['pearson_r'])
print(f"\n=== drug_score vs Pearson_r ===")
print(f"Pearson r = {r:.3f}, P = {p:.4f}")

# === 2. drug_scoreのGPCRとの分布比較 ===
print(f"\n=== drug_score distribution ===")
print(f"SLC:  mean={df['drug_score'].mean():.3f}, median={df['drug_score'].median():.3f}, "
      f"std={df['drug_score'].std():.3f}")
print(f"GPCR: mean=0.804, median=0.875, std=?(ref)")

# === 3. drug_scoreで二値化してPearson_rと比較 ===
threshold = df['drug_score'].median()
df['high_drug'] = (df['drug_score'] >= threshold).astype(int)
high = df[df['high_drug']==1]['pearson_r']
low  = df[df['high_drug']==0]['pearson_r']
u, p2 = stats.mannwhitneyu(high, low, alternative='two-sided')
print(f"\n=== High vs Low drug_score (median split={threshold:.3f}) ===")
print(f"High: n={len(high)}, mean r={high.mean():.3f}, median r={high.median():.3f}")
print(f"Low:  n={len(low)},  mean r={low.mean():.3f}, median r={low.median():.3f}")
print(f"Mann-Whitney U P = {p2:.4f}")

# === 4. 最適閾値探索 ===
print(f"\n=== drug_score threshold optimization ===")
best_p = 1.0
best_t = 0
for t in np.arange(0.3, 0.95, 0.05):
    h = df[df['drug_score'] >= t]['pearson_r']
    l = df[df['drug_score'] < t]['pearson_r']
    if len(h) > 5 and len(l) > 5:
        _, p_t = stats.mannwhitneyu(h, l, alternative='two-sided')
        print(f"  threshold={t:.2f}: n_high={len(h):2d}, mean_high={h.mean():.3f}, "
              f"n_low={len(l):2d}, mean_low={l.mean():.3f}, P={p_t:.4f}")
        if p_t < best_p:
            best_p = p_t
            best_t = t
print(f"\nBest threshold: {best_t:.2f}, P = {best_p:.4f}")

# === 5. Clear Orthosteric Pocket vs drug_score ===
print(f"\n=== Clear Orthosteric Pocket vs drug_score ===")
clear = df[df['pocket_type']=='Clear_orthosteric']['drug_score']
other = df[df['pocket_type']=='Other']['drug_score']
u3, p3 = stats.mannwhitneyu(clear, other, alternative='two-sided')
print(f"Clear: n={len(clear)}, mean={clear.mean():.3f}, median={clear.median():.3f}")
print(f"Other: n={len(other)}, mean={other.mean():.3f}, median={other.median():.3f}")
print(f"Mann-Whitney U P = {p3:.4f}")

# === 6. TM12 vs drug_score ===
print(f"\n=== TM12 vs drug_score ===")
tm12 = df[df['tm12']=='TM12']['drug_score']
non_tm12 = df[df['tm12']=='non-TM12']['drug_score']
u4, p4 = stats.mannwhitneyu(tm12, non_tm12, alternative='two-sided')
print(f"TM12:     n={len(tm12)}, mean={tm12.mean():.3f}, median={tm12.median():.3f}")
print(f"non-TM12: n={len(non_tm12)}, mean={non_tm12.mean():.3f}, median={non_tm12.median():.3f}")
print(f"Mann-Whitney U P = {p4:.4f}")

# === 7. Fold別drug_score ===
print(f"\n=== drug_score by Fold ===")
for fold, group in df.groupby('fold'):
    print(f"{fold:6s}: n={len(group):2d}, mean={group['drug_score'].mean():.3f}, "
          f"median={group['drug_score'].median():.3f}, "
          f"mean_r={group['pearson_r'].mean():.3f}")

# === 8. Variance explained ===
print(f"\n=== Variance explained (eta²) ===")
grand_mean = df['pearson_r'].mean()
ss_total = ((df['pearson_r'] - grand_mean)**2).sum()
ss_pocket = sum(
    len(df[df['pocket_type']==v]) * (df[df['pocket_type']==v]['pearson_r'].mean() - grand_mean)**2
    for v in df['pocket_type'].unique()
)
ss_tm12 = sum(
    len(df[df['tm12']==v]) * (df[df['tm12']==v]['pearson_r'].mean() - grand_mean)**2
    for v in df['tm12'].unique()
)
r_drug, _ = stats.pearsonr(df['drug_score'], df['pearson_r'])
print(f"Clear Orthosteric Pocket eta²: {ss_pocket/ss_total:.4f} ({ss_pocket/ss_total*100:.1f}%)")
print(f"TM12 eta²:                     {ss_tm12/ss_total:.4f} ({ss_tm12/ss_total*100:.1f}%)")
print(f"drug_score r²:                 {r_drug**2:.4f} ({r_drug**2*100:.1f}%)")

df.to_csv('fpocket_slc_integrated.csv', index=False)
print("\nSaved: fpocket_slc_integrated.csv")
