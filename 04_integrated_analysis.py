import pandas as pd
import numpy as np
from scipy import stats

# データ読み込み
df_gpcr = pd.read_csv('fpocket_gpcr_integrated.csv')
df_slc = pd.read_csv('fpocket_slc_integrated.csv')

# カラム名を統一
df_gpcr = df_gpcr.rename(columns={'Pearson_r': 'pearson_r'})

# 必要カラムを選択してprotein_classを追加
gpcr_cols = ['Target','UniProt','pearson_r','drug_score','pocket_volume',
             'n_pockets','has_CWxP','GPCR_class']
slc_cols  = ['Target','UniProt','pearson_r','drug_score','pocket_volume',
             'n_pockets','fold','tm12','pocket_type']

df_g = df_gpcr[gpcr_cols].copy()
df_g['protein_class'] = 'GPCR'
df_g['structural_predictor'] = df_g['has_CWxP'].apply(lambda x: 'Positive' if x==1 else 'Negative')

df_s = df_slc[slc_cols].copy()
df_s['protein_class'] = 'SLC'
df_s['structural_predictor'] = df_s['pocket_type'].apply(
    lambda x: 'Positive' if x=='Clear_orthosteric' else 'Negative')

# 統合DataFrame
df = pd.concat([
    df_g[['Target','UniProt','pearson_r','drug_score','pocket_volume',
          'n_pockets','protein_class','structural_predictor']],
    df_s[['Target','UniProt','pearson_r','drug_score','pocket_volume',
          'n_pockets','protein_class','structural_predictor']]
], ignore_index=True)

print(f"Integrated: {len(df)} targets (GPCR={len(df_g)}, SLC={len(df_s)})")

# === 1. 全体サマリー ===
print(f"\n=== Overall Summary ===")
for pc, group in df.groupby('protein_class'):
    print(f"{pc}: n={len(group)}, mean r={group['pearson_r'].mean():.3f}, "
          f"median r={group['pearson_r'].median():.3f}, "
          f"drug_score mean={group['drug_score'].mean():.3f}")

# === 2. 構造的予測因子の効果（GPCR vs SLC） ===
print(f"\n=== Structural Predictor Effect ===")
for pc in ['GPCR', 'SLC']:
    sub = df[df['protein_class']==pc]
    pos = sub[sub['structural_predictor']=='Positive']['pearson_r']
    neg = sub[sub['structural_predictor']=='Negative']['pearson_r']
    u, p = stats.mannwhitneyu(pos, neg, alternative='two-sided')
    print(f"{pc}:")
    print(f"  Positive: n={len(pos)}, mean r={pos.mean():.3f}, median r={pos.median():.3f}")
    print(f"  Negative: n={len(neg)}, mean r={neg.mean():.3f}, median r={neg.median():.3f}")
    print(f"  P = {p:.4f}")

# === 3. 統合した構造的予測因子の効果 ===
print(f"\n=== Integrated Structural Predictor (GPCR+SLC) ===")
pos_all = df[df['structural_predictor']=='Positive']['pearson_r']
neg_all = df[df['structural_predictor']=='Negative']['pearson_r']
u, p = stats.mannwhitneyu(pos_all, neg_all, alternative='two-sided')
print(f"Positive: n={len(pos_all)}, mean r={pos_all.mean():.3f}, median r={pos_all.median():.3f}")
print(f"Negative: n={len(neg_all)}, mean r={neg_all.mean():.3f}, median r={neg_all.median():.3f}")
print(f"P = {p:.4f}")

# === 4. drug_scoreの統合効果 ===
print(f"\n=== drug_score vs Pearson_r (integrated) ===")
r, p = stats.pearsonr(df['drug_score'], df['pearson_r'])
print(f"Pearson r = {r:.3f}, P = {p:.4f}")

# 二値化
threshold = df['drug_score'].median()
high = df[df['drug_score'] >= threshold]['pearson_r']
low  = df[df['drug_score'] < threshold]['pearson_r']
u, p2 = stats.mannwhitneyu(high, low, alternative='two-sided')
print(f"Median split ({threshold:.3f}): High mean r={high.mean():.3f}, Low mean r={low.mean():.3f}, P={p2:.4f}")

# === 5. 4グループ比較（protein_class × structural_predictor） ===
print(f"\n=== Four-group: protein_class × structural_predictor ===")
for pc in ['GPCR', 'SLC']:
    for sp in ['Positive', 'Negative']:
        sub = df[(df['protein_class']==pc) & (df['structural_predictor']==sp)]['pearson_r']
        print(f"{pc} × {sp:8s}: n={len(sub):3d}, mean r={sub.mean():.3f}, median r={sub.median():.3f}")

# === 6. Variance explained ===
print(f"\n=== Variance explained (eta²) ===")
grand_mean = df['pearson_r'].mean()
ss_total = ((df['pearson_r'] - grand_mean)**2).sum()

ss_class = sum(
    len(df[df['protein_class']==v]) * (df[df['protein_class']==v]['pearson_r'].mean() - grand_mean)**2
    for v in ['GPCR','SLC']
)
ss_pred = sum(
    len(df[df['structural_predictor']==v]) * (df[df['structural_predictor']==v]['pearson_r'].mean() - grand_mean)**2
    for v in ['Positive','Negative']
)
r_drug, _ = stats.pearsonr(df['drug_score'], df['pearson_r'])
print(f"Protein class eta²:           {ss_class/ss_total:.4f} ({ss_class/ss_total*100:.1f}%)")
print(f"Structural predictor eta²:    {ss_pred/ss_total:.4f} ({ss_pred/ss_total*100:.1f}%)")
print(f"drug_score r²:                {r_drug**2:.4f} ({r_drug**2*100:.1f}%)")

# === 7. 最高精度グループ vs 最低精度グループ ===
print(f"\n=== Best vs Worst group ===")
best  = df[(df['structural_predictor']=='Positive')]['pearson_r']
worst = df[(df['structural_predictor']=='Negative') & (df['protein_class']=='SLC')]['pearson_r']
u, p = stats.mannwhitneyu(best, worst, alternative='two-sided')
print(f"Predictor(+) all:     n={len(best)},  mean r={best.mean():.3f}")
print(f"Predictor(-) SLC:     n={len(worst)}, mean r={worst.mean():.3f}")
print(f"P = {p:.4f}")

# === 8. 各グループのperformance category ===
print(f"\n=== Performance categories by group ===")
for pc in ['GPCR', 'SLC']:
    for sp in ['Positive', 'Negative']:
        sub = df[(df['protein_class']==pc) & (df['structural_predictor']==sp)]['pearson_r']
        if len(sub) == 0:
            continue
        print(f"{pc} × {sp}:")
        print(f"  Strong (r>=0.5): {(sub>=0.5).sum()} ({(sub>=0.5).mean()*100:.1f}%)")
        print(f"  Inverse (r<0):   {(sub<0).sum()} ({(sub<0).mean()*100:.1f}%)")

df.to_csv('integrated_analysis.csv', index=False)
print("\nSaved: integrated_analysis.csv")
