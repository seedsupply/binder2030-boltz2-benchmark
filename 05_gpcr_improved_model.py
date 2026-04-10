import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from rdkit import Chem
from rdkit.Chem import DataStructs
from rdkit.Chem import rdMolDescriptors
import warnings
warnings.filterwarnings('ignore')

df_gpcr = pd.read_csv('correlation_final.csv').rename(columns={'Pearson_r':'pearson_r'})
df_gpcr_props = pd.read_csv('gpcr_compound_props.csv').drop(columns=['pearson_r'])
df_conf = pd.read_csv('boltz2_confidence.csv')
df_pairs = pd.read_csv('merged_results2.csv')
df_ligand = pd.read_csv('fpocket_gpcr_ligandtype.csv')

# ipTM計算
df_pairs_conf = pd.merge(df_pairs, df_conf, on='fname')
iptm_stats = df_pairs_conf.groupby('Target').agg(
    mean_iptm=('iptm', 'mean')).reset_index()

# === 方法1: CWxP x残基タイプの細分化 ===
print("=" * 60)
print("1. CWxP x-residue type refinement")
print("=" * 60)

def get_cwxp_xresidue(seq):
    """TM6のCWxPモチーフのx残基を取得"""
    import re
    matches = list(re.finditer(r'CW(.)P', seq))
    if matches:
        return matches[0].group(1)
    return None

df_gpcr['cwxp_x'] = df_gpcr['Sequence'].apply(get_cwxp_xresidue)
small_hydrophobic = ['G','A','L','M','V']
df_gpcr['cwxp_type'] = df_gpcr['cwxp_x'].apply(
    lambda x: 2 if x in small_hydrophobic else (1 if x is not None else 0)
)
# 0: CWxPなし, 1: CWxP(other x), 2: CWxP(small/hydrophobic x)

print("CWxP type distribution:")
print(df_gpcr['cwxp_type'].value_counts().sort_index())
for t in [0,1,2]:
    sub = df_gpcr[df_gpcr['cwxp_type']==t]['pearson_r']
    print(f"  type {t}: n={len(sub)}, mean r={sub.mean():.3f}")

r, p = stats.pearsonr(df_gpcr['cwxp_type'], df_gpcr['pearson_r'])
print(f"\ncwxp_type vs pearson_r: r={r:.3f}, r²={r**2:.4f} ({r**2*100:.1f}%), P={p:.4f}")

# === 方法2: motif_score（CWxP + DRY + NPxxY）===
print(f"\n{'='*60}")
print("2. Multiple motif score")
print(f"{'='*60}")

df_gpcr['motif_score'] = df_gpcr['has_CWxP'] + df_gpcr['has_DRY'] + df_gpcr['has_NPxxY']
df_gpcr['motif_score_cwxp'] = df_gpcr['cwxp_type'] + df_gpcr['has_DRY'] + df_gpcr['has_NPxxY']

for feat in ['motif_score', 'motif_score_cwxp']:
    r, p = stats.pearsonr(df_gpcr[feat], df_gpcr['pearson_r'])
    sig = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else ''
    print(f"  {feat:25s}: r={r:.3f}, r²={r**2:.4f} ({r**2*100:.1f}%), P={p:.4f} {sig}")

# === 方法3: Ligand typeのダミー変数 ===
print(f"\n{'='*60}")
print("3. Ligand type dummy variables")
print(f"{'='*60}")

df_merged = pd.merge(df_gpcr, df_ligand[['Target','ligand_type']], on='Target', how='left')
df_merged['ligand_type'] = df_merged['ligand_type'].fillna('other')

# ダミー変数作成
dummies = pd.get_dummies(df_merged['ligand_type'], prefix='lt')
df_merged = pd.concat([df_merged, dummies], axis=1)

lt_cols = [c for c in dummies.columns]
print("Ligand type dummy correlations:")
for col in lt_cols:
    r, p = stats.pearsonr(df_merged[col], df_merged['pearson_r'])
    sig = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else ''
    print(f"  {col:25s}: r={r:+.3f}, P={p:.4f} {sig}")

# === 方法4: Tanimoto類似度（化合物構造多様性）===
print(f"\n{'='*60}")
print("4. Tanimoto similarity (structural diversity)")
print(f"{'='*60}")

def calc_tanimoto_diversity(smiles_list):
    """化合物セットの平均Tanimoto類似度（1=全て同じ、0=全て異なる）"""
    mols = [Chem.MolFromSmiles(s) for s in smiles_list if Chem.MolFromSmiles(s)]
    if len(mols) < 2:
        return None
    fps = [rdMolDescriptors.GetMorganFingerprintAsBitVect(m, 2, 2048) for m in mols]
    sims = []
    for i in range(len(fps)):
        for j in range(i+1, len(fps)):
            sims.append(DataStructs.TanimotoSimilarity(fps[i], fps[j]))
    return np.mean(sims)

print("Calculating Tanimoto diversity per target...")
tanimoto_results = []
for target, group in df_pairs.groupby('Target'):
    div = calc_tanimoto_diversity(group['SMILES'].tolist())
    tanimoto_results.append({'Target': target, 'mean_tanimoto': div})
df_tanimoto = pd.DataFrame(tanimoto_results)
print(f"  Calculated: {df_tanimoto['mean_tanimoto'].notna().sum()} targets")

df_merged = pd.merge(df_merged, df_tanimoto, on='Target', how='left')
r, p = stats.pearsonr(df_merged['mean_tanimoto'].dropna(),
                      df_merged.loc[df_merged['mean_tanimoto'].notna(), 'pearson_r'])
print(f"  mean_tanimoto vs pearson_r: r={r:.3f}, r²={r**2:.4f} ({r**2*100:.1f}%), P={p:.4f}")

# 低類似度（高多様性）vs 高類似度（低多様性）
threshold = df_merged['mean_tanimoto'].median()
low_sim = df_merged[df_merged['mean_tanimoto'] < threshold]['pearson_r']
high_sim = df_merged[df_merged['mean_tanimoto'] >= threshold]['pearson_r']
u, p2 = stats.mannwhitneyu(low_sim, high_sim, alternative='two-sided')
print(f"  Low similarity (high diversity): mean r={low_sim.mean():.3f}")
print(f"  High similarity (low diversity): mean r={high_sim.mean():.3f}")
print(f"  P = {p2:.4f}")

# === 方法5: 改善された複合モデル ===
print(f"\n{'='*60}")
print("5. Improved combined model (GPCR)")
print(f"{'='*60}")

df_full = pd.merge(df_merged, df_gpcr_props, on='Target', how='left')
df_full = pd.merge(df_full, iptm_stats, on='Target', how='left')

# 全特徴量の相関サマリー
print("All features vs pearson_r:")
all_feats = ['cwxp_type','motif_score_cwxp','mean_iptm',
             'logP_mean','diversity_MW','mean_tanimoto',
             'lt_small_molecule','lt_peptide','lt_lipid','lt_photon']
for feat in all_feats:
    if feat in df_full.columns:
        vals = df_full[feat].fillna(df_full[feat].median())
        r, p = stats.pearsonr(vals, df_full['pearson_r'])
        sig = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else ''
        print(f"  {feat:25s}: r={r:+.3f}, r²={r**2:.4f} ({r**2*100:.1f}%), P={p:.4f} {sig}")

# モデル1: 元のモデル（CWxP + ipTM + logP）
feats1 = ['has_CWxP','mean_iptm','logP_mean']
df_m1 = df_full[feats1 + ['pearson_r']].dropna()
scaler = StandardScaler()
X1 = scaler.fit_transform(df_m1[feats1])
m1 = LinearRegression().fit(X1, df_m1['pearson_r'])
r2_1 = m1.score(X1, df_m1['pearson_r'])
print(f"\n  Model 1 (CWxP+ipTM+logP): R²={r2_1:.4f} ({r2_1*100:.1f}%)")

# モデル2: cwxp_type使用
feats2 = ['cwxp_type','mean_iptm','logP_mean']
df_m2 = df_full[feats2 + ['pearson_r']].dropna()
X2 = scaler.fit_transform(df_m2[feats2])
m2 = LinearRegression().fit(X2, df_m2['pearson_r'])
r2_2 = m2.score(X2, df_m2['pearson_r'])
print(f"  Model 2 (cwxp_type+ipTM+logP): R²={r2_2:.4f} ({r2_2*100:.1f}%)")

# モデル3: motif_score + ipTM + logP
feats3 = ['motif_score_cwxp','mean_iptm','logP_mean']
df_m3 = df_full[feats3 + ['pearson_r']].dropna()
X3 = scaler.fit_transform(df_m3[feats3])
m3 = LinearRegression().fit(X3, df_m3['pearson_r'])
r2_3 = m3.score(X3, df_m3['pearson_r'])
print(f"  Model 3 (motif_score+ipTM+logP): R²={r2_3:.4f} ({r2_3*100:.1f}%)")

# モデル4: cwxp_type + ipTM + logP + tanimoto
feats4 = ['cwxp_type','mean_iptm','logP_mean','mean_tanimoto']
df_m4 = df_full[feats4 + ['pearson_r']].dropna()
X4 = scaler.fit_transform(df_m4[feats4])
m4 = LinearRegression().fit(X4, df_m4['pearson_r'])
r2_4 = m4.score(X4, df_m4['pearson_r'])
print(f"  Model 4 (cwxp_type+ipTM+logP+tanimoto): R²={r2_4:.4f} ({r2_4*100:.1f}%)")

# モデル5: 全特徴量
feats5 = ['cwxp_type','motif_score_cwxp','mean_iptm','logP_mean',
          'mean_tanimoto','lt_small_molecule','lt_peptide']
df_m5 = df_full[feats5 + ['pearson_r']].dropna()
X5 = scaler.fit_transform(df_m5[feats5])
m5 = LinearRegression().fit(X5, df_m5['pearson_r'])
r2_5 = m5.score(X5, df_m5['pearson_r'])
print(f"  Model 5 (all features): R²={r2_5:.4f} ({r2_5*100:.1f}%)")
for feat, coef in zip(feats5, m5.coef_):
    print(f"    {feat}: coef={coef:.3f}")

# === 6. 最終サマリー ===
print(f"\n{'='*60}")
print("6. Final comparison GPCR vs SLC")
print(f"{'='*60}")
print(f"  GPCR baseline (CWxP+ipTM+logP):     8.6%")
print(f"  GPCR improved (best model):          {max(r2_1,r2_2,r2_3,r2_4,r2_5)*100:.1f}%")
print(f"  SLC model (pocket+logP+div+RO5):    27.3%")

df_full.to_csv('gpcr_full_analysis.csv', index=False)
print("\nSaved: gpcr_full_analysis.csv")
