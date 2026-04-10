import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
import warnings
warnings.filterwarnings('ignore')
RDLogger.DisableLog('rdApp.warning')

df_pairs = pd.read_csv('merged_results2.csv')
df_conf = pd.read_csv('boltz2_confidence.csv')
df_gpcr = pd.read_csv('correlation_final.csv').rename(columns={'Pearson_r':'pearson_r'})
df_gpcr_props = pd.read_csv('gpcr_compound_props.csv').drop(columns=['pearson_r'])

# merged_results2にconfidenceをマージ
df_pairs_conf = pd.merge(df_pairs, df_conf, on='fname')
print(f"Pairs with confidence: {len(df_pairs_conf)}")
print(f"Columns: {list(df_pairs_conf.columns)}")

# ===================================================
# ⑧ affinity_probability_binary
# ===================================================
print(f"\n{'='*60}")
print("8. affinity_probability_binary vs Pearson_r")
print(f"{'='*60}")

aff_stats = df_pairs_conf.groupby('Target').agg(
    mean_aff_prob=('affinity_probability_binary', 'mean'),
    std_aff_prob=('affinity_probability_binary', 'std'),
    mean_aff_value=('affinity_pred_value', 'mean'),
    std_aff_value=('affinity_pred_value', 'std'),
).reset_index()

df_g = pd.merge(df_gpcr[['Target','pearson_r','has_CWxP']], aff_stats, on='Target')
print(f"Merged: {len(df_g)} targets")

for feat in ['mean_aff_prob','std_aff_prob','mean_aff_value','std_aff_value']:
    r, p = stats.pearsonr(df_g[feat], df_g['pearson_r'])
    sig = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else ''
    print(f"  {feat:22s}: r={r:+.3f}, r²={r**2:.4f} ({r**2*100:.1f}%), P={p:.4f} {sig}")

# ===================================================
# ⑨ アンサンブル一致度（pred_value1 vs pred_value2）
# ===================================================
print(f"\n{'='*60}")
print("9. Ensemble consistency (pred_value1 vs pred_value2)")
print(f"{'='*60}")

# pred_value1とpred_value2の差の絶対値
if 'affinity_pred_value1' in df_pairs_conf.columns:
    df_pairs_conf['ensemble_diff'] = abs(
        df_pairs_conf['affinity_pred_value1'] - df_pairs_conf['affinity_pred_value2'])
    df_pairs_conf['ensemble_std'] = df_pairs_conf[[
        'affinity_pred_value','affinity_pred_value1','affinity_pred_value2']].std(axis=1)

    ens_stats = df_pairs_conf.groupby('Target').agg(
        mean_ensemble_diff=('ensemble_diff', 'mean'),
        mean_ensemble_std=('ensemble_std', 'mean'),
    ).reset_index()

    df_g2 = pd.merge(df_g, ens_stats, on='Target')
    for feat in ['mean_ensemble_diff','mean_ensemble_std']:
        r, p = stats.pearsonr(df_g2[feat], df_g2['pearson_r'])
        sig = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else ''
        print(f"  {feat:25s}: r={r:+.3f}, r²={r**2:.4f} ({r**2*100:.1f}%), P={p:.4f} {sig}")
else:
    print("  affinity_pred_value1/2 not found in data")
    df_g2 = df_g.copy()

# ===================================================
# ⑩ ipTMのばらつき（化合物間）
# ===================================================
print(f"\n{'='*60}")
print("10. ipTM variability across compounds")
print(f"{'='*60}")

iptm_var = df_pairs_conf.groupby('Target').agg(
    mean_iptm=('iptm', 'mean'),
    std_iptm=('iptm', 'std'),
    cv_iptm=('iptm', lambda x: x.std()/x.mean() if x.mean()>0 else 0),
    mean_ligand_iptm=('ligand_iptm', 'mean'),
    std_ligand_iptm=('ligand_iptm', 'std'),
).reset_index()

df_g3 = pd.merge(df_g2, iptm_var, on='Target', how='left')
for feat in ['mean_iptm','std_iptm','cv_iptm','mean_ligand_iptm','std_ligand_iptm']:
    if feat in df_g3.columns:
        vals = df_g3[feat].dropna()
        r, p = stats.pearsonr(vals, df_g3.loc[vals.index,'pearson_r'])
        sig = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else ''
        print(f"  {feat:25s}: r={r:+.3f}, r²={r**2:.4f} ({r**2*100:.1f}%), P={p:.4f} {sig}")

# ===================================================
# ③ RotBonds分散
# ===================================================
print(f"\n{'='*60}")
print("3. RotBonds diversity vs Pearson_r")
print(f"{'='*60}")

from rdkit.Chem import Descriptors, rdMolDescriptors

rot_stats = []
for target, group in df_pairs.groupby('Target'):
    rots = []
    for smi in group['SMILES']:
        mol = Chem.MolFromSmiles(str(smi))
        if mol:
            rots.append(rdMolDescriptors.CalcNumRotatableBonds(mol))
    if rots:
        rot_stats.append({
            'Target': target,
            'RotBonds_mean': np.mean(rots),
            'RotBonds_std': np.std(rots),
            'RotBonds_range': max(rots) - min(rots)
        })
df_rot = pd.DataFrame(rot_stats)
df_g4 = pd.merge(df_g3, df_rot, on='Target', how='left')

for feat in ['RotBonds_std','RotBonds_range']:
    r, p = stats.pearsonr(df_g4[feat].dropna(),
                          df_g4.loc[df_g4[feat].notna(),'pearson_r'])
    sig = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else ''
    print(f"  {feat:25s}: r={r:+.3f}, r²={r**2:.4f} ({r**2*100:.1f}%), P={p:.4f} {sig}")

# ===================================================
# ① Scaffold多様性
# ===================================================
print(f"\n{'='*60}")
print("1. Scaffold diversity (Murcko)")
print(f"{'='*60}")

scaffold_stats = []
for target, group in df_pairs.groupby('Target'):
    scaffolds = set()
    for smi in group['SMILES']:
        mol = Chem.MolFromSmiles(str(smi))
        if mol:
            try:
                sc = MurckoScaffold.GetScaffoldForMol(mol)
                scaffolds.add(Chem.MolToSmiles(sc))
            except:
                pass
    scaffold_stats.append({
        'Target': target,
        'n_scaffolds': len(scaffolds),
        'scaffold_ratio': len(scaffolds)/len(group) if len(group)>0 else 0
    })
df_scaffold = pd.DataFrame(scaffold_stats)
df_g5 = pd.merge(df_g4, df_scaffold, on='Target', how='left')

for feat in ['n_scaffolds','scaffold_ratio']:
    r, p = stats.pearsonr(df_g5[feat].dropna(),
                          df_g5.loc[df_g5[feat].notna(),'pearson_r'])
    sig = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else ''
    print(f"  {feat:25s}: r={r:+.3f}, r²={r**2:.4f} ({r**2*100:.1f}%), P={p:.4f} {sig}")

# ===================================================
# 改善された複合モデル
# ===================================================
print(f"\n{'='*60}")
print("Improved combined model")
print(f"{'='*60}")

scaler = StandardScaler()
df_full = pd.merge(df_g5, df_gpcr_props, on='Target', how='left')

# 元のモデル
feats0 = ['has_CWxP','mean_iptm','logP_mean']
df_m0 = df_full[feats0+['pearson_r']].dropna()
X0 = scaler.fit_transform(df_m0[feats0])
r2_0 = LinearRegression().fit(X0, df_m0['pearson_r']).score(X0, df_m0['pearson_r'])
print(f"  Baseline (CWxP+ipTM+logP):              R²={r2_0:.4f} ({r2_0*100:.1f}%)")

# 新特徴量を追加
new_feats_candidates = ['mean_aff_prob','mean_ensemble_std',
                        'std_iptm','n_scaffolds','RotBonds_std']
for new_feat in new_feats_candidates:
    if new_feat in df_full.columns:
        feats_new = feats0 + [new_feat]
        df_mn = df_full[feats_new+['pearson_r']].dropna()
        if len(df_mn) > 10:
            Xn = scaler.fit_transform(df_mn[feats_new])
            r2_n = LinearRegression().fit(Xn, df_mn['pearson_r']).score(Xn, df_mn['pearson_r'])
            delta = r2_n - r2_0
            print(f"  +{new_feat:22s}: R²={r2_n:.4f} ({r2_n*100:.1f}%), Δ={delta*100:+.1f}%")

# 最良の組み合わせ
best_feats = ['has_CWxP','mean_iptm','logP_mean',
              'mean_aff_prob','std_iptm','n_scaffolds']
df_best = df_full[best_feats+['pearson_r']].dropna()
X_best = scaler.fit_transform(df_best[best_feats])
model_best = LinearRegression().fit(X_best, df_best['pearson_r'])
r2_best = model_best.score(X_best, df_best['pearson_r'])
print(f"\n  Best combined model: R²={r2_best:.4f} ({r2_best*100:.1f}%)")
for feat, coef in zip(best_feats, model_best.coef_):
    print(f"    {feat}: coef={coef:.3f}")

# ===================================================
# 最終サマリー
# ===================================================
print(f"\n{'='*60}")
print("Final Summary: Variance explained progression")
print(f"{'='*60}")
print(f"  Step 1 - CWxP only:              4.2%")
print(f"  Step 2 - +ipTM:                  ~7%")
print(f"  Step 3 - +logP:                  8.6%")
print(f"  Step 4 - cwxp_type+ligand_type:  12.9%")
print(f"  Step 5 - best model:             {r2_best*100:.1f}%")
print(f"  Remaining residual:              {(1-r2_best)*100:.1f}%")

df_g5.to_csv('gpcr_residual_analysis.csv', index=False)
print("\nSaved: gpcr_residual_analysis.csv")
