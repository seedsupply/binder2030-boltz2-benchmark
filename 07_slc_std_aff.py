import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import os
import json
import warnings
warnings.filterwarnings('ignore')

predictions_dir = os.path.expanduser(
    '~/boltz_outputs_slc/boltz_results_boltz_inputs_slc4/predictions')
df_slc_corr = pd.read_csv('slc_correlation_annotated2.csv')
df_slc_props = pd.read_csv('slc_compound_props.csv').drop(columns=['pearson_r'])
df_slc_ann = pd.read_csv('slc_tanimoto_analysis.csv')

# === SLCのaffinity値を全compound分収集 ===
print("Collecting SLC affinity predictions...")
records = []
for compound in os.listdir(predictions_dir):
    aff_file = os.path.join(predictions_dir, compound, f'affinity_{compound}.json')
    if not os.path.exists(aff_file):
        continue
    with open(aff_file) as f:
        aff = json.load(f)
    records.append({
        'Compound': compound,
        'affinity_pred_value':  aff.get('affinity_pred_value'),
        'affinity_pred_value1': aff.get('affinity_pred_value1'),
        'affinity_pred_value2': aff.get('affinity_pred_value2'),
        'affinity_probability_binary': aff.get('affinity_probability_binary'),
    })

df_pred = pd.DataFrame(records)
print(f"  Loaded: {len(df_pred)} compounds")

# SLC_AI_calとマージしてTargetを取得
df_slc_pairs = pd.read_excel('SLC_AI_cal.xlsx')[['Compound','Target','UniProt','pKd']]
df_merged = pd.merge(df_slc_pairs, df_pred, on='Compound')
print(f"  Merged: {len(df_merged)} compound-target pairs")

# アンサンブル標準偏差
df_merged['ensemble_std'] = df_merged[[
    'affinity_pred_value','affinity_pred_value1','affinity_pred_value2']].std(axis=1)

# === ターゲットごとに集計 ===
aff_stats = df_merged.groupby('Target').agg(
    std_aff_value=('affinity_pred_value', 'std'),
    mean_aff_prob=('affinity_probability_binary', 'mean'),
    mean_ensemble_std=('ensemble_std', 'mean'),
    std_pKd=('pKd', 'std'),
    n=('pKd', 'count')
).reset_index()

df_g = pd.merge(df_slc_corr[['Target','UniProt','pearson_r','pocket_type']],
                aff_stats, on='Target')
df_g = pd.merge(df_g, df_slc_props[['Target','logP_mean','diversity_MW','RO5_ratio']], on='Target')
df_g['clear_pocket'] = (df_g['pocket_type'] == 'Clear_orthosteric').astype(int)
print(f"  Final: {len(df_g)} targets")

# === 1. 各指標 vs Pearson_r ===
print(f"\n{'='*60}")
print("1. New features vs Pearson_r (SLC)")
print(f"{'='*60}")

for feat in ['std_aff_value','mean_aff_prob','mean_ensemble_std','std_pKd']:
    r, p = stats.pearsonr(df_g[feat], df_g['pearson_r'])
    sig = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else ''
    print(f"  {feat:25s}: r={r:+.3f}, r²={r**2:.4f} ({r**2*100:.1f}%), P={p:.4f} {sig}")

# === 2. 循環論法の検証 ===
print(f"\n{'='*60}")
print("2. Circular reasoning check (SLC)")
print(f"{'='*60}")

r_pkd, p_pkd = stats.pearsonr(df_g['std_pKd'], df_g['pearson_r'])
r_cross, p_cross = stats.pearsonr(df_g['std_aff_value'], df_g['std_pKd'])
print(f"  std_pKd vs pearson_r:       r={r_pkd:.3f}, P={p_pkd:.4f}")
print(f"  std_aff_value vs std_pKd:   r={r_cross:.3f}, P={p_cross:.4f}")

# 偏相関
scaler = StandardScaler()
df_clean = df_g[['pearson_r','std_aff_value','std_pKd']].dropna()
X_pkd = scaler.fit_transform(df_clean[['std_pKd']])
resid_r = df_clean['pearson_r'] - LinearRegression().fit(
    X_pkd, df_clean['pearson_r']).predict(X_pkd)
resid_aff = df_clean['std_aff_value'] - LinearRegression().fit(
    X_pkd, df_clean['std_aff_value']).predict(X_pkd)
r_partial, p_partial = stats.pearsonr(resid_aff, resid_r)
sig = '***' if p_partial<0.001 else '**' if p_partial<0.01 else '*' if p_partial<0.05 else 'n.s.'
print(f"  Partial corr (std_pKd除去後): r={r_partial:.3f}, P={p_partial:.4f} {sig}")

# === 3. 改善された複合モデル ===
print(f"\n{'='*60}")
print("3. Updated SLC combined model")
print(f"{'='*60}")

models = {
    'Baseline (pocket+logP+divMW+RO5)':
        ['clear_pocket','logP_mean','diversity_MW','RO5_ratio'],
    '+std_aff_value':
        ['clear_pocket','logP_mean','diversity_MW','RO5_ratio','std_aff_value'],
    '+mean_ensemble_std':
        ['clear_pocket','logP_mean','diversity_MW','RO5_ratio','mean_ensemble_std'],
    '+both':
        ['clear_pocket','logP_mean','diversity_MW','RO5_ratio',
         'std_aff_value','mean_ensemble_std'],
    'Full model':
        ['clear_pocket','logP_mean','diversity_MW','RO5_ratio',
         'std_aff_value','mean_ensemble_std','mean_aff_prob'],
}

for name, feats in models.items():
    df_m = df_g[feats+['pearson_r']].dropna()
    X = scaler.fit_transform(df_m[feats])
    r2 = LinearRegression().fit(X, df_m['pearson_r']).score(X, df_m['pearson_r'])
    print(f"  {name:45s}: R²={r2:.4f} ({r2*100:.1f}%)")

# === 4. clear_pocket × std_aff_value交互作用 ===
print(f"\n{'='*60}")
print("4. clear_pocket × std_aff_value interaction")
print(f"{'='*60}")

threshold = df_g['std_aff_value'].median()
df_g['high_std'] = (df_g['std_aff_value'] >= threshold).astype(int)
for pocket in ['Clear_orthosteric', 'Other']:
    for std in ['high', 'low']:
        mask = (df_g['pocket_type']==pocket) & \
               (df_g['high_std']==(1 if std=='high' else 0))
        sub = df_g[mask]['pearson_r']
        print(f"  {pocket:20s} × std_{std}: n={len(sub):2d}, mean r={sub.mean():.3f}")

# === 5. GPCRとSLCの比較 ===
print(f"\n{'='*60}")
print("5. GPCR vs SLC: std_aff_value effect")
print(f"{'='*60}")
print(f"  GPCR std_aff_value: r²=14.0%, P=0.0003 ***")
r_slc, p_slc = stats.pearsonr(df_g['std_aff_value'], df_g['pearson_r'])
print(f"  SLC  std_aff_value: r²={r_slc**2*100:.1f}%, P={p_slc:.4f} "
      f"{'***' if p_slc<0.001 else '**' if p_slc<0.01 else '*' if p_slc<0.05 else 'n.s.'}")

# === 6. 最終分散説明率サマリー ===
print(f"\n{'='*60}")
print("6. Final variance explained summary")
print(f"{'='*60}")

best_feats = ['clear_pocket','logP_mean','diversity_MW','RO5_ratio','std_aff_value']
df_best = df_g[best_feats+['pearson_r']].dropna()
X_best = scaler.fit_transform(df_best[best_feats])
model_best = LinearRegression().fit(X_best, df_best['pearson_r'])
r2_best = model_best.score(X_best, df_best['pearson_r'])

print(f"\n  SLC progression:")
print(f"    Baseline (pocket+logP+divMW+RO5): 27.3%")
print(f"    +std_aff_value:                   {r2_best*100:.1f}%")
print(f"    Remaining residual:               {(1-r2_best)*100:.1f}%")
print(f"\n  GPCR progression:")
print(f"    cwxp_type+ligand_type:            12.9%")
print(f"    +std_aff_value+std_iptm:          21.4%")
print(f"    Remaining residual:               78.6%")
print(f"\n  Both classes combined insight:")
print(f"    std_aff_value = Boltz-2の識別能力の直接指標")
print(f"    → 予測前に計算不可（事後的指標）")
print(f"    → 論文ではPost-hoc analysisとして位置づけ")

df_g.to_csv('slc_std_aff_analysis.csv', index=False)
print("\nSaved: slc_std_aff_analysis.csv")
