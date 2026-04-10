import pandas as pd
import subprocess
import os

df = pd.read_excel('GPCR_Uniprot_Sequence.xlsx')
results = []

for i, row in df.iterrows():
    uniprot = row['UniProt']
    target = row['Target']
    pdb_path = f"alphafold_gpcr/{uniprot}.pdb"

    if not os.path.exists(pdb_path):
        print(f"[{i+1}/100] SKIP (no pdb): {uniprot}")
        results.append({'UniProt': uniprot, 'Target': target,
                       'drug_score': None, 'pocket_volume': None, 'n_pockets': 0})
        continue

    # fpocket実行（未実行のみ）
    info_file = f"alphafold_gpcr/{uniprot}_out/{uniprot}_info.txt"
    if not os.path.exists(info_file):
        subprocess.run(f"fpocket -f {uniprot}.pdb", shell=True,
                      capture_output=True, cwd='alphafold_gpcr')

    if not os.path.exists(info_file):
        print(f"[{i+1}/100] No info: {uniprot}")
        results.append({'UniProt': uniprot, 'Target': target,
                       'drug_score': None, 'pocket_volume': None, 'n_pockets': 0})
        continue

    # info.txtからスコア抽出
    drug_scores = []
    volumes = []
    with open(info_file) as f:
        content = f.read()

    current_drug = None
    current_vol = None
    for line in content.splitlines():
        if 'Druggability Score' in line:
            try:
                current_drug = float(line.split(':')[1].strip())
            except:
                pass
        if line.strip().startswith('Volume :'):
            try:
                current_vol = float(line.split(':')[1].strip())
            except:
                pass
        if current_drug is not None and current_vol is not None:
            drug_scores.append(current_drug)
            volumes.append(current_vol)
            current_drug = None
            current_vol = None

    if drug_scores:
        best_idx = drug_scores.index(max(drug_scores))
        results.append({
            'UniProt': uniprot,
            'Target': target,
            'drug_score': max(drug_scores),
            'pocket_volume': volumes[best_idx] if volumes else None,
            'n_pockets': len(drug_scores)
        })
        print(f"[{i+1}/100] {target}: {len(drug_scores)} pockets, best={max(drug_scores):.3f}")
    else:
        results.append({'UniProt': uniprot, 'Target': target,
                       'drug_score': None, 'pocket_volume': None, 'n_pockets': 0})
        print(f"[{i+1}/100] {target}: no pockets")

df_results = pd.DataFrame(results)
df_results.to_csv('fpocket_gpcr_results.csv', index=False)
print(f"\nDone. Success: {df_results['drug_score'].notna().sum()}/100")
print(df_results[['Target','drug_score','pocket_volume','n_pockets']].to_string())
