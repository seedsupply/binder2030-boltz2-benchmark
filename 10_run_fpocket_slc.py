import pandas as pd
import requests
import os
import subprocess

df = pd.read_excel('SLC_UniProt_Sequence.xlsx')
print(f"Loaded: {len(df)} SLC targets")

os.makedirs('alphafold_slc', exist_ok=True)

results = []

for i, row in df.iterrows():
    uniprot = row['UniProt']
    target = row['Target']
    print(f"[{i+1}/82] {target} ({uniprot})")

    # AlphaFold構造ダウンロード
    pdb_path = f"alphafold_slc/{uniprot}.pdb"
    if not os.path.exists(pdb_path):
        api_url = f"https://alphafold.ebi.ac.uk/api/prediction/{uniprot}"
        r = requests.get(api_url)
        if r.status_code == 200:
            pdb_url = r.json()[0]['pdbUrl']
            r2 = requests.get(pdb_url)
            if r2.status_code == 200:
                with open(pdb_path, 'w') as f:
                    f.write(r2.text)
                print(f"  Downloaded")
            else:
                print(f"  PDB failed: {r2.status_code}")
                results.append({'UniProt': uniprot, 'Target': target,
                               'drug_score': None, 'pocket_volume': None, 'n_pockets': 0})
                continue
        else:
            print(f"  API failed: {r.status_code}")
            results.append({'UniProt': uniprot, 'Target': target,
                           'drug_score': None, 'pocket_volume': None, 'n_pockets': 0})
            continue
    else:
        print(f"  SKIP (cached)")

    # fpocket実行
    info_file = f"alphafold_slc/{uniprot}_out/{uniprot}_info.txt"
    if not os.path.exists(info_file):
        subprocess.run(f"fpocket -f {uniprot}.pdb", shell=True,
                      capture_output=True, cwd='alphafold_slc')

    if not os.path.exists(info_file):
        print(f"  No info file")
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
        print(f"  -> {len(drug_scores)} pockets, best={max(drug_scores):.3f}")
    else:
        results.append({'UniProt': uniprot, 'Target': target,
                       'drug_score': None, 'pocket_volume': None, 'n_pockets': 0})
        print(f"  -> no pockets")

df_results = pd.DataFrame(results)
df_results.to_csv('fpocket_slc_results.csv', index=False)
print(f"\nDone. Success: {df_results['drug_score'].notna().sum()}/82")
print(df_results[['Target','drug_score','pocket_volume','n_pockets']].to_string())
