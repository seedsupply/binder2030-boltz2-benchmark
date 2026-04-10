"""
08_prepare_slc_inputs.py

Prepare Boltz-2 input YAML files from SLC dataset.

NOTE: This script requires SLC_AI_cal.xlsx (proprietary Binder2030 data)
which is NOT publicly available. The script is provided for transparency
of methodology only. To reproduce this analysis, please contact
naoki.tarui@seedsupply.co.jp for a Data Use Agreement.

The SMILES strings and experimental Kd values in SLC_AI_cal.xlsx
are proprietary to SEEDSUPPLY INC. and cannot be shared.
"""
import pandas as pd
import os

# NOTE: SLC_AI_cal.xlsx is proprietary and not publicly available.
# See Data Availability statement in the manuscript.
df = pd.read_excel('SLC_AI_cal.xlsx')
os.makedirs('boltz_inputs_slc4', exist_ok=True)

count = 0
for _, row in df.iterrows():
    compound = str(row['Compound']).strip()
    smiles = str(row['SMILES']).strip()
    sequence = str(row['Sequence']).strip()
    uniprot = str(row['UniProt']).strip()

    if not smiles or smiles == 'nan' or not sequence or sequence == 'nan':
        continue

    msa_path = f"/home/ubuntu/msa_cache_slc/{uniprot}.a3m"

    # SMILES enclosed in single quotes to avoid YAML backslash issues
    yaml_content = f"""version: 1
sequences:
  - protein:
      id: A
      sequence: {sequence}
      msa: {msa_path}
  - ligand:
      id: B
      smiles: '{smiles}'
properties:
  - affinity:
      binder: B
"""
    with open(f"boltz_inputs_slc4/{compound}.yaml", 'w') as f:
        f.write(yaml_content)
    count += 1

print(f"Created: {count} YAML files in boltz_inputs_slc4/")
