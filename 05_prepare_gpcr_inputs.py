"""
05_prepare_gpcr_inputs.py
=========================
Boltz-2 input YAML preparation for GPCR virtual screening
Generates input YAML files for each compound-target pair

NOTE: Requires proprietary compound SMILES data
      (Data Use Agreement required - contact naoki.tarui@seedsupply.co.jp)

Input:  compounds_1200.csv (proprietary SMILES data)
        gpcr_correlation_results.csv
        MSA files (pre-computed via ColabFold)
Output: boltz2_inputs/{TARGET}/*.yaml
"""

import os
import pandas as pd
import argparse

def generate_yaml(sequence, smiles, msa_path):
    """Generate Boltz-2 input YAML content."""
    return f"""version: 1
sequences:
  - protein:
      id: A
      msa: {msa_path}
      sequence: {sequence}
  - ligand:
      id: B
      smiles: '{smiles}'
properties:
  - affinity:
      binder: B
"""

def main(smiles_file, target_list, output_dir, msa_dir):
    # Load compound data (proprietary)
    try:
        compounds = pd.read_csv(smiles_file)
    except FileNotFoundError:
        print(f"Error: {smiles_file} not found.")
        print("Compound SMILES data requires a Data Use Agreement.")
        print("Contact: naoki.tarui@seedsupply.co.jp")
        return

    id_col = [c for c in compounds.columns
              if 'compound' in c.lower() or 'id' in c.lower()][0]
    smiles_col = [c for c in compounds.columns
                  if 'smiles' in c.lower()][0]

    # Load target list with sequences
    targets_df = pd.read_csv(target_list)

    total = 0
    for _, target_row in targets_df.iterrows():
        target = target_row['Target']
        sequence = target_row.get('Sequence', '')

        if not sequence:
            print(f"  Skipping {target}: no sequence")
            continue

        # MSA path
        msa_path = os.path.join(msa_dir, f"{target}_msa", "bfd_clean.a3m")

        target_dir = os.path.join(output_dir, target)
        os.makedirs(target_dir, exist_ok=True)

        count = 0
        for _, row in compounds.iterrows():
            cid = str(row[id_col]).strip()
            smi = str(row[smiles_col]).strip()
            if not smi or smi == 'nan':
                continue

            yaml_content = generate_yaml(sequence, smi, msa_path)
            with open(os.path.join(target_dir, f"{cid}.yaml"), 'w') as f:
                f.write(yaml_content)
            count += 1

        print(f"  {target}: {count} YAML files generated")
        total += count

    print(f"\nTotal: {total} YAML files generated in {output_dir}/")
    print("Next step: run Boltz-2 predictions")
    print("  boltz predict boltz2_inputs/{TARGET}/ \\")
    print("    --out_dir boltz2_outputs/{TARGET} \\")
    print("    --accelerator gpu --devices 1")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Prepare Boltz-2 input YAML files for GPCR virtual screening'
    )
    parser.add_argument('--smiles', required=True,
                        help='Compound SMILES CSV file (proprietary)')
    parser.add_argument('--targets', default='data/gpcr_correlation_results.csv',
                        help='Target list with sequences')
    parser.add_argument('--output_dir', default='boltz2_inputs',
                        help='Output directory for YAML files')
    parser.add_argument('--msa_dir', default='msa',
                        help='Directory containing MSA files')
    args = parser.parse_args()
    main(args.smiles, args.targets, args.output_dir, args.msa_dir)
