#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from rdkit import Chem, rdBase
from build_molecule_index import GENERATOR, sha256
from sample_splits import read_records


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--iteration-dir', required=True)
    a = p.parse_args()
    root = Path(a.iteration_dir).resolve()
    audit = dict(rdkit_version=rdBase.rdkitVersion, fingerprint='Morgan radius=2 bits=1024 includeChirality=True', files={})
    for part in ['training', 'tuning', 'test']:
        records = read_records(root / 'splits' / f'{part}_split.txt')
        smiles = root / 'smiles' / f'{part}_split.smi'
        if read_records(smiles) != records:
            raise ValueError(f'SMILES membership/order differs for {part}')
        fp_path = root / 'fingerprints' / f'{part}_split_morgan.csv'
        found = {}
        with fp_path.open() as f:
            for line in f:
                fields = line.strip().split(',')
                cid = fields[0]
                if cid in found:
                    raise ValueError(f'Duplicate fingerprint ID: {cid}')
                found[cid] = [int(b) for b in fields[1:]]
        if set(found) != {cid for cid, _ in records}:
            raise ValueError(f'Fingerprint ID coverage differs for {part}')
        for cid, smi in records:
            expected = list(GENERATOR.GetFingerprint(Chem.MolFromSmiles(smi)).GetOnBits())
            if sorted(found[cid]) != expected:
                raise ValueError(f'Fingerprint differs from original SMILES: {cid}')
        # Normalize feature order to the frozen membership order; downstream label loaders match IDs.
        with fp_path.open('w') as f:
            for cid, _ in records:
                f.write(','.join([cid] + [str(b) for b in sorted(found[cid])]) + '\n')
        audit[part] = len(records)
        for path in [smiles, fp_path]:
            audit['files'][str(path.relative_to(root))] = sha256(path)
    (root / 'preparation_metadata.json').write_text(json.dumps(audit, indent=2) + '\n')
    print('Verified extracted SMILES and original chiral Morgan fingerprints:', root)


if __name__ == '__main__':
    main()
