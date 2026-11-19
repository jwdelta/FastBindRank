#!/usr/bin/env python3

import argparse
from multiprocessing import Pool, cpu_count
from pathlib import Path
import time

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024, includeChirality=True)


def fingerprint_chunk(task):
    """Write one chunk of original-molecule fingerprints; invalid structures are skipped."""
    source, directory = task
    source = Path(source)
    stem = source.stem
    name = f'morgan_chunk_{stem[len("smile_all_"):]}.csv' if stem.startswith('smile_all_') else f'{stem}.csv'
    with (Path(directory) / name).open('w') as out, source.open() as handle:
        for n, line in enumerate(handle, 1):
            fields = line.split()
            if len(fields) != 2:
                raise ValueError(f'Malformed SMILES record in {source}:{n}')
            smiles, cid = fields
            molecule = Chem.MolFromSmiles(smiles)
            if molecule is None:
                continue
            bits = GENERATOR.GetFingerprint(molecule).GetOnBits()
            out.write(','.join([cid] + [str(bit) for bit in bits]) + '\n')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('-sfp','--smile_folder_path', required=True)
    p.add_argument('-fn','--folder_name', required=True)
    p.add_argument('-tp','--tot_process', type=int, required=True)
    a = p.parse_args()
    if a.tot_process < 1:
        p.error('--tot_process must be positive')
    files = sorted(Path(a.smile_folder_path).glob('*.txt'))
    if not files:
        raise ValueError('No SMILES chunks found')
    Path(a.folder_name).mkdir(parents=True, exist_ok=True)
    start = time.time()
    with Pool(min(a.tot_process, cpu_count(), len(files))) as pool:
        pool.map(fingerprint_chunk, [(str(f),a.folder_name) for f in files])
    print(f'Fingerprint generation completed in {time.time()-start:.2f} seconds')


if __name__ == '__main__':
    main()
