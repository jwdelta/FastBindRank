#!/usr/bin/env python3

import argparse
import hashlib
import json
import os
from pathlib import Path
from multiprocessing import Pool
import sqlite3
import sys

from rdkit import Chem, rdBase
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold

RULES = 'parent Murcko; largest fragment preferOrganic; nonisomeric scaffold; no neutralization/tautomer normalization; original isomeric canonical dedup; exclude empty scaffold/noncarbon/invalid/missing FP'
CHOOSER = rdMolStandardize.LargestFragmentChooser(preferOrganic=True)
GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024, includeChirality=True)


def sha256(path):
    """Hash file content without loading the file into memory."""
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def structure_record(line):
    """Return original structure, canonical identity, parent scaffold and eligibility status."""
    fields = line.split()
    if len(fields) != 2:
        raise ValueError('Expected exactly SMILES and ID: ' + line[:100])
    smiles, cid = fields
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return cid, smiles, '', '', 'invalid_structure'
    canonical = Chem.MolToSmiles(mol, isomericSmiles=True)
    parent = CHOOSER.choose(mol)
    scaffold = Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(parent), isomericSmiles=False)
    status = 'eligible'
    if not any(a.GetAtomicNum() == 6 for a in parent.GetAtoms()):
        status = 'nonorganic'
    elif not scaffold:
        status = 'empty_scaffold'
    else:
        GENERATOR.GetFingerprint(mol)  # Ensure original structure supports the model's FP.
    return cid, smiles, canonical, scaffold, status


def batches(iterable, size=10000):
    """Yield bounded batches for database insertion."""
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--smiles-file', required=True)
    p.add_argument('--morgan-directory', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--workers', type=int, default=8)
    args = p.parse_args()
    if args.workers < 1:
        p.error('--workers must be positive')
    source = Path(args.smiles_file).resolve()
    fp_files = sorted(Path(args.morgan_directory).resolve().glob('*.csv'))
    if not fp_files:
        raise ValueError('Morgan fingerprint files are required before building the index')
    signatures = {f.name: sha256(f) for f in fp_files}
    identity = dict(schema_version=1, rules=RULES, rdkit_version=rdBase.rdkitVersion,
                    smiles_sha256=sha256(source), fingerprint_sha256=signatures)
    output = Path(args.output).resolve()
    if output.exists():
        with sqlite3.connect(f'file:{output}?mode=ro', uri=True) as con:
            old = json.loads(con.execute('SELECT value FROM metadata WHERE key="identity"').fetchone()[0])
        if old != identity:
            raise ValueError('Existing molecule index differs from the source library or RDKit/rules. Use a new workspace or explicitly remove the obsolete index before rebuilding.')
        print('Reusing verified molecule index:', output)
        return
    temporary = output.with_suffix('.building.sqlite')
    if temporary.exists():
        raise ValueError(f'Incomplete index exists: {temporary}; remove it before rebuilding')
    con = sqlite3.connect(temporary)
    con.executescript('''PRAGMA journal_mode=DELETE; PRAGMA temp_store=FILE;
      CREATE TABLE molecules(cid TEXT PRIMARY KEY, original_smiles TEXT NOT NULL,
        canonical_smiles TEXT NOT NULL, scaffold TEXT NOT NULL, status TEXT NOT NULL);
      CREATE TABLE fingerprint_ids(cid TEXT PRIMARY KEY);
      CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);''')
    try:
        with source.open() as f, Pool(args.workers) as pool:
            for batch in batches(pool.imap(structure_record, f, chunksize=1000)):
                con.executemany('INSERT INTO molecules VALUES(?,?,?,?,?)', batch)
                con.commit()
        # Disk-backed ID matching verifies actual library fingerprints, without a 122M-ID Python set.
        for file in fp_files:
            with file.open() as f:
                def ids():
                    for line in f:
                        fields = line.strip().split(',')
                        if len(fields) < 2 or not fields[0]:
                            raise ValueError(f'Malformed fingerprint in {file}')
                        bits = [int(b) for b in fields[1:]]
                        if len(bits) != len(set(bits)) or any(b < 0 or b >= 1024 for b in bits):
                            raise ValueError(f'Invalid 1024-bit fingerprint in {file}: {fields[0]}')
                        yield (fields[0],)
                for batch in batches(ids()):
                    con.executemany('INSERT INTO fingerprint_ids VALUES(?)', batch)
                    con.commit()
        con.execute('''UPDATE molecules SET status='missing_fingerprint'
          WHERE status='eligible' AND NOT EXISTS
          (SELECT 1 FROM fingerprint_ids f WHERE f.cid=molecules.cid)''')
        con.execute('CREATE INDEX molecule_canonical ON molecules(canonical_smiles)')
        con.execute('CREATE INDEX molecule_scaffold ON molecules(scaffold)')
        # One deterministic ID per exact canonical structure; no score-dependent deduplication.
        con.executescript('''CREATE TABLE sampling_molecules AS
          SELECT m.* FROM molecules m JOIN
          (SELECT MIN(cid) AS cid FROM molecules WHERE status='eligible' GROUP BY canonical_smiles) d
          ON m.cid=d.cid;
          CREATE UNIQUE INDEX sampling_cid ON sampling_molecules(cid);
          CREATE UNIQUE INDEX sampling_canonical ON sampling_molecules(canonical_smiles);
          CREATE INDEX sampling_scaffold ON sampling_molecules(scaffold,cid);
          DROP TABLE fingerprint_ids;''')
        counts = dict(con.execute('SELECT status,COUNT(*) FROM molecules GROUP BY status'))
        counts['unique_eligible_structures'] = con.execute('SELECT COUNT(*) FROM sampling_molecules').fetchone()[0]
        counts['eligible_scaffolds'] = con.execute('SELECT COUNT(DISTINCT scaffold) FROM sampling_molecules').fetchone()[0]
        for k, v in dict(identity=identity, source=str(source), counts=counts,
                         python_version=sys.version, deduplication='lexicographically smallest eligible CID per original canonical structure').items():
            con.execute('INSERT INTO metadata VALUES(?,?)', (k, json.dumps(v, sort_keys=True)))
        con.commit()
        con.close()
        os.replace(temporary, output)
        print(json.dumps(counts, indent=2))
        print('Created reusable molecule index:', output)
    except Exception:
        con.close()
        raise


if __name__ == '__main__':
    main()
