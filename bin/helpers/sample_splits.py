#!/usr/bin/env python3

import argparse
import csv
import fcntl
import json
import os
import shutil
from pathlib import Path
import random
import sqlite3
import sys

from locked_command import check_lock
from build_molecule_index import sha256, structure_record, RULES, rdBase

FIELDS = ['cid', 'original_smiles', 'canonical_smiles', 'scaffold', 'status']
PARTS = ['training', 'tuning', 'test']


def read_records(path):
    """Read the public two-column SMILES/ID format, rejecting duplicate IDs."""
    records = []
    seen = set()
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            fields = line.split()
            if len(fields) != 2 or fields[1] in seen:
                raise ValueError(f'Expected unique SMILES/ID records in {path}')
            seen.add(fields[1])
            records.append((fields[1], fields[0]))
    return records


def reservoir(rows, size, rng):
    """Uniformly sample a bounded number of records; return them in randomized order."""
    chosen = []
    count = 0
    for count, row in enumerate(rows, 1):
        if len(chosen) < size:
            chosen.append(row)
        else:
            j = rng.randrange(count)
            if j < size:
                chosen[j] = row
    if count < size:
        raise ValueError(f'Insufficient eligible units: requested {size}, available {count}')
    rng.shuffle(chosen)
    return chosen


def representatives(rows, rng):
    """For scaffold-sorted rows, uniformly select one eligible molecule from each group."""
    group = None
    selected = None
    n = 0
    for row in rows:
        if row['scaffold'] != group:
            if selected is not None:
                yield selected
            group, selected, n = row['scaffold'], row, 1
        else:
            n += 1
            if rng.randrange(n) == 0:
                selected = row
    if selected is not None:
        yield selected


def write_summary(root):
    """Rebuild the non-authoritative ID summary from committed iterations atomically."""
    pending = root / '.used_compound_ids.pending.tsv'
    directories = sorted(root.glob('iteration_*/splits'), key=lambda d: int(d.parent.name.split('_')[-1]))
    with pending.open('w') as out:
        for directory in directories:
            number = int(directory.parent.name.split('_')[-1])
            with (directory / 'membership.txt').open() as source:
                for row in csv.DictReader(source, delimiter='\t'):
                    if number == 1 or row['partition'] == 'training':
                        out.write(f"{number}\t{row['cid']}\n")
    os.replace(pending, root / 'used_compound_ids.tsv')


def main():
    p = argparse.ArgumentParser(description='Reproducible, structure-aware iteration sampling')
    p.add_argument('--lock-fd', type=int, help=argparse.SUPPRESS)
    p.add_argument('--index', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--results-dir', required=True)
    p.add_argument('--sampling-strategy', choices=['scaffold', 'random'], default='scaffold')
    p.add_argument('--sampling-pool', help='Optional SMILES/ID subset; otherwise full indexed library')
    p.add_argument('--test-set', help='Frozen shared SMILES/ID test set; otherwise created once on iteration 1')
    p.add_argument('--training-size', type=int, required=True)
    p.add_argument('--tuning-size', type=int, required=True)
    p.add_argument('--test-size', type=int, required=True)
    p.add_argument('--iteration', type=int, required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--exclude-ids-file')
    a = p.parse_args()
    if min(a.training_size, a.tuning_size, a.test_size, a.iteration) < 1:
        p.error('Sizes and iteration must be positive')
    root = Path(a.results_dir).resolve()
    out = Path(a.output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if a.lock_fd is not None:
        check_lock(a.lock_fd, root / '.sampling.lock')
        prepare(a, root, out)
        return
    with (root / '.sampling.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prepare(a, root, out)


def prepare(a, root, out):
    """Validate frozen history, sample new members, audit isolation, and save immutable membership."""
    con = sqlite3.connect(':memory:', uri=True)
    con.row_factory = sqlite3.Row
    con.execute('PRAGMA temp_store=FILE')
    index = Path(a.index).resolve()
    con.execute('ATTACH DATABASE ? AS lib', (f'file:{index}?mode=ro',))
    identity = json.loads(con.execute('SELECT value FROM lib.metadata WHERE key="identity"').fetchone()[0])
    if identity['rules'] != RULES or identity['rdkit_version'] != rdBase.rdkitVersion or identity['schema_version'] != 1:
        raise ValueError('Index schema/rules/RDKit version differs from the current implementation')
    config = dict(index=str(index), index_identity=identity, strategy=a.sampling_strategy,
                  training_size=a.training_size, tuning_size=a.tuning_size, test_size=a.test_size,
                  seed=a.seed)
    # Optional paths may be omitted on later iterations; their recorded content identity remains fixed.
    settings = root / 'sampling_manifest.json'
    saved = json.loads(settings.read_text()) if settings.exists() else None
    for name, value in [('sampling_pool', a.sampling_pool), ('test_set', a.test_set), ('exclude_ids_file', a.exclude_ids_file)]:
        if value:
            path = Path(value).resolve()
            config[name] = dict(path=str(path), sha256=sha256(path))
        else:
            config[name] = saved['config'][name] if saved else None
    if saved and saved['config'] != config:
        raise ValueError('Sampling configuration differs from this results directory; use a separate results directory')
    for name in ['sampling_pool', 'test_set', 'exclude_ids_file']:
        item = config[name]
        if item and sha256(item['path']) != item['sha256']:
            raise ValueError(f'Frozen input changed: {item["path"]}')
    history = []
    completed = {}
    for directory in root.glob('iteration_*/splits'):
        number = int(directory.parent.name.split('_')[-1])
        metadata = directory / 'sampling_metadata.json'
        existing = [directory / f'{part}_split.txt' for part in PARTS]
        if not metadata.exists():
            if any(f.exists() for f in existing):
                raise ValueError(f'Incomplete/unregistered sampling output: {directory}')
            continue
        info = json.loads(metadata.read_text())
        if not saved or info['config'] != config:
            raise ValueError(f'Unregistered or incompatible sampling history: {directory}')
        for filename, digest in info['files'].items():
            if sha256(directory / filename) != digest:
                raise ValueError(f'Frozen output changed: {directory / filename}')
        completed[number] = directory
        with (directory / 'membership.txt').open() as f:
            history.extend(dict(r, iteration=number) for r in csv.DictReader(f, delimiter='\t'))
    if a.iteration in completed:
        if completed[a.iteration] != out:
            raise ValueError('Output directory is inconsistent with existing iteration')
        write_summary(root)
        print('Reusing verified splits:', out)
        return
    if sorted(completed) != list(range(1, a.iteration)):
        raise ValueError('Prepare iterations sequentially without gaps')
    if out != root / f'iteration_{a.iteration}' / 'splits':
        raise ValueError('Output must be <results-dir>/iteration_<N>/splits')
    if not saved:
        pending_manifest = root / '.sampling_manifest.pending.json'
        pending_manifest.write_text(json.dumps(dict(config=config), indent=2, sort_keys=True) + '\n')
        os.replace(pending_manifest, settings)
    pending = out.parent / '.splits_pending'
    if pending.exists():
        shutil.rmtree(pending)
    # Only this private staging directory is disposable; committed files are never repaired silently.
    pending.mkdir(parents=True)
    con.executescript('''CREATE TEMP TABLE blocked_canonical(value TEXT PRIMARY KEY);
      CREATE TEMP TABLE blocked_scaffold(value TEXT PRIMARY KEY);
      CREATE TEMP TABLE pool_ids(cid TEXT PRIMARY KEY);''')

    def resolve(cid, smiles=None):
        row = con.execute('SELECT * FROM lib.molecules WHERE cid=?', (cid,)).fetchone()
        if row is None or row['status'] != 'eligible':
            raise ValueError(f'Missing or ineligible indexed compound: {cid}')
        if smiles is not None and structure_record(f'{smiles}\t{cid}')[2] != row['canonical_smiles']:
            raise ValueError(f'SMILES differs from library for {cid}')
        return dict(row)

    if config['sampling_pool']:
        for cid, smiles in read_records(config['sampling_pool']['path']):
            row = resolve(cid, smiles)
            # Use one ID per canonical within the custom pool, rather than losing non-global representative IDs.
            con.execute('INSERT INTO pool_ids VALUES(?)', (cid,))
        con.execute('''CREATE TEMP TABLE candidates AS SELECT m.* FROM lib.molecules m JOIN
          (SELECT MIN(m.cid) cid FROM lib.molecules m JOIN pool_ids p ON m.cid=p.cid
           GROUP BY m.canonical_smiles) d ON m.cid=d.cid''')
        con.execute('CREATE INDEX candidate_scaffold ON candidates(scaffold,cid)')
        table = 'candidates'
    else:
        table = 'lib.sampling_molecules'

    def block(rows, scaffolds=False):
        con.executemany('INSERT OR IGNORE INTO blocked_canonical VALUES(?)', ((r['canonical_smiles'],) for r in rows))
        if scaffolds:
            con.executemany('INSERT OR IGNORE INTO blocked_scaffold VALUES(?)', ((r['scaffold'],) for r in rows))

    if config['exclude_ids_file']:
        with open(config['exclude_ids_file']['path']) as f:
            for line in f:
                if line.strip():
                    cid = line.split()[-1]
                    row = con.execute('SELECT canonical_smiles FROM lib.molecules WHERE cid=?', (cid,)).fetchone()
                    if row and row[0]:
                        con.execute('INSERT OR IGNORE INTO blocked_canonical VALUES(?)', (row[0],))
    def draw(size, salt, one_per_scaffold=False):
        rng = random.Random(a.seed + a.iteration * 1000003 + salt)
        rows = con.execute(f'''SELECT m.* FROM {table} m WHERE
          NOT EXISTS (SELECT 1 FROM blocked_canonical b WHERE b.value=m.canonical_smiles)
          AND NOT EXISTS (SELECT 1 FROM blocked_scaffold b WHERE b.value=m.scaffold)
          ORDER BY m.scaffold,m.cid''')
        stream = representatives(rows, rng) if one_per_scaffold or a.sampling_strategy == 'scaffold' else rows
        result = [dict(r) for r in reservoir(stream, size, rng)]
        block(result, one_per_scaffold or a.sampling_strategy == 'scaffold')
        return result

    if config['test_set']:
        test = [resolve(cid, smiles) for cid, smiles in read_records(config['test_set']['path'])]
    elif a.iteration > 1:
        test = [resolve(cid, smiles) for cid, smiles in read_records(completed[1] / 'test_split.txt')]
    else:
        # Both strategies use a one-per-scaffold test; pass this file to subsequent experiment roots.
        test = draw(a.test_size, 11, one_per_scaffold=True)
    if len(test) != a.test_size or len({r['canonical_smiles'] for r in test}) != len(test):
        raise ValueError('Test size mismatch or duplicate canonical test structures')
    if config['test_set']:
        with open(config['exclude_ids_file']['path']) if config['exclude_ids_file'] else open('/dev/null') as f:
            for line in f:
                if line.strip():
                    row = con.execute('SELECT canonical_smiles FROM lib.molecules WHERE cid=?', (line.split()[-1],)).fetchone()
                    if row and any(r['canonical_smiles'] == row[0] for r in test):
                        raise ValueError('Test set contains an explicitly excluded structure')
    if len({r['scaffold'] for r in test}) != len(test):
        raise ValueError('The shared test set requires one molecule per scaffold')
    block(test, scaffolds=True)
    # Previous train/tuning are excluded by exact structure for both strategies, by scaffold only for scaffold.
    block([r for r in history if r['partition'] != 'test'], a.sampling_strategy == 'scaffold')
    if a.iteration == 1:
        tuning = draw(a.tuning_size, 17)
    else:
        tuning = [resolve(cid, smiles) for cid, smiles in read_records(completed[1] / 'tuning_split.txt')]
    training = draw(a.training_size, 29)
    parts = dict(training=training, tuning=tuning, test=test)
    # Validate the complete experiment, counting the fixed tuning/test only once.
    all_train = [r for r in history if r['partition'] == 'training'] + training
    groups = [all_train, tuning, test]
    for rows in groups:
        if len(rows) != len({r['cid'] for r in rows}) or len(rows) != len({r['canonical_smiles'] for r in rows}):
            raise ValueError('Duplicate ID or canonical structure within an experiment partition')
        if a.sampling_strategy == 'scaffold' and len(rows) != len({r['scaffold'] for r in rows}):
            raise ValueError('Repeated scaffold within scaffold-sampled partition')
    audit = {}
    for i, left in enumerate(groups):
        for j in range(i + 1, 3):
            right = groups[j]
            key = f'{PARTS[i]}_{PARTS[j]}'
            id_overlap = len({r['cid'] for r in left} & {r['cid'] for r in right})
            canonical_overlap = len({r['canonical_smiles'] for r in left} & {r['canonical_smiles'] for r in right})
            scaffold_overlap = len({r['scaffold'] for r in left} & {r['scaffold'] for r in right})
            audit[key] = dict(id_overlap=id_overlap, canonical_overlap=canonical_overlap, scaffold_overlap=scaffold_overlap)
            if id_overlap or canonical_overlap or (scaffold_overlap and (j == 2 or a.sampling_strategy == 'scaffold')):
                raise ValueError(f'Split isolation failed: {key}')
    for part, rows in parts.items():
        (pending / f'{part}_split.txt').write_text(''.join(f'{r["original_smiles"]}\t{r["cid"]}\n' for r in rows))
    with (pending / 'membership.txt').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=['partition'] + FIELDS, delimiter='\t')
        writer.writeheader()
        for part, rows in parts.items():
            writer.writerows(dict(r, partition=part) for r in rows)
    files = {f.name: sha256(f) for f in pending.glob('*.txt')}
    info = dict(config=config, iteration=a.iteration, python=sys.version, audit=audit, files=files,
                sizes={p: len(v) for p, v in parts.items()}, status='complete')
    (pending / 'sampling_metadata.json').write_text(json.dumps(info, indent=2, sort_keys=True) + '\n')
    os.replace(pending, out)
    write_summary(root)
    print(json.dumps(dict(sizes=info['sizes'], audit=audit), indent=2))
    con.close()


if __name__ == '__main__':
    main()
