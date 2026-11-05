#!/usr/bin/env python3
# Software verified: Python 3.9.6; PyTorch 2.8.0 (checkpoint/model inference).
# Uses the existing train_dnn_morgan evaluator: NumPy 1.26.4, SciPy 1.13.1,
# scikit-learn 1.6.1 and matplotlib 3.9.4 (existing helper dependencies).
# No training is performed.
"""Evaluate a saved FastBindRank checkpoint on independent external inputs."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import torch
from train_dnn_morgan import MLP, make_loader, evaluate


def sha256(path):
    """Hash input bytes for reproducible evaluation and checkpoint preservation."""
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def validate_inputs(fp, labels, args):
    """Reject duplicate IDs, invalid bits and missing/invalid labels; never drop rows."""
    ids = set()
    with fp.open() as handle:
        for number, line in enumerate(handle, 1):
            fields = line.strip().split(',')
            cid = fields[0]
            if not cid or cid in ids:
                raise ValueError(f'Empty/duplicate fingerprint ID at line {number}')
            ids.add(cid)
            indices = [int(value) - int(args.one_based_indices) for value in fields[1:]]
            if any(value < 0 or value >= args.bits for value in indices):
                raise ValueError(f'Fingerprint bit outside checkpoint dimensions at line {number}')
    if not ids:
        raise ValueError('Fingerprint file is empty')
    seen = set()
    with labels.open() as handle:
        reader = csv.DictReader(handle, delimiter='\t')
        if not {args.id_col, args.label_col}.issubset(reader.fieldnames or []):
            raise ValueError(f'Labels require columns {args.id_col} and {args.label_col}')
        for row in reader:
            cid = row[args.id_col]
            value = float(row[args.label_col])
            if not cid or cid in seen or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f'Duplicate/empty ID or invalid probability label: {cid}')
            seen.add(cid)
    missing = ids - seen
    if missing:
        raise ValueError(f'{len(missing)} fingerprint IDs have no label')
    return len(ids)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--test-fp', type=Path, required=True,
                        help='Headerless sparse Morgan CSV: ID,bit_index,bit_index,...')
    parser.add_argument('--test-labels', type=Path, required=True,
                        help='Required tab-separated labels; all fingerprint IDs must be labeled')
    parser.add_argument('--output-dir', type=Path, required=True,
                        help='New or empty independent directory')
    parser.add_argument('--id-col', help='Defaults to checkpoint id_col, otherwise CID')
    parser.add_argument('--label-col', help='Defaults to checkpoint label_col')
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    cli = parser.parse_args()
    if cli.batch_size < 1:
        parser.error('--batch-size must be positive')
    checkpoint, fp, out = [p.resolve() for p in [cli.checkpoint, cli.test_fp, cli.output_dir]]
    labels = cli.test_labels.resolve()
    for path in [checkpoint, fp, labels]:
        if not path.is_file():
            raise FileNotFoundError(path)
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ValueError('Output directory must be new or empty; choose an independent directory')
    checkpoint_hash = sha256(checkpoint)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    config = saved['args']
    dims = saved['hidden_dims']
    if isinstance(dims, str):
        dims = [int(value) for value in dims.split(',')]
    args = SimpleNamespace(bits=int(saved['bits']),
        one_based_indices=config.get('one_based_indices', False),
        id_col=cli.id_col or config.get('id_col', 'CID'),
        label_col=cli.label_col or config.get('label_col', 'affinity_probability_binary'),
        hard_label=config.get('hard_label', False), threshold=config.get('threshold', 0.5),
        batch_size=cli.batch_size, num_workers=0)
    count = validate_inputs(fp, labels, args)
    device = torch.device(cli.device)
    model = MLP(args.bits, dims, config['dropout']).to(device)
    model.load_state_dict(saved['model'], strict=True)
    del saved
    dataset, loader = make_loader(str(fp), str(labels), args, False)
    assert len(dataset) == count
    model.eval()
    out.mkdir(parents=True, exist_ok=True)
    with (out/'external_test_pred_out.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['ligand_id','label','prob'])
        with torch.inference_mode():
            for ids, features, target in loader:
                probabilities = torch.sigmoid(model(features.to(device))).cpu().tolist()
                for i, cid in enumerate(ids):
                    writer.writerow([cid, target[i].item(), probabilities[i]])
    values = evaluate(model, loader, device, torch.nn.BCEWithLogitsLoss(),
                      threshold=args.threshold, hard_label=args.hard_label, plot=True,
                      out_prefix=str(out/'external_test'))
    if args.hard_label:
        metrics = dict(zip(['loss','accuracy','f1','roc_auc'], values[:4]))
    else:
        metrics = dict(zip(['loss','brier','pearson','spearman','soft_pr_auc'], values[:5]))
        metrics.update({f'prec_at_{k}':v for k,v in values[5].items()})
    with (out/'external_test_metrics.txt').open('w') as handle:
        handle.write('metric\tvalue\n')
        for name, value in metrics.items():
            formatted = f'{value:.6f}' if value is not None else 'NA'
            handle.write(f'{name}\t{formatted}\n')
    assert sha256(checkpoint) == checkpoint_hash, 'Checkpoint changed during evaluation'
    metadata = dict(checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_hash,
        test_fp=str(fp), test_fp_sha256=sha256(fp),
        test_labels=str(labels),
        test_labels_sha256=sha256(labels),
        n_compounds=count, mean_label=float(dataset.y.mean(dtype='float64')), device=str(device), torch_version=str(torch.__version__),
        model_config=dict(bits=args.bits, hidden_dims=dims, dropout=config['dropout'],
                          one_based_indices=args.one_based_indices, id_col=args.id_col,
                          label_col=args.label_col, hard_label=args.hard_label, threshold=args.threshold),
        evaluation_precision='float32; AMP disabled', no_training=True)
    (out/'evaluation_metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(f'Evaluated {count} compounds on {device}; results: {out}')


if __name__ == '__main__':
    main()
