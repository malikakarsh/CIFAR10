"""Audit permitted image statistics and submission integrity without test labels."""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).absolute().parents[1]))
import numpy as np
from PIL import Image
from cifar10.config import load_config
from cifar10.data import ImageDataset, read_rows, split_rows, train_rows


def image_statistics(dataset):
    values = []
    formats, sizes, modes = Counter(), Counter(), Counter()
    for index, path in enumerate(dataset.paths):
        with Image.open(path) as image:
            formats[image.format] += 1
            sizes[str(image.size)] += 1
            modes[image.mode] += 1
            pixels = np.asarray(image.convert('RGB'), dtype=np.float64) / 255
        values.append(np.r_[pixels.mean((0, 1)), pixels.std((0, 1)),
                            np.abs(np.diff(pixels, axis=0)).mean(),
                            np.abs(np.diff(pixels, axis=1)).mean()])
        if (index + 1) % 10000 == 0:
            print(f'Image audit: {index + 1}/{len(dataset.paths)}', flush=True)
    matrix = np.asarray(values)
    return {'images': len(values), 'formats': dict(formats), 'sizes': dict(sizes),
            'modes': dict(modes), 'features': [
                'mean_red', 'mean_green', 'mean_blue', 'std_red', 'std_green',
                'std_blue', 'vertical_edge', 'horizontal_edge'],
            'feature_mean': matrix.mean(0).tolist(),
            'feature_std': matrix.std(0).tolist()}


def audit_csv(path, expected_ids):
    with Path(path).open(newline='') as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ['id', 'label']:
            raise ValueError(f'{path}: expected id,label columns')
        rows = list(reader)
    ids = [r['id'] for r in rows]
    if ids != expected_ids:
        raise ValueError(f'{path}: submission IDs/order differ from sample')
    labels = [int(r['label']) for r in rows]
    if any(label < 0 or label > 9 for label in labels):
        raise ValueError(f'{path}: labels outside 0..9')
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    metadata_path = Path(path).with_suffix('.metadata.json')
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else None
    if metadata and metadata.get('csv_sha256') != digest:
        raise ValueError(f'{path}: recorded CSV hash does not match')
    return {'rows': len(rows), 'ids_and_order_match': True, 'csv_sha256': digest,
            'metadata_hash_matches': True if metadata else None,
            'predicted_class_counts': dict(sorted(Counter(labels).items())),
            'checkpoint_epoch': metadata.get('checkpoint_epoch') if metadata else None}, labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/hpc_a100/sla600.yaml')
    parser.add_argument('--data-root', default='dataset')
    parser.add_argument('--submissions', nargs='+', required=True)
    parser.add_argument('--output', default='docs/submission_gap_audit.json')
    args = parser.parse_args()
    config = load_config(args.config)
    config['data'].update(root=args.data_root, cache_images=False)
    records = train_rows(config)
    training, validation = split_rows(records, config)
    d = config['data']
    test = read_rows(Path(args.data_root) / d['test_csv'], [d['id_column']])
    sample = read_rows(Path(args.data_root) / d['sample_submission'], ['id', 'label'])
    expected_ids = [r['id'] for r in sample]
    if {r[d['id_column']] for r in test} != set(expected_ids):
        raise ValueError('Test metadata IDs differ from sample')
    result = {'scope': 'Aggregate pixel statistics and CSV integrity only; no hidden labels, '
              'external data, source matching, or evaluation images used for training.',
              'images': {}, 'submissions': {}, 'pairwise_disagreements': {}}
    for name, rows, is_test in [('training', training, False),
                              ('validation', validation, False), ('test', test, True)]:
        print(f'Auditing {name}: {len(rows)} images', flush=True)
        result['images'][name] = image_statistics(ImageDataset(rows, config, test=is_test))
    # Standardized mean differences are descriptive; they do not establish causality.
    for name in ['validation', 'test']:
        a, b = result['images']['training'], result['images'][name]
        scale = np.sqrt((np.square(a['feature_std']) + np.square(b['feature_std'])) / 2)
        b['standardized_mean_difference_from_training'] = (
            (np.array(b['feature_mean']) - a['feature_mean']) / np.maximum(scale, 1e-12)).tolist()
    predictions = {}
    for path in args.submissions:
        result['submissions'][path], predictions[path] = audit_csv(path, expected_ids)
    paths = list(predictions)
    for i, first in enumerate(paths):
        for second in paths[i + 1:]:
            count = sum(a != b for a, b in zip(predictions[first], predictions[second]))
            result['pairwise_disagreements'][f'{first} vs {second}'] = {
                'images': count, 'fraction': count / len(expected_ids)}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(f'Saved {output}', flush=True)


if __name__ == '__main__':
    main()
