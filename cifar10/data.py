import csv
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T


def read_rows(path, required):
    with Path(path).open(newline='') as f:
        reader = csv.DictReader(f)
        if not set(required).issubset(reader.fieldnames or []):
            raise ValueError(f"{path}: expected columns {required}, found {reader.fieldnames}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path} is empty")
    ids = [r[required[0]] for r in rows]
    if len(set(ids)) != len(ids) or any(not i for i in ids):
        raise ValueError(f"{path}: IDs must be nonempty and unique")
    return rows


def train_rows(config):
    d = config['data']
    rows = read_rows(Path(d['root']) / d['train_csv'], [d['id_column'], d['label_column']])
    for row in rows:
        label = int(row[d['label_column']])
        if not 0 <= label < 10:
            raise ValueError(f"Label outside [0, 9]: {label}")
    return rows


def split_rows(rows, config):
    d = config['data']
    if d['val_fraction'] == 0:
        return rows, []
    rng = np.random.default_rng(d['split_seed'])
    train, val = [], []
    for label in range(10):
        group = sorted([r for r in rows if int(r[d['label_column']]) == label],
                       key=lambda r: r[d['id_column']])
        if len(group) < 2:
            raise ValueError(f"Class {label} needs at least two examples for a stratified split")
        order = rng.permutation(len(group))
        n = min(len(group) - 1, max(1, round(len(group) * d['val_fraction'])))
        val.extend(group[i] for i in order[:n])
        train.extend(group[i] for i in order[n:])
    return train, val


def split_manifest(rows, train, val, config):
    d = config['data']
    # Hash labels and IDs to detect metadata changes on resume.
    canonical = sorted((r[d['id_column']], r[d['label_column']],
                        r.get(d['filename_column'], '')) for r in rows)
    return {'metadata_sha256': hashlib.sha256(json.dumps(canonical).encode()).hexdigest(),
            'train_ids': [r[d['id_column']] for r in train],
            'val_ids': [r[d['id_column']] for r in val]}


def transform(config, training):
    d, a = config['data'], config['augmentation']
    ops = []
    if training:
        if a['crop_padding']:
            ops.append(T.RandomCrop(32, padding=a['crop_padding'], padding_mode='reflect'))
        if a['horizontal_flip']:
            ops.append(T.RandomHorizontalFlip())
        if a['randaugment_ops']:
            ops.append(T.RandAugment(a['randaugment_ops'], a['randaugment_magnitude']))
    ops += [T.ToTensor(), T.Normalize(d['mean'], d['std'])]
    if training and a['random_erasing']:
        ops.append(T.RandomErasing(p=a['random_erasing']))
    return T.Compose(ops)


class ImageDataset(Dataset):
    def __init__(self, rows, config, training=False, test=False):
        self.rows, self.config, self.test = rows, config, test
        self.transform = transform(config, training)
        d = config['data']
        root = Path(d['root']) / d['test_dir' if test else 'train_dir']
        self.paths = []
        for row in rows:
            name = row[d['filename_column']] if d['filename_column'] else d['filename_template'].format(id=row[d['id_column']])
            path = root / name
            if not path.is_file():
                raise FileNotFoundError(f"Missing image: {path}. Check filename_column/template and image directories.")
            self.paths.append(path)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            if image.size != (32, 32):
                raise ValueError(f"Expected 32x32 image: {self.paths[index]} has size {image.size}")
            tensor = self.transform(image.convert('RGB'))
        d = self.config['data']
        target = self.rows[index][d['id_column']] if self.test else int(self.rows[index][d['label_column']])
        return tensor, target


def seed_worker(_worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def loader(dataset, config, training=False, generator=None):
    return DataLoader(dataset, batch_size=config['training' if training else 'inference']['batch_size'],
                      shuffle=training, num_workers=config['data']['num_workers'],
                      worker_init_fn=seed_worker, generator=generator,
                      pin_memory=torch.cuda.is_available())


def check_data(config):
    """Validate metadata, disjoint splits, and image integrity without training."""
    d = config['data']
    rows = train_rows(config)
    train, val = split_rows(rows, config)
    test = read_rows(Path(d['root']) / d['test_csv'], [d['id_column']])
    sample = read_rows(Path(d['root']) / d['sample_submission'], ['id', 'label'])
    if {r[d['id_column']] for r in test} != {r['id'] for r in sample}:
        raise ValueError('Test IDs and sample submission IDs do not match')
    for records, is_test in ((rows, False), (test, True)):
        dataset = ImageDataset(records, config, test=is_test)
        paths = [p.resolve() for p in dataset.paths]
        if len(set(paths)) != len(paths):
            raise ValueError('Different IDs reference the same image path')
        for path in paths:
            with Image.open(path) as image:
                if image.size != (32, 32) or image.mode != 'RGB':
                    raise ValueError(f'Expected 32x32 RGB image: {path}')
                image.verify()
    return {'root': str(Path(d['root']).resolve()), 'training_images': len(rows),
            'train_split': len(train), 'validation_split': len(val), 'test_images': len(test),
            'class_counts': {str(label): sum(int(r[d['label_column']]) == label for r in rows)
                             for label in range(10)},
            'split_seed': d['split_seed'],
            'metadata_sha256': split_manifest(rows, train, val, config)['metadata_sha256'],
            'all_images_verified': True, 'sample_submission_ids_match': True}
