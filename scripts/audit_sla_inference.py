"""Compare checkpoint inference policies on their shared, never-trained-on IDs."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).absolute().parents[1]))
import torch
from torch.nn import functional as F
from cifar10.data import ImageDataset, loader, train_rows, split_manifest
from cifar10.engine import device_for
from cifar10.models import build_model
from cifar10.sla import four_views, aggregate_logits


def common_validation(states):
    common = set.intersection(*(set(s['split']['val_ids']) for s in states))
    if not common:
        raise ValueError('Checkpoints have no common validation IDs')
    for state in states:
        if common & set(state['split']['train_ids']):
            raise ValueError('Validation IDs overlap checkpoint training IDs')
    return common


@torch.inference_mode()
def policies(model, images):
    # Reuse rotation-zero features for the ordinary head; BN is in eval mode.
    padded = F.pad(images, (1, 1, 1, 1), mode='reflect')
    head = joint = mirror_head = mirror_joint = 0.
    ordinary = None
    for index, (crop, weight) in enumerate([(images, .5),
            (padded[:, :, :32, :32], .25), (padded[:, :, 2:, 2:], .25)]):
        for flip in (False, True):
            view = crop.flip(-1) if flip else crop
            features = model.features(four_views(view))
            logits = model.classifier(features[:len(images)])
            vote = aggregate_logits(model.joint_classifier(features))
            if index == 0:
                mirror_head = mirror_head + logits / 2
                mirror_joint = mirror_joint + vote / 2
                if not flip:
                    ordinary = logits.softmax(1)
            head = head + weight * logits / 2
            joint = joint + weight * vote / 2
    return {'head_only': ordinary,
            'head_mirror': mirror_head.softmax(1),
            'head_mirror_translation': head.softmax(1),
            'joint_mirror_translation': joint.softmax(1),
            'combined_mirror': (mirror_head.softmax(1) + mirror_joint.softmax(1)) / 2,
            'combined_mirror_translation': (head.softmax(1) + joint.softmax(1)) / 2}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoints', nargs='+', required=True)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--data-root', default='dataset')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--output', default='docs/sla_inference_audit.json')
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    device = device_for(args.device)
    states = [torch.load(p, map_location='cpu', weights_only=True) for p in args.checkpoints]
    common = common_validation(states)
    result = {'validation_images': len(common),
              'common_ids_sha256': hashlib.sha256(json.dumps(sorted(common)).encode()).hexdigest(),
              'training_overlap': False, 'device': str(device), 'checkpoints': []}
    print(f'Auditing {len(common)} shared validation images; no training overlap.', flush=True)
    for path, state in zip(args.checkpoints, states):
        config = state['config']
        config['data'].update(root=args.data_root, num_workers=0)
        config['inference']['batch_size'] = args.batch_size
        rows = train_rows(config)
        assert split_manifest(rows, [], [], config)['metadata_sha256'] == state['split']['metadata_sha256']
        selected = [r for r in rows if r[config['data']['id_column']] in common]
        model, count = build_model(config['model'])
        assert model.joint_classifier is not None
        model.load_state_dict(state['model'])
        model.to(device).eval()
        batches = loader(ImageDataset(selected, config), config)
        scores = {}
        started = time.monotonic()
        for index, (images, labels) in enumerate(batches, 1):
            images, labels = images.to(device), labels.to(device)
            for name, probabilities in policies(model, images).items():
                score = scores.setdefault(name, {'correct': 0, 'loss_sum': 0.})
                score['correct'] += (probabilities.argmax(1) == labels).sum().item()
                score['loss_sum'] += F.nll_loss(probabilities.clamp_min(1e-30).log(), labels,
                                              reduction='sum').item()
            if index == 1 or index % 5 == 0 or index == len(batches):
                print(f'{Path(path).parent.name}: {index}/{len(batches)} batches; '
                      f'{time.monotonic()-started:.1f}s', flush=True)
        metrics = {name: {'correct': s['correct'], 'accuracy': s['correct']/len(common),
                         'loss': s['loss_sum']/len(common)} for name, s in scores.items()}
        result['checkpoints'].append({'checkpoint': path, 'epoch': state['epoch']+1,
                                      'parameters': count, 'policies': metrics})
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(metrics, indent=2), flush=True)
        del model
    print(f'Saved {args.output}', flush=True)


if __name__ == '__main__':
    main()
