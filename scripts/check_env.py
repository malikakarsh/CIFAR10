"""Check the pinned training environment; optionally require a working MPS device."""
import argparse
import importlib.metadata as metadata
import json
from pathlib import Path
import platform
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--require-mps', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    versions, failures = {}, []
    for line in (root / 'requirements.txt').read_text().splitlines():
        if not line or line.startswith(('#', '-')):
            continue
        name, expected = line.split('==')
        try:
            actual = metadata.version(name)
            versions[name] = actual
            if actual != expected:
                failures.append(f'{name}: expected {expected}, found {actual}')
        except metadata.PackageNotFoundError:
            failures.append(f'{name}: missing')
    if failures:
        raise SystemExit('\n'.join(failures))

    import numpy
    import PIL
    import torch
    import torchvision
    import yaml
    import cifar10

    mps = torch.backends.mps.is_available()
    report = {'python': platform.python_version(), 'executable': sys.executable,
              'architecture': platform.machine(), 'platform': platform.platform(),
              'in_virtualenv': sys.prefix != sys.base_prefix,
              'packages': versions, 'torch_location': torch.__file__,
              'mps_built': torch.backends.mps.is_built(), 'mps_available': mps}
    if mps:
        device = torch.device('mps')
        model = torch.nn.Conv2d(3, 4, 3, padding=1).to(device)
        loss = model(torch.randn(2, 3, 32, 32, device=device)).square().mean()
        loss.backward()
        torch.mps.synchronize()
        report['mps_forward_backward'] = 'passed'
    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + '\n')
    if args.require_mps and not mps:
        raise SystemExit('Training dependencies are ready, but MPS is unavailable to this process.')


if __name__ == '__main__':
    main()
