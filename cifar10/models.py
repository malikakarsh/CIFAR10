import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, kernel_size, dropout):
        super().__init__()
        pad = kernel_size // 2
        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=pad, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.skip = nn.Identity() if stride == 1 and in_channels == out_channels else nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        return torch.relu(self.residual(x) + self.skip(x))


class ResNet(nn.Module):
    def __init__(self, blocks, channels, kernel_size=3, dropout=0.0, num_classes=10):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, channels[0], 3, padding=1, bias=False),
                                  nn.BatchNorm2d(channels[0]), nn.ReLU(inplace=True))
        layers = []
        previous = channels[0]
        for stage, (count, width) in enumerate(zip(blocks, channels)):
            for block in range(count):
                stride = 2 if stage > 0 and block == 0 else 1
                layers.append(ResidualBlock(previous, width, stride, kernel_size, dropout))
                previous = width
        self.stages = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(previous, num_classes)
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        return self.classifier(self.pool(self.stages(self.stem(x))).flatten(1))


def build_model(config):
    options = dict(config)
    limit = min(options.pop('max_parameters'), 5_000_000)
    architecture = options.pop('architecture', 'resnet')
    if architecture == 'preact_resnet':
        from .sla import PreActResNet
        model = PreActResNet(**options)
    elif architecture == 'attention_resnet':
        from .attention_resnet import AttentionResNet
        model = AttentionResNet(**options)
    elif architecture == 'resnet':
        options.pop('se_reduction', None)
        if options.pop('rotation_head', False):
            raise ValueError('Rotation head requires attention_resnet')
        model = ResNet(**options)
    else:
        raise ValueError(f'Unknown model architecture: {architecture}')
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if count > limit:
        raise ValueError(f"Model has {count:,} trainable parameters; limit is {limit:,}")
    return model, count
