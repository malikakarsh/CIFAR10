"""Three-stage grouped bottleneck residual CNN, following the ResNeXt design.

Independently implemented for this project's SLA feature/head interface.
No pretrained weights are loaded. Method: https://arxiv.org/abs/1611.05431.
"""
import torch
from torch import nn


class GroupedBottleneck(nn.Module):
    def __init__(self, incoming, channels, stride, groups, base_width, dropout):
        super().__init__()
        hidden = max(1, int(channels * base_width / 64)) * groups
        outgoing = channels * 4
        self.residual = nn.Sequential(
            nn.Conv2d(incoming, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, stride, 1, groups=groups, bias=False),
            nn.BatchNorm2d(hidden), nn.ReLU(inplace=True), nn.Dropout2d(dropout),
            nn.Conv2d(hidden, outgoing, 1, bias=False), nn.BatchNorm2d(outgoing),
        )
        self.shortcut = (nn.Identity() if stride == 1 and incoming == outgoing else
                         nn.Sequential(nn.Conv2d(incoming, outgoing, 1, stride, bias=False),
                                       nn.BatchNorm2d(outgoing)))

    def forward(self, images):
        return torch.relu(self.residual(images) + self.shortcut(images))


class GroupedResNet(nn.Module):
    def __init__(self, blocks, channels, kernel_size=3, dropout=0., num_classes=10,
                 sla_head=False, groups=8, base_width=14):
        super().__init__()
        if kernel_size != 3:
            raise ValueError('Grouped bottleneck model requires kernel_size=3')
        incoming = channels[0]
        self.stem = nn.Sequential(nn.Conv2d(3, incoming, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(incoming), nn.ReLU(inplace=True))
        stages = []
        for stage, (width, count) in enumerate(zip(channels, blocks)):
            for index in range(count):
                stride = 2 if stage > 0 and index == 0 else 1
                stages.append(GroupedBottleneck(incoming, width, stride, groups, base_width, dropout))
                incoming = width * 4
        self.stages = nn.Sequential(*stages)
        self.classifier = nn.Linear(incoming, num_classes)
        self.joint_classifier = nn.Linear(incoming, num_classes * 4) if sla_head else None
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')

    def features(self, images):
        return self.stages(self.stem(images)).mean(dim=(-2, -1))

    def forward(self, images):
        return self.classifier(self.features(images))
