"""Pre-activation residual CNN with channel attention and an optional rotation task.

Method references are documented in docs/attention_resnet.md. This module uses
the project's own model interface; it does not load third-party model weights.
"""
import torch
from torch import nn


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.gate = nn.Sequential(nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.gate(x)


class AttentionBlock(nn.Module):
    def __init__(self, incoming, outgoing, stride, kernel, dropout, reduction):
        super().__init__()
        self.normalize_input = nn.BatchNorm2d(incoming)
        self.first = nn.Conv2d(incoming, outgoing, kernel, stride, kernel // 2, bias=False)
        self.normalize_hidden = nn.BatchNorm2d(outgoing)
        self.dropout = nn.Dropout2d(dropout)
        self.second = nn.Conv2d(outgoing, outgoing, kernel, padding=kernel // 2, bias=False)
        self.attention = ChannelAttention(outgoing, reduction) if reduction else nn.Identity()
        self.projection = (nn.Conv2d(incoming, outgoing, 1, stride, bias=False)
                           if stride != 1 or incoming != outgoing else None)

    def forward(self, x):
        activated = torch.relu(self.normalize_input(x))
        shortcut = x if self.projection is None else self.projection(activated)
        residual = self.first(activated)
        residual = self.second(self.dropout(torch.relu(self.normalize_hidden(residual))))
        return shortcut + self.attention(residual)


class AttentionResNet(nn.Module):
    def __init__(self, blocks, channels, kernel_size=3, dropout=0., num_classes=10,
                 se_reduction=16, rotation_head=False):
        super().__init__()
        self.stem = nn.Conv2d(3, channels[0], 3, padding=1, bias=False)
        incoming = channels[0]
        stages = []
        for stage, (width, count) in enumerate(zip(channels, blocks)):
            for index in range(count):
                stride = 2 if stage > 0 and index == 0 else 1
                stages.append(AttentionBlock(incoming, width, stride, kernel_size, dropout, se_reduction))
                incoming = width
        self.stages = nn.Sequential(*stages)
        self.final_norm = nn.BatchNorm2d(incoming)
        self.classifier = nn.Linear(incoming, num_classes)
        self.rotation_classifier = nn.Linear(incoming, 4) if rotation_head else None
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')

    def features(self, x):
        x = torch.relu(self.final_norm(self.stages(self.stem(x))))
        return x.mean(dim=(-2, -1))

    def forward(self, x):
        return self.classifier(self.features(x))

    def predict_rotation(self, x):
        if self.rotation_classifier is None:
            raise ValueError('Rotation head is disabled')
        return self.rotation_classifier(self.features(x))
