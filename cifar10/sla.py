"""Joint class/transformation labels and self-distillation (Lee et al., ICML 2020).

All four views share one backbone. Layout is rotation-major, then sample-major;
joint output index is class * 4 + rotation. See README.md for the objective.
"""
import torch
from torch import nn
from torch.nn import functional as F
from .attention_resnet import AttentionResNet


class PreActResNet(AttentionResNet):
    def __init__(self, blocks, channels, kernel_size=3, dropout=0., num_classes=10,
                 sla_head=False, rotation_head=False):
        super().__init__(blocks, channels, kernel_size, dropout, num_classes,
                         se_reduction=None, rotation_head=rotation_head)
        self.joint_classifier = nn.Linear(channels[-1], num_classes * 4) if sla_head else None


def four_views(images):
    return torch.cat([images.rot90(k, (-2, -1)) for k in range(4)], dim=0)


def aggregate_logits(joint_logits):
    """Average the class logits assigned to each view's actual transformation."""
    scores = joint_logits.reshape(4, -1, 10, 4)
    return torch.stack([scores[k, :, :, k] for k in range(4)]).mean(0)


def training_loss(model, images, labels, smoothing=0.):
    features = model.features(four_views(images))
    joint = model.joint_classifier(features)
    logits = model.classifier(features[:len(labels)])
    rotations = torch.arange(4, device=labels.device).repeat_interleave(len(labels))
    targets = labels.repeat(4) * 4 + rotations
    joint_loss = F.cross_entropy(joint, targets)
    teacher = aggregate_logits(joint).detach().softmax(1)
    distillation = F.kl_div(logits.log_softmax(1), teacher, reduction='batchmean')
    classification = F.cross_entropy(logits, labels, label_smoothing=smoothing)
    parts = {'joint': joint_loss, 'distillation': distillation, 'classification': classification}
    return sum(parts.values()), logits, parts


def probabilities(model, images, rotation_vote=False):
    if not rotation_vote:
        return model(images).softmax(1)
    features = model.features(four_views(images))
    ordinary = model.classifier(features[:len(images)]).softmax(1)
    aggregated = aggregate_logits(model.joint_classifier(features)).softmax(1)
    return (ordinary + aggregated) / 2


def inference_probabilities(model, images, flip=False, translate=False, rotation_vote=False):
    """Average logits over spatial TTA, then average the two head probabilities."""
    crops = [(images, 1.)]
    if translate:
        padded = F.pad(images, (1, 1, 1, 1), mode='reflect')
        crops = [(images, .5), (padded[:, :, :32, :32], .25),
                 (padded[:, :, 2:, 2:], .25)]
    ordinary = aggregate = 0.
    for crop, weight in crops:
        for view in (crop, crop.flip(-1)) if flip else (crop,):
            factor = weight / (2 if flip else 1)
            if rotation_vote:
                features = model.features(four_views(view))
                logits = model.classifier(features[:len(images)])
                aggregate = aggregate + factor * aggregate_logits(model.joint_classifier(features))
            else:
                logits = model(view)
            ordinary = ordinary + factor * logits
    if rotation_vote:
        return (ordinary.softmax(1) + aggregate.softmax(1)) / 2
    return ordinary.softmax(1)
