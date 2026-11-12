import argparse
import json
from pathlib import Path

import torch

from cifar10.config import load_config
from cifar10.sla import PreActResNet


EXPECTED_PARAMETERS = 4_956_062
PARAMETER_LIMIT = 5_000_000

# Use the same resolved configuration schema as the training CLI.
CONFIG = load_config(Path(__file__).resolve().parent / "configs/final_model.yaml")

RESULTS = {
    "selected_epoch": 588,
    "validation_accuracy_percent": 98.92,
    "validation_cross_entropy": 0.0862,
    "public_accuracy_percent": 97.20,
    "trainable_parameters": EXPECTED_PARAMETERS,
}


class FinalModel(PreActResNet):
    """Shared pre-activation backbone with class, joint, and rotation heads."""

    def __init__(self):
        settings = CONFIG["model"]
        super().__init__(
            blocks=settings["blocks"],
            channels=settings["channels"],
            kernel_size=settings["kernel_size"],
            dropout=settings["dropout"],
            num_classes=settings["num_classes"],
            sla_head=settings["sla_head"],
            rotation_head=settings["rotation_head"],
        )

    def forward_heads(self, images):
        """Return all three heads from a single backbone evaluation."""
        features = self.features(images)
        return {
            "classification": self.classifier(features),
            "joint": self.joint_classifier(features),
            "rotation": self.rotation_classifier(features),
        }


def build_model():
    model = FinalModel()
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if count != EXPECTED_PARAMETERS or count > PARAMETER_LIMIT:
        raise ValueError(f"Unexpected trainable parameter count: {count:,}")
    return model


def main():
    parser = argparse.ArgumentParser(description="Inspect the final model and configuration.")
    parser.add_argument("--show-config", action="store_true")
    args = parser.parse_args()
    if args.show_config:
        print(json.dumps({"config": CONFIG, "results": RESULTS}, indent=2))
        return
    model = build_model().eval()
    with torch.inference_mode():
        outputs = model.forward_heads(torch.zeros(2, 3, 32, 32))
    print(f"Trainable parameters: {EXPECTED_PARAMETERS:,} / {PARAMETER_LIMIT:,}")
    for name, logits in outputs.items():
        print(f"{name}: {tuple(logits.shape)}")


if __name__ == "__main__":
    main()
