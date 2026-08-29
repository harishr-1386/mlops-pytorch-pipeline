from torch import nn
from torchvision import models


CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def get_model(architecture: str = "resnet18", num_classes: int = 10) -> nn.Module:
    """
    Build a torchvision model adapted for CIFAR-10 (32x32 inputs).

    ResNet-18 is designed for ImageNet (224x224). For CIFAR-10 we replace
    the first conv layer (kernel 7, stride 2) with a smaller kernel (3, stride 1)
    and remove the max-pool, preserving spatial resolution through early layers.
    """
    if architecture == "resnet18":
        model = models.resnet18(weights=None, num_classes=num_classes)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        return model

    if architecture == "resnet34":
        model = models.resnet34(weights=None, num_classes=num_classes)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        return model

    raise ValueError(f"Unsupported architecture: {architecture!r}")
