"""Optional NPR feature branch for CO-SPY fusion v3."""

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms

from utils import data_augment

from .artifact import AlignedCrop
from .base import BaseDetector


class _Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, planes * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class _NPRResNet(nn.Module):
    """Compact NPR encoder matching the released CO-SPY/ProGAN layout."""

    def __init__(self, layers=(3, 4, 6, 3)):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=2, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, layers[0])
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def _make_layer(self, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * _Bottleneck.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * _Bottleneck.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * _Bottleneck.expansion),
            )
        layers = [_Bottleneck(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * _Bottleneck.expansion
        layers.extend(_Bottleneck(self.inplanes, planes) for _ in range(1, blocks))
        return nn.Sequential(*layers)

    @staticmethod
    def _interpolate(image, factor):
        down = F.interpolate(
            image,
            scale_factor=factor,
            mode="nearest",
            recompute_scale_factor=True,
        )
        return F.interpolate(
            down,
            scale_factor=1 / factor,
            mode="nearest",
            recompute_scale_factor=True,
        )

    def forward(self, x):
        artifact = (x - self._interpolate(x, 0.5)) * (2.0 / 3.0)
        x = self.maxpool(self.relu(self.bn1(self.conv1(artifact))))
        x = self.layer2(self.layer1(x))
        x = self.avgpool(x)
        return x.flatten(1)


class NPRDetector(BaseDetector):
    """ResNet artifact detector based on neighboring-pixel relationships."""

    def __init__(self, dim_artifact: int = 512, num_classes: int = 1):
        super().__init__()
        self.artifact_encoder = _NPRResNet()
        self.fc = torch.nn.Linear(dim_artifact, num_classes)
        self._build_transforms()

    def _build_transforms(self):
        self.mean = [0.0, 0.0, 0.0]
        self.std = [1.0, 1.0, 1.0]
        self.loadSize = 256
        self.cropSize = 224

        self.aug_config = {
            "blur_prob": 0.0,
            "blur_sig": [0.0, 3.0],
            "jpg_prob": 0.5,
            "jpg_method": ["cv2", "pil"],
            "jpg_qual": list(range(70, 96)),
        }
        aug_func = transforms.Lambda(
            lambda image: data_augment(image, self.aug_config)
        )
        tensor_and_normalize = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std),
            ]
        )

        self.train_transform = transforms.Compose(
            [
                aug_func,
                AlignedCrop(self.cropSize, train=True),
                transforms.RandomHorizontalFlip(),
                tensor_and_normalize,
            ]
        )
        self.val_transform = transforms.Compose(
            [
                AlignedCrop(self.cropSize, train=False),
                tensor_and_normalize,
            ]
        )
        self.test_transform = transforms.Compose(
            [
                AlignedCrop(self.cropSize, train=False),
                tensor_and_normalize,
            ]
        )

    def forward(self, x, return_feat: bool = False):
        feat = self.artifact_encoder(x)
        out = self.fc(feat)
        if return_feat:
            return feat, out
        return out
