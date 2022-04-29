import torch
import torch.nn as nn
import torch.nn.init as init
from typing import Any

class Fire(nn.Module):

    def __init__(
        self,
        inplanes: int,
        squeeze_planes: int,
        expand1x1_planes: int,
        expand3x3_planes: int
    ) -> None:
        super(Fire, self).__init__() #the whole fire module does not change w and h
        self.inplanes = inplanes
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1) #reduce number of channels
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes,
                                   kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes,
                                   kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(expand1x1_planes+expand3x3_planes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.squeeze_activation(self.squeeze(x))
        x=torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))
        ], 1)
        return self.bn1(x)

class SqueezeNet(nn.Module):

    def __init__(
        self,
        input_channels: int,
        num_classes: int
    ) -> None:
        super(SqueezeNet, self).__init__()
        self.num_classes = num_classes
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3,padding='same'), #padding added
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(64, 16, 64, 64),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(128, 32, 128, 128),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(256, 64, 256, 256),
            Fire(512, 80, 512, 512),
            Fire(1024, 96, 1024, 1024),
        )

        # Final convolution is initialized differently from the rest
        final_conv = nn.Conv2d(2048, self.num_classes, kernel_size=1)

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            final_conv,
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m is final_conv:
                    init.normal_(m.weight, mean=0.0, std=0.01)
                else:
                    init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
    @staticmethod
    def get_classifiers():
        return ['my-squeezenet']
    @classmethod
    def build_classifier(cls, arch: str, num_classes: int, input_channels: int):
        cls_instance=cls(num_classes=num_classes,input_channels=input_channels)
        return cls_instance

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return torch.flatten(x, 1)
