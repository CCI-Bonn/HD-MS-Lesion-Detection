"""                                                                                                            
:AUTHOR: Hagen Meredig                                                                                         
:ORGANIZATION: Department of Neuroradiology, Heidelberg Univeristy Hospital                                    
:CONTACT: Hagen.Meredig@med.uni-heidelberg.de                                                                  
:SINCE: August 18 2021                                                                            
""" 
import torch
import torch.nn as nn
import torch.nn.functional as F
from .resnet import ResNet
from .resnet import ResNet25D
from .resnet import BasicBlock
from .resnet import Bottleneck
from .resnet3d import ResNet3D
from .resnet3d import BasicBlock3D
from .resnet3d import Bottleneck3D
from .autoencoders import ConvAutoencoder2D
from .mil_models import DeepMILAttention2d
from .mil_models import DeepMILAttention3d, DeepAttention3d, DeepAttentionBN3d
from .mil_models import DeepMILResnet2d
from .mil_models import AttentionOnly2d
from .custom_resnet import CustomResnet2d
from .custom_resnet import CustomResnet3d
from torchvision.models.utils import load_state_dict_from_url


model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-b121ed2d.pth",
    "resnext50_32x4d": "https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth",
    "resnext101_32x8d": "https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth",
    "wide_resnet50_2": "https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth",
    "wide_resnet101_2": "https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth",
}


class ResNet18(ResNet):
    def __init__(
        self,
        num_classes=4,
        num_channels=3,
        pretrained: bool = False,
        progress: bool = True,
    ):
        if pretrained:
            super(ResNet18, self).__init__(
                BasicBlock, [2, 2, 2, 2], num_classes=1000, num_channels=3
            )
        else:
            super(ResNet18, self).__init__(
                BasicBlock,
                [2, 2, 2, 2],
                num_classes=num_classes,
                num_channels=num_channels,
            )
        if pretrained:
            state_dict = load_state_dict_from_url(
                model_urls["resnet18"], progress=progress
            )
            self.load_state_dict(state_dict)
            num_ftrs = self.fc.in_features
            self.fc = nn.Linear(num_ftrs, num_classes)

class ResNet18_3D(ResNet3D):
    def __init__(self, num_classes=4, num_channels=3):
        super(ResNet18_3D, self).__init__(
            BasicBlock3D, [2, 2, 2, 2], [64, 128, 256, 512], num_classes=num_classes, num_channels=num_channels
        )            

class ResNet34(ResNet):
    def __init__(self, num_classes=4, num_channels=3):
        super(ResNet34, self).__init__(
            BasicBlock, [3, 4, 6, 3], num_classes=num_classes, num_channels=num_channels
        )


class ResNet34_3D(ResNet3D):
    def __init__(self, num_classes=4, num_channels=3):
        super(ResNet34_3D, self).__init__(
            BasicBlock3D, [3, 4, 6, 3], num_classes=num_classes, num_channels=num_channels
        )        

class ResNet50(ResNet):
    def __init__(self, num_classes=4, num_channels=3):
        super(ResNet50, self).__init__(
            Bottleneck, [3, 4, 6, 3], num_classes=num_classes, num_channels=num_channels
        )


class ResNet101(ResNet):
    def __init__(self, num_classes=4, num_channels=3):
        super(ResNet101, self).__init__(
            Bottleneck,
            [3, 4, 23, 3],
            num_classes=num_classes,
            num_channels=num_channels,
        )


class ResNet152(ResNet):
    def __init__(self, num_classes=4, num_channels=3):
        super(ResNet152, self).__init__(
            Bottleneck,
            [3, 4, 36, 3],
            num_classes=num_classes,
            num_channels=num_channels,
        )


class WideResNet50(ResNet):
    def __init__(self, num_classes=4, num_channels=3):
        super(WideResNet50, self).__init__(
            Bottleneck,
            [3, 4, 6, 3],
            num_classes=num_classes,
            num_channels=num_channels,
            width_per_group=64 * 2,
        )


class WideResNet101(ResNet):
    def __init__(self, num_classes=4, num_channels=3):
        super(WideResNet101, self).__init__(
            Bottleneck,
            [3, 4, 23, 3],
            num_classes=num_classes,
            num_channels=num_channels,
            width_per_group=64 * 2,
        )


class ResNet18_25D(ResNet25D):
    def __init__(self, num_classes=4, num_channels=3):
        super(ResNet18_25D, self).__init__(
            BasicBlock, [2, 2, 2, 2], num_classes=num_classes, num_channels=num_channels
        )
