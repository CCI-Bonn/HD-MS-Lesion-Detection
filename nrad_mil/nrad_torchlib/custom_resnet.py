"""                                                                                                            
:AUTHOR: Hagen Meredig                                                                                         
:ORGANIZATION: Department of Neuroradiology, Heidelberg Univeristy Hospital                                    
:CONTACT: Hagen.Meredig@med.uni-heidelberg.de                                                                  
:SINCE: August 18, 2021                                                                            
"""

import torch
from torch import nn
# preactivation resnet with bottleneck blocks as in https://arxiv.org/pdf/1603.05027.pdf


class CustomResnet2d(nn.Module):
    def __init__(
        self,
        blocks=[2, 2, 2, 2],
        input_size=(224, 224),
        n_classes=2,
        n_channels=1,
        init_nb_filters=64,
        init_kernel_size=7,
        init_padding=3,
        init_stride=2,
        growth_factor=2,
        batchnorm_eps=1e-5,
        batchnorm_mom=0.1,
        batchnorm_affine=True,
        batchnorm_track_stats=True,
        dilation=1,
    ):

        super(CustomResnet2d, self).__init__()

        self.blocks = blocks
        self.input_shape = input_size
        self.n_channels = n_channels
        self.init_nb_filters = init_nb_filters
        self.growth_factor = growth_factor

        self.initial_conv = nn.Sequential(
            nn.BatchNorm2d(
                self.n_channels,
                eps=batchnorm_eps,
                momentum=batchnorm_mom,
                affine=batchnorm_affine,
                track_running_stats=batchnorm_track_stats,
            ),
            nn.ReLU(),
            nn.Conv2d(
                self.n_channels,
                init_nb_filters,
                kernel_size=init_kernel_size,
                padding=init_padding,
                stride=init_stride,
                bias=False,
            ),
            nn.MaxPool2d(3, stride=2, padding=1),
        )

        n_filters = self.init_nb_filters

        for i in range(len(blocks)):
            for j in range(blocks[i]):
                if i == 0 and j == 0:
                    self.add_module(
                        f"block_{i}_layer{j}",
                        nn.Sequential(
                            nn.BatchNorm2d(
                                n_filters,
                                eps=batchnorm_eps,
                                momentum=batchnorm_mom,
                                affine=batchnorm_affine,
                                track_running_stats=batchnorm_track_stats,
                            ),
                            nn.ReLU(),
                            nn.Conv2d(n_filters, n_filters, kernel_size=1, bias=False),
                            nn.BatchNorm2d(
                                n_filters,
                                eps=batchnorm_eps,
                                momentum=batchnorm_mom,
                                affine=batchnorm_affine,
                                track_running_stats=batchnorm_track_stats,
                            ),
                            nn.ReLU(),
                            nn.Conv2d(
                                n_filters,
                                n_filters,
                                kernel_size=3,
                                padding=1,
                                bias=False,
                            ),
                            nn.BatchNorm2d(
                                n_filters,
                                eps=batchnorm_eps,
                                momentum=batchnorm_mom,
                                affine=batchnorm_affine,
                                track_running_stats=batchnorm_track_stats,
                            ),
                            nn.ReLU(),
                            nn.Conv2d(
                                n_filters,
                                int(n_filters * growth_factor * 2),
                                kernel_size=1,
                                bias=False,
                            ),
                        ),
                    )
                    n_filters = int(n_filters * growth_factor * 2)
                elif j == 0:
                    self.add_module(
                        f"block_{i}_layer{j}",
                        nn.Sequential(
                            nn.BatchNorm2d(
                                n_filters,
                                eps=batchnorm_eps,
                                momentum=batchnorm_mom,
                                affine=batchnorm_affine,
                                track_running_stats=batchnorm_track_stats,
                            ),
                            nn.ReLU(),
                            nn.Conv2d(
                                n_filters,
                                int(n_filters * growth_factor // 4),
                                kernel_size=1,
                                bias=False,
                            ),
                            nn.BatchNorm2d(
                                int(n_filters * growth_factor // 4),
                                eps=batchnorm_eps,
                                momentum=batchnorm_mom,
                                affine=batchnorm_affine,
                                track_running_stats=batchnorm_track_stats,
                            ),
                            nn.ReLU(),
                            nn.Conv2d(
                                int(n_filters * growth_factor // 4),
                                int(n_filters * growth_factor // 4),
                                kernel_size=3,
                                padding=1,
                                stride=2,
                                bias=False,
                            ),
                            nn.BatchNorm2d(
                                int(n_filters * growth_factor // 4),
                                eps=batchnorm_eps,
                                momentum=batchnorm_mom,
                                affine=batchnorm_affine,
                                track_running_stats=batchnorm_track_stats,
                            ),
                            nn.ReLU(),
                            nn.Conv2d(
                                int(n_filters * growth_factor // 4),
                                int(n_filters * growth_factor),
                                kernel_size=1,
                                bias=False,
                            ),
                        ),
                    )
                    n_filters = int(n_filters * growth_factor)
                else:
                    self.add_module(
                        f"block_{i}_layer{j}",
                        nn.Sequential(
                            nn.BatchNorm2d(
                                n_filters,
                                eps=batchnorm_eps,
                                momentum=batchnorm_mom,
                                affine=batchnorm_affine,
                                track_running_stats=batchnorm_track_stats,
                            ),
                            nn.ReLU(),
                            nn.Conv2d(
                                n_filters, n_filters // 4, kernel_size=1, bias=False
                            ),
                            nn.BatchNorm2d(
                                n_filters // 4,
                                eps=batchnorm_eps,
                                momentum=batchnorm_mom,
                                affine=batchnorm_affine,
                                track_running_stats=batchnorm_track_stats,
                            ),
                            nn.ReLU(),
                            nn.Conv2d(
                                n_filters // 4,
                                n_filters // 4,
                                kernel_size=3,
                                padding=1,
                                bias=False,
                            ),
                            nn.BatchNorm2d(
                                n_filters // 4,
                                eps=batchnorm_eps,
                                momentum=batchnorm_mom,
                                affine=batchnorm_affine,
                                track_running_stats=batchnorm_track_stats,
                            ),
                            nn.ReLU(),
                            nn.Conv2d(
                                n_filters // 4, n_filters, kernel_size=1, bias=False
                            ),
                        ),
                    )

        self.final_n_filters = int(n_filters)

        n_filters = self.init_nb_filters
        for i in range(len(blocks)):
            for j in range(blocks[i]):
                if i == 0 and j == 0:
                    self.add_module(
                        f"shortcut_block_{i}_layer{j}",
                        nn.Conv2d(
                            n_filters,
                            int(n_filters * growth_factor * 2),
                            kernel_size=1,
                            bias=False,
                        ),
                    )
                    n_filters = int(n_filters * growth_factor * 2)
                elif j == 0:
                    self.add_module(
                        f"shortcut_block_{i}_layer{j}",
                        nn.Conv2d(
                            n_filters,
                            int(n_filters * growth_factor),
                            kernel_size=1,
                            stride=2,
                            dilation=dilation,
                            bias=False,
                        ),
                    )
                    n_filters = int(n_filters * growth_factor)

        self.gap = nn.Sequential(
            nn.BatchNorm2d(
                n_filters,
                eps=batchnorm_eps,
                momentum=batchnorm_mom,
                affine=batchnorm_affine,
                track_running_stats=batchnorm_track_stats,
            ),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.fc = nn.Linear(n_filters, 2)

        try:
            self._forward_test(torch.normal(1, 1, size=(1, n_channels, *input_size)))
        except Exception as e:
            self.min_featuremap_dim = None
            print("Warning: Input shape may be ill defined (likely too small)", e)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)) and batchnorm_affine:
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _forward_test(self, x):

        x = self.initial_conv(x)

        for i in range(len(self.blocks)):
            for j in range(self.blocks[i]):
                if j == 0:
                    x_ = self._modules[f"block_{i}_layer{j}"](x)
                    x = self._modules[f"shortcut_block_{i}_layer{j}"](x)
                    x = torch.add(x, x_, alpha=1)
                else:
                    x_ = self._modules[f"block_{i}_layer{j}"](x)
                    x = torch.add(x, x_, alpha=1)
        self.min_featuremap_dim = x.shape[-2:]
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):

        x = self.initial_conv(x)

        for i in range(len(self.blocks)):
            for j in range(self.blocks[i]):
                if j == 0:
                    x_ = self._modules[f"block_{i}_layer{j}"](x)
                    x = self._modules[f"shortcut_block_{i}_layer{j}"](x)
                    x = torch.add(x, x_, alpha=1)
                else:
                    x_ = self._modules[f"block_{i}_layer{j}"](x)
                    x = torch.add(x, x_, alpha=1)

        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


class CustomResnet2d_FE(nn.Module):
    def __init__(
        self,
        blocks=[2, 2, 2, 2],
        input_size=(224, 224),
        n_classes=2,
        n_channels=1,
        init_nb_filters=64,
        init_kernel_size=7,
        init_padding=3,
        init_stride=2,
        growth_factor=2,
        batchnorm_eps=1e-5,
        batchnorm_mom=0.1,
        batchnorm_affine=True,
        batchnorm_track_stats=True,
        dilation=1,
    ):

        super(CustomResnet2d_FE, self).__init__()

        self.blocks = blocks
        self.input_shape = input_size
        self.n_channels = n_channels
        self.init_nb_filters = init_nb_filters
        self.growth_factor = growth_factor

        self.initial_conv = nn.Sequential(
            nn.BatchNorm2d(
                self.n_channels,
                eps=batchnorm_eps,
                momentum=batchnorm_mom,
                affine=batchnorm_affine,
                track_running_stats=batchnorm_track_stats,
            ),
            nn.ReLU(),
            nn.Conv2d(
                self.n_channels,
                init_nb_filters,
                kernel_size=init_kernel_size,
                padding=init_padding,
                stride=init_stride,
                bias=False,
            ),
            nn.MaxPool2d(3, stride=2, padding=1),
        )

        n_filters = self.init_nb_filters

        for i in range(len(blocks)):
            for j in range(blocks[i]):
                if i == 0 and j == 0:
                    self.add_module(
                        f"block_{i}_layer{j}",
                        nn.Sequential(
                            nn.BatchNorm2d(
                                n_filters,
                                eps=batchnorm_eps,
                                momentum=batchnorm_mom,
                                affine=batchnorm_affine,
                                track_running_stats=batchnorm_track_stats,
                            ),
                            nn.ReLU(),
                            nn.Conv2d(n_filters, n_filters, kernel_size=1, bias=False),
                            nn.BatchNorm2d(
                                n_filters,
                                eps=batchnorm_eps,
                                momentum=batchnorm_mom,
                                affine=batchnorm_affine,
                                track_running_stats=batchnorm_track_stats,
                            ),
                            nn.ReLU(),
                            nn.Conv2d(
                                n_filters,
                                n_filters,
                                kernel_size=3,
                                padding=1,
                                bias=False,
                            ),
                            nn.BatchNorm2d(
                                n_filters,
                                eps=batchnorm_eps,
                                momentum=batchnorm_mom,
                                affine=batchnorm_affine,
                                track_running_stats=batchnorm_track_stats,
                            ),
                            nn.ReLU(),
                            nn.Conv2d(
                                n_filters,
                                int(n_filters * growth_factor * 2),
                                kernel_size=1,
                                bias=False,
                            ),
                        ),
                    )
                    n_filters = int(n_filters * growth_factor * 2)
                elif j == 0:
                    self.add_module(
                        f"block_{i}_layer{j}",
                        nn.Sequential(
                            nn.BatchNorm2d(
                                n_filters,
                                eps=batchnorm_eps,
                                momentum=batchnorm_mom,
                                affine=batchnorm_affine,
                                track_running_stats=batchnorm_track_stats,
                            ),
                            nn.ReLU(),
                            nn.Conv2d(
                                n_filters,
                                int(n_filters * growth_factor // 4),
                                kernel_size=1,
                                bias=False,
                            ),
                            nn.BatchNorm2d(
                                int(n_filters * growth_factor // 4),
                                eps=batchnorm_eps,
                                momentum=batchnorm_mom,
                                affine=batchnorm_affine,
                                track_running_stats=batchnorm_track_stats,
                            ),
                            nn.ReLU(),
                            nn.Conv2d(
                                int(n_filters * growth_factor // 4),
                                int(n_filters * growth_factor // 4),
                                kernel_size=3,
                                padding=1,
                                stride=2,
                                bias=False,
                            ),
                            nn.BatchNorm2d(
                                int(n_filters * growth_factor // 4),
                                eps=batchnorm_eps,
                                momentum=batchnorm_mom,
                                affine=batchnorm_affine,
                                track_running_stats=batchnorm_track_stats,
                            ),
                            nn.ReLU(),
                            nn.Conv2d(
                                int(n_filters * growth_factor // 4),
                                int(n_filters * growth_factor),
                                kernel_size=1,
                                bias=False,
                            ),
                        ),
                    )
                    n_filters = int(n_filters * growth_factor)
                else:
                    self.add_module(
                        f"block_{i}_layer{j}",
                        nn.Sequential(
                            nn.BatchNorm2d(
                                n_filters,
                                eps=batchnorm_eps,
                                momentum=batchnorm_mom,
                                affine=batchnorm_affine,
                                track_running_stats=batchnorm_track_stats,
                            ),
                            nn.ReLU(),
                            nn.Conv2d(
                                n_filters, n_filters // 4, kernel_size=1, bias=False
                            ),
                            nn.BatchNorm2d(
                                n_filters // 4,
                                eps=batchnorm_eps,
                                momentum=batchnorm_mom,
                                affine=batchnorm_affine,
                                track_running_stats=batchnorm_track_stats,
                            ),
                            nn.ReLU(),
                            nn.Conv2d(
                                n_filters // 4,
                                n_filters // 4,
                                kernel_size=3,
                                padding=1,
                                bias=False,
                            ),
                            nn.BatchNorm2d(
                                n_filters // 4,
                                eps=batchnorm_eps,
                                momentum=batchnorm_mom,
                                affine=batchnorm_affine,
                                track_running_stats=batchnorm_track_stats,
                            ),
                            nn.ReLU(),
                            nn.Conv2d(
                                n_filters // 4, n_filters, kernel_size=1, bias=False
                            ),
                        ),
                    )

        self.final_n_filters = int(n_filters)

        n_filters = self.init_nb_filters
        for i in range(len(blocks)):
            for j in range(blocks[i]):
                if i == 0 and j == 0:
                    self.add_module(
                        f"shortcut_block_{i}_layer{j}",
                        nn.Conv2d(
                            n_filters,
                            int(n_filters * growth_factor * 2),
                            kernel_size=1,
                            bias=False,
                        ),
                    )
                    n_filters = int(n_filters * growth_factor * 2)
                elif j == 0:
                    self.add_module(
                        f"shortcut_block_{i}_layer{j}",
                        nn.Conv2d(
                            n_filters,
                            int(n_filters * growth_factor),
                            kernel_size=1,
                            stride=2,
                            dilation=dilation,
                            bias=False,
                        ),
                    )
                    n_filters = int(n_filters * growth_factor)

        try:
            self._forward_test(torch.normal(1, 1, size=(1, n_channels, *input_size)))
        except Exception as e:
            self.min_featuremap_dim = None
            print("Warning: Input shape may be ill defined (likely too small)", e)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)) and batchnorm_affine:
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _forward_test(self, x):

        x = self.initial_conv(x)

        for i in range(len(self.blocks)):
            for j in range(self.blocks[i]):
                if j == 0:
                    x_ = self._modules[f"block_{i}_layer{j}"](x)
                    x = self._modules[f"shortcut_block_{i}_layer{j}"](x)
                    x = torch.add(x, x_, alpha=1)
                else:
                    x_ = self._modules[f"block_{i}_layer{j}"](x)
                    x = torch.add(x, x_, alpha=1)
        self.min_featuremap_dim = x.shape[-2:]

        return x

    def forward(self, x):

        x = self.initial_conv(x)

        for i in range(len(self.blocks)):
            for j in range(self.blocks[i]):
                if j == 0:
                    x_ = self._modules[f"block_{i}_layer{j}"](x)
                    x = self._modules[f"shortcut_block_{i}_layer{j}"](x)
                    x = torch.add(x, x_, alpha=1)
                else:
                    x_ = self._modules[f"block_{i}_layer{j}"](x)
                    x = torch.add(x, x_, alpha=1)

        return x


class CustomResnet3d(nn.Module):
    def __init__(
        self,
        blocks=[2, 2, 2, 2],
        input_size=(224, 224, 224),
        n_classes=2,
        n_channels=1,
        init_nb_filters=64,
        init_kernel_size=7,
        bottom_nb_filters=256,
        init_padding=3,
        init_stride=2,
        growth_factor=2,
        batchnorm_eps=1e-5,
        batchnorm_mom=0.1,
        batchnorm_affine=True,
        batchnorm_track_stats=True,
        dilation=1,
    ):

        super(CustomResnet3d, self).__init__()

        self.blocks = blocks
        self.input_shape = input_size
        self.n_channels = n_channels
        self.init_nb_filters = init_nb_filters
        self.bottom_nb_filters = bottom_nb_filters
        self.growth_factor = growth_factor

        self.initial_conv = nn.Sequential(
            nn.BatchNorm3d(
                self.n_channels,
                eps=batchnorm_eps,
                momentum=batchnorm_mom,
                affine=batchnorm_affine,
                track_running_stats=batchnorm_track_stats,
            ),
            nn.ReLU(),
            nn.Conv3d(
                self.n_channels,
                init_nb_filters,
                kernel_size=init_kernel_size,
                padding=init_padding,
                stride=init_stride,
                bias=False,
            ),
            nn.MaxPool3d(3, stride=2, padding=1),
        )

        n_filters = self.init_nb_filters

        for i in range(len(blocks)):
            for j in range(blocks[i]):
                if i == 0 and j == 0:
                    self.add_module(
                        f"block_{i}_layer{j}",
                        nn.Sequential(
                            nn.BatchNorm3d(
                                n_filters,
                                eps=batchnorm_eps,
                                momentum=batchnorm_mom,
                                affine=batchnorm_affine,
                                track_running_stats=batchnorm_track_stats,
                            ),
                            nn.ReLU(),
                            nn.Conv3d(n_filters, n_filters, kernel_size=1, bias=False),
                            nn.BatchNorm3d(
                                n_filters,
                                eps=batchnorm_eps,
                                momentum=batchnorm_mom,
                                affine=batchnorm_affine,
                                track_running_stats=batchnorm_track_stats,
                            ),
                            nn.ReLU(),
                            nn.Conv3d(
                                n_filters,
                                n_filters,
                                kernel_size=3,
                                padding=1,
                                bias=False,
                            ),
                            nn.BatchNorm3d(
                                n_filters,
                                eps=batchnorm_eps,
                                momentum=batchnorm_mom,
                                affine=batchnorm_affine,
                                track_running_stats=batchnorm_track_stats,
                            ),
                            nn.ReLU(),
                            nn.Conv3d(
                                n_filters,
                                n_filters * growth_factor * 2,
                                kernel_size=1,
                                bias=False,
                            ),
                        ),
                    )
                    n_filters *= growth_factor * 2
                elif j == 0:
                    self.add_module(
                        f"block_{i}_layer{j}",
                        nn.Sequential(
                            nn.BatchNorm3d(
                                n_filters,
                                eps=batchnorm_eps,
                                momentum=batchnorm_mom,
                                affine=batchnorm_affine,
                                track_running_stats=batchnorm_track_stats,
                            ),
                            nn.ReLU(),
                            nn.Conv3d(
                                n_filters,
                                n_filters * growth_factor // 4,
                                kernel_size=1,
                                bias=False,
                            ),
                            nn.BatchNorm3d(
                                n_filters * growth_factor // 4,
                                eps=batchnorm_eps,
                                momentum=batchnorm_mom,
                                affine=batchnorm_affine,
                                track_running_stats=batchnorm_track_stats,
                            ),
                            nn.ReLU(),
                            nn.Conv3d(
                                n_filters * growth_factor // 4,
                                n_filters * growth_factor // 4,
                                kernel_size=3,
                                padding=1,
                                stride=2,
                                bias=False,
                            ),
                            nn.BatchNorm3d(
                                n_filters * growth_factor // 4,
                                eps=batchnorm_eps,
                                momentum=batchnorm_mom,
                                affine=batchnorm_affine,
                                track_running_stats=batchnorm_track_stats,
                            ),
                            nn.ReLU(),
                            nn.Conv3d(
                                n_filters * growth_factor // 4,
                                n_filters * growth_factor,
                                kernel_size=1,
                                bias=False,
                            ),
                        ),
                    )
                    n_filters *= growth_factor
                else:
                    self.add_module(
                        f"block_{i}_layer{j}",
                        nn.Sequential(
                            nn.BatchNorm3d(
                                n_filters,
                                eps=batchnorm_eps,
                                momentum=batchnorm_mom,
                                affine=batchnorm_affine,
                                track_running_stats=batchnorm_track_stats,
                            ),
                            nn.ReLU(),
                            nn.Conv3d(
                                n_filters, n_filters // 4, kernel_size=1, bias=False
                            ),
                            nn.BatchNorm3d(
                                n_filters // 4,
                                eps=batchnorm_eps,
                                momentum=batchnorm_mom,
                                affine=batchnorm_affine,
                                track_running_stats=batchnorm_track_stats,
                            ),
                            nn.ReLU(),
                            nn.Conv3d(
                                n_filters // 4,
                                n_filters // 4,
                                kernel_size=3,
                                padding=1,
                                bias=False,
                            ),
                            nn.BatchNorm3d(
                                n_filters // 4,
                                eps=batchnorm_eps,
                                momentum=batchnorm_mom,
                                affine=batchnorm_affine,
                                track_running_stats=batchnorm_track_stats,
                            ),
                            nn.ReLU(),
                            nn.Conv3d(
                                n_filters // 4, n_filters, kernel_size=1, bias=False
                            ),
                        ),
                    )

        self.final_n_filters = int(n_filters)

        n_filters = self.init_nb_filters
        for i in range(len(blocks)):
            for j in range(blocks[i]):
                if i == 0 and j == 0:
                    self.add_module(
                        f"shortcut_block_{i}_layer{j}",
                        nn.Conv3d(
                            n_filters,
                            n_filters * growth_factor * 2,
                            kernel_size=1,
                            bias=False,
                        ),
                    )
                    n_filters *= growth_factor * 2
                elif j == 0:
                    self.add_module(
                        f"shortcut_block_{i}_layer{j}",
                        nn.Conv3d(
                            n_filters,
                            n_filters * growth_factor,
                            kernel_size=1,
                            stride=2,
                            dilation=dilation,
                            bias=False,
                        ),
                    )
                    n_filters *= growth_factor

        self.gap = nn.Sequential(
            nn.BatchNorm3d(
                n_filters,
                eps=batchnorm_eps,
                momentum=batchnorm_mom,
                affine=batchnorm_affine,
                track_running_stats=batchnorm_track_stats,
            ),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
        )

        self.fc = nn.Linear(n_filters, 2)

        try:
            self._forward_test(torch.normal(1, 1, size=(1, n_channels, *input_size)))
        except Exception as e:
            self.min_featuremap_dim = None
            print("Warning: Input shape may be ill defined (likely too small)", e)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)) and batchnorm_affine:
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _forward_test(self, x):

        x = self.initial_conv(x)

        for i in range(len(self.blocks)):
            for j in range(self.blocks[i]):
                if j == 0:
                    x_ = self._modules[f"block_{i}_layer{j}"](x)
                    x = self._modules[f"shortcut_block_{i}_layer{j}"](x)
                    x = torch.add(x, x_, alpha=1)
                else:
                    x_ = self._modules[f"block_{i}_layer{j}"](x)
                    x = torch.add(x, x_, alpha=1)
        self.min_featuremap_dim = x.shape[-3:]
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):

        x = self.initial_conv(x)

        for i in range(len(self.blocks)):
            for j in range(self.blocks[i]):
                if j == 0:
                    x_ = self._modules[f"block_{i}_layer{j}"](x)
                    x = self._modules[f"shortcut_block_{i}_layer{j}"](x)
                    x = torch.add(x, x_, alpha=1)
                else:
                    x_ = self._modules[f"block_{i}_layer{j}"](x)
                    x = torch.add(x, x_, alpha=1)

        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
