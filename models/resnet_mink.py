# Please cite "4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural
# Networks", CVPR'19 (https://arxiv.org/abs/1904.08755) if you use any part
# of the code.
import torch.nn as nn

import MinkowskiEngine as ME
from MinkowskiEngine.modules.resnet_block import BasicBlock, Bottleneck


class ResNetBase(nn.Module):
    BLOCK = None
    LAYERS = ()
    INIT_DIM = 64
    PLANES = (64, 128, 256, 512)

    def __init__(self, in_channels, out_channels, D=3):
        nn.Module.__init__(self)
        self.D = D
        assert self.BLOCK is not None

        self.network_initialization(in_channels, out_channels, D)
        self.weight_initialization()

    def network_initialization(self, in_channels, out_channels, D):

        self.inplanes = self.INIT_DIM
        self.conv1 = ME.MinkowskiConvolution(
            in_channels, self.inplanes, kernel_size=5, stride=2, dimension=D)

        self.bn1 = ME.MinkowskiBatchNorm(self.inplanes)
        self.relu = ME.MinkowskiReLU(inplace=True)

        self.pool = ME.MinkowskiAvgPooling(kernel_size=2, stride=2, dimension=D)

        self.layer1 = self._make_layer(
            self.BLOCK, self.PLANES[0], self.LAYERS[0], stride=2)
        self.layer2 = self._make_layer(
            self.BLOCK, self.PLANES[1], self.LAYERS[1], stride=2)
        self.layer3 = self._make_layer(
            self.BLOCK, self.PLANES[2], self.LAYERS[2], stride=2)
        self.layer4 = self._make_layer(
            self.BLOCK, self.PLANES[3], self.LAYERS[3], stride=2)

        self.conv5 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=3, stride=3, dimension=D)
        self.bn5 = ME.MinkowskiBatchNorm(self.inplanes)

        self.glob_avg = ME.MinkowskiGlobalMaxPooling(dimension=D)

        self.final = ME.MinkowskiLinear(self.inplanes, out_channels, bias=True)

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, ME.MinkowskiConvolution):
                ME.utils.kaiming_normal_(m.kernel, mode='fan_out', nonlinearity='relu')

            if isinstance(m, ME.MinkowskiBatchNorm):
                nn.init.constant_(m.bn.weight, 1)
                nn.init.constant_(m.bn.bias, 0)

    def _make_layer(self,
                    block,
                    planes,
                    blocks,
                    stride=1,
                    dilation=1,
                    bn_momentum=0.1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                ME.MinkowskiConvolution(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    dimension=self.D),
                ME.MinkowskiBatchNorm(planes * block.expansion))
        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride=stride,
                dilation=dilation,
                downsample=downsample,
                dimension=self.D))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    stride=1,
                    dilation=dilation,
                    dimension=self.D))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)

        x = self.glob_avg(x)
        return self.final(x)


class ResNet14(ResNetBase):
    BLOCK = BasicBlock
    LAYERS = (1, 1, 1, 1)


class ResNet18(ResNetBase):
    BLOCK = BasicBlock
    LAYERS = (2, 2, 2, 2)


class ResNet34(ResNetBase):
    BLOCK = BasicBlock
    LAYERS = (3, 4, 6, 3)


class ResNet50(ResNetBase):
    BLOCK = Bottleneck
    LAYERS = (3, 4, 6, 3)


class ResNet101(ResNetBase):
    BLOCK = Bottleneck
    LAYERS = (3, 4, 23, 3)
