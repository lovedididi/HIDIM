
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from collections import OrderedDict


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(in_channels // reduction_ratio, in_channels)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x).view(x.size(0), -1))
        max_out = self.fc(self.max_pool(x).view(x.size(0), -1))
        out = avg_out + max_out
        return self.sigmoid(out).unsqueeze(2).unsqueeze(3) * x


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return self.sigmoid(out) * x


class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super().__init__()
        self.channel_att = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_att = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x



class StridedConv(nn.Module):
    """
    downsampling conv layer
    """
    def __init__(self, in_planes, planes, use_relu=False) -> None:
        super(StridedConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_planes, out_channels=planes,
                      kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(planes)
        )
        self.use_relu = use_relu
        if use_relu:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        if self.use_relu:
            out = self.relu(out)
        return out


class ShallowExpert(nn.Module):
    """
    shallow features alignment wrt. depth
    """
    def __init__(self, input_dim=None, depth=None) -> None:
        super(ShallowExpert, self).__init__()
        layers = OrderedDict()
        for k in range(depth):
            layers[f'StridedConv{k}'] = StridedConv(
                in_planes=input_dim * (2 ** k),
                planes=input_dim * (2 ** (k + 1)),
                use_relu = (k != 1)
            )
            layers[f'CBAM{k}'] = CBAM(in_channels=input_dim * (2 ** (k + 1)))
        self.convs = nn.Sequential(layers)
    def forward(self, x):
        out = self.convs(x)
        return out


class NormedLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(NormedLinear, self).__init__()
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):
        out = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        return out


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, num_experts=10, use_norm=False):
        super(ResNet, self).__init__()
        self.s = 1
        self.in_channels = 64
        self.num_experts = num_experts
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)

        dropout_rate = 0.2
        if num_experts:
            self.layer4s = nn.ModuleList([self._make_layer(
                block, 512, layers[3], stride=2, in_channels=1024) for _ in range(self.num_experts)])
            if use_norm:
                self.s = 30
                self.classifiers = nn.ModuleList([
                    nn.Sequential(
                        nn.Dropout(dropout_rate),
                        NormedLinear(512 * block.expansion, num_classes)
                    ) for _ in range(self.num_experts)
                ])
                self.rt_classifiers = nn.ModuleList([
                    nn.Sequential(
                        nn.Dropout(dropout_rate),
                        NormedLinear(512 * block.expansion, num_classes)
                    ) for _ in range(self.num_experts)
                ])
            else:
                self.classifiers = nn.ModuleList([
                    nn.Sequential(
                        nn.Dropout(dropout_rate),
                        NormedLinear(512 * block.expansion, num_classes)
                    ) for _ in range(self.num_experts)
                ])
        else:
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
            self.linear = NormedLinear(512 * block.expansion, num_classes) if use_norm else nn.Linear(
                512 * block.expansion, num_classes, bias=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # [3,2,1]
        self.depth = list(reversed([i + 1 for i in range(len(layers) - 1)]))

        feat_dim = 256
        self.shallow_exps = nn.ModuleList([ShallowExpert(
            input_dim=feat_dim * (2 ** (-1 * (d - 3))), depth=d) for d in self.depth])


    def _make_layer(self, block, out_channels, blocks, stride=1, in_channels=None):
        if in_channels is None:
            in_channels = self.in_channels

        downsample = None
        if stride != 1 or in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x, crt=False, isExpert4=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        shallow_outs = [out1, out2, out3]
        if isExpert4:
            outs = out3
        else:
            if self.num_experts:
                out4s = [self.layer4s[_](out3) for _ in range(self.num_experts)]
                shallow_expe_outs = [self.shallow_exps[i](shallow_outs[i]) for i in range(self.num_experts)]
                exp_outs = [out4s[i] * shallow_expe_outs[i] for i in range(self.num_experts)]
                exp_outs = [F.avg_pool2d(output, output.size()[3]).view(
                    output.size(0), -1) for output in exp_outs]
                if crt == True:
                    outs = [self.s * self.rt_classifiers[i](exp_outs[i]) for i in range(self.num_experts)]
                else:
                    outs = [self.s * self.classifiers[i](exp_outs[i]) for i in range(self.num_experts)]
            else:
                out = self.layer4(out3)
                out = self.avgpool(out)
                out = torch.flatten(out, 1)
                outs = self.linear(out)
        return outs






def resnet50(num_classes=1000, num_exps=None, use_norm=False):
    """Constructs a ResNet-50 model.

    Args:
        num_classes (int): Number of output classes (default: 1000)
    """
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes, num_experts=num_exps, use_norm=use_norm)