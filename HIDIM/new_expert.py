import torch
import torch.nn as nn

import torch.nn.functional as F


class LWSClassifier(nn.Module):
    def __init__(self, in_features, out_features, dropout_prob=0.2):
        super(LWSClassifier, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.scale = nn.Parameter(torch.ones(1, out_features))
    def forward(self, x):
        logits = self.fc(x)
        w = self.fc.weight.data
        scaled_logits = logits * self.scale
        return scaled_logits



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
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, in_channels=1024)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifiers = LWSClassifier(512 * block.expansion, num_classes, dropout_prob=0.2)

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

    def forward(self, x, model):
        x = model(x, False, True)
        out4 = self.layer4(x)
        exp_outs = F.avg_pool2d(out4, out4.size()[3]).view(out4.size(0), -1)
        out_z = self.classifiers(exp_outs)
        return out_z



def expert_resnet50(num_classes=1000, num_exps=None, use_norm=False):
    """Constructs a ResNet-50 model.
    Args:
        num_classes (int): Number of output classes (default: 1000)
    """
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes, num_experts=num_exps, use_norm=use_norm)