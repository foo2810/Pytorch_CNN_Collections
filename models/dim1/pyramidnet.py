# Based on torchvision.model.resnet

import torch
import torch.nn as nn


__all__ = ['PyramidNet', 'pyramidnet18', 'pyramidnet34', 'pyramidnet50', 'pyramidnet101',
           'pyramidnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_pyramidnet50_2', 'wide_pyramidnet101_2']


def conv3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3 convolution with padding"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1(in_planes, out_planes, stride=1):
    """1 convolution"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1):
        super(BasicBlock, self).__init__()
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.bn1 = nn.BatchNorm1d(inplanes)
        self.conv1 = conv3(inplanes, planes, stride)
        self.relu = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv2 = conv3(planes, planes)
        self.bn3 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        
        c_identity = identity.size(1)
        c_out = out.size(1)
        if c_identity != c_out:
            B = identity.size(0)
            s = identity.size(2)
            padding = torch.zeros(B, c_out - c_identity, s).to(identity.device)
            out += torch.cat([identity, padding], 1)
        else:
            out += identity

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1):
        super(Bottleneck, self).__init__()
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.bn1 = nn.BatchNorm1d(inplanes)
        self.conv1 = conv1(inplanes, width)
        self.bn2 = nn.BatchNorm1d(width)
        self.conv2 = conv3(width, width, stride, groups, dilation)
        self.bn3 = nn.BatchNorm1d(width)
        self.conv3 = conv1(width, width * self.expansion)
        self.bn4 = nn.BatchNorm1d(width * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.bn1(x)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        out = self.bn4(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        
        c_identity = identity.size(1)
        c_out = out.size(1)
        if c_identity != c_out:
            B = identity.size(0)
            s0 = identity.size(2)
            padding = torch.zeros(B, c_out - c_identity, s).to(identity.device)
            out += torch.cat([identity, padding], 1)
        else:
            out += identity

        return out


class PyramidNet(nn.Module):

    def __init__(self, block, layers, in_channels=3, num_classes=1000, alpha=48, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None):
        super(PyramidNet, self).__init__()

        self.alpha = alpha # wideing step factor
        self.N = sum(layers) 
        self.inplanes = 64
        self.outplanes = self.inplanes + int(self.alpha / self.N)
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv1d(in_channels, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm1d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], is_first=True)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(self.inplanes, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, is_first=False):
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1:#or self.inplanes != self.outplanes * block.expansion:
            downsample = nn.Sequential(
                conv1(self.inplanes, self.inplanes, stride),
                nn.BatchNorm1d(self.inplanes),
            )

        layers = []

        layers.append(block(self.inplanes, self.outplanes, stride, downsample, self.groups,
                            self.base_width, previous_dilation))
        
        self.inplanes = self.outplanes
        self.outplanes += int(self.alpha / self.N)

        for _ in range(1, blocks):
            layers.append(block(self.inplanes * block.expansion, self.outplanes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation))
            
            self.inplanes = self.outplanes
            self.outplanes += int(self.alpha / self.N)
        self.inplanes *= block.expansion

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


def _pyramidnet(arch, block, layers, **kwargs):
    model = PyramidNet(block, layers, **kwargs)
    return model


def pyramidnet18(**kwargs):
    return _pyramidnet('pyramidnet18', BasicBlock, [2, 2, 2, 2], **kwargs)


def pyramidnet34(**kwargs):
    return _pyramidnet('pyramidnet34', BasicBlock, [3, 4, 6, 3], **kwargs)


def pyramidnet50(**kwargs):
    return _pyramidnet('pyramidnet50', Bottleneck, [3, 4, 6, 3], **kwargs)

def pyramidnet101(**kwargs):
    return _pyramidnet('pyramidnet101', Bottleneck, [3, 4, 23, 3], **kwargs)


def pyramidnet152(**kwargs):
    return _pyramidnet('pyramidnet152', Bottleneck, [3, 8, 36, 3], **kwargs)


def resnext50_32x4d(**kwargs):
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _pyramidnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3], **kwargs)


def resnext101_32x8d(**kwargs):
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _pyramidnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3], **kwargs)


def wide_pyramidnet50_2(**kwargs):
    kwargs['width_per_group'] = 64 * 2
    return _pyramidnet('wide_pyramidnet50_2', Bottleneck, [3, 4, 6, 3], **kwargs)


def wide_pyramidnet101_2(**kwargs):
    kwargs['width_per_group'] = 64 * 2
    return _pyramidnet('wide_pyramidnet101_2', Bottleneck, [3, 4, 23, 3], **kwargs)

if __name__ == '__main__':
    import sys
    sys.path.append('./')

    device = 'cpu'
    model = pyramidnet34(in_channels=3, num_classes=10, alpha=200).to(device)

    n_params = 0
    for name, param in model.named_parameters():
        n = 1
        for s in param.size(): n *= s
        n_params += n
    print('Number of parameters: {}'.format(n_params))

    dummy = torch.ones(100, 3, 224, dtype=torch.float).to(device)

    out = model(dummy)

    print(out.size())
