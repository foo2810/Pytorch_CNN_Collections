import torch
import torch.nn as nn

class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        return x

class InceptionResNetV2(nn.Module):
    def __init__(self, in_shape, n_classes, dropout_rate=0.5):
        super(InceptionResNetV2, self).__init__()
        
        self.in_shape = in_shape
        in_channels = in_shape[0]

        stage_a = IRV2StageA
        stage_b = IRV2StageB
        stage_c = IRV2StageC
        self.ir_base = InceptionResNetV2Base(in_channels, stage_a, stage_b, stage_c)
        self.aux = None

        in_ch = self._calc_last_feature_channels()
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_ch, n_classes)
        )
    
    def _calc_last_feature_channels(self):
        dummy = torch.zeros([2]+list(self.in_shape), dtype=torch.float)
        out = self.ir_base(dummy)
        return out.size(1)

    def forward(self, x):
        x = self.ir_base(x)
        x = self.classifier(x)
        return x

class InceptionResNetV2Base(nn.Module):
    def __init__(self, in_channels, stage_a, stage_b, stage_c):
        super(InceptionResNetV2Base, self).__init__()

        self.head_path = nn.Sequential(
            ConvBNReLU(in_channels, 32, kernel_size=3, stride=2, padding=3//2),
            ConvBNReLU(32, 32, kernel_size=3, stride=1, padding=3//2),
            ConvBNReLU(32, 64, kernel_size=3, stride=1, padding=3//2),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=3//2),
            ConvBNReLU(64, 80, kernel_size=1, stride=1, padding=0),
            ConvBNReLU(80, 192, kernel_size=3, stride=1, padding=3//2),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=3//2)
        )

        # Branches 1
        self.b1_conv11 = ConvBNReLU(192, 96, kernel_size=1, stride=1, padding=0)

        self.b1_conv21 = ConvBNReLU(192, 48, kernel_size=1, stride=1, padding=0)
        self.b1_conv22 = ConvBNReLU(48, 64, kernel_size=5, stride=1, padding=5//2)

        self.b1_conv31 = ConvBNReLU(192, 64, kernel_size=1, stride=1, padding=0)
        self.b1_conv32 = ConvBNReLU(64, 96, kernel_size=3, stride=1, padding=3//2)
        self.b1_conv33 = ConvBNReLU(96, 96, kernel_size=3, stride=1, padding=3//2)

        self.b1_avgpool41 = nn.AvgPool2d(kernel_size=3, stride=1, padding=3//2)
        self.b1_conv42 = ConvBNReLU(192, 64, kernel_size=1, stride=1, padding=0)

        self.stage_a_path = stage_a(320)

        # Branches 2
        self.b2_conv11 = ConvBNReLU(320, 384, kernel_size=3, stride=2, padding=3//2)

        self.b2_conv21 = ConvBNReLU(320, 256, kernel_size=1, stride=1, padding=0)
        self.b2_conv22 = ConvBNReLU(256, 256, kernel_size=3, stride=1, padding=3//2)
        self.b2_conv23 = ConvBNReLU(256, 384, kernel_size=3, stride=2, padding=3//2)

        self.b2_avgpool31 = nn.AvgPool2d(kernel_size=3, stride=2, padding=3//2)

        self.stage_b_path = stage_b(1088)


        # Branches 3
        self.b3_conv11 = ConvBNReLU(1088, 256, kernel_size=1, stride=1, padding=0)
        self.b3_conv12 = ConvBNReLU(256, 384, kernel_size=3, stride=2, padding=3//2)

        self.b3_conv21 = ConvBNReLU(1088, 256, kernel_size=1, stride=1, padding=0)
        self.b3_conv22 = ConvBNReLU(256, 288, kernel_size=3, stride=2, padding=3//2)
 
        self.b3_conv31 = ConvBNReLU(1088, 256, kernel_size=1, stride=1, padding=0)
        self.b3_conv32 = ConvBNReLU(256, 288, kernel_size=3, stride=1, padding=3//2)
        self.b3_conv33 = ConvBNReLU(288, 320, kernel_size=3, stride=2, padding=3//2)

        self.b3_maxpool31 = nn.MaxPool2d(kernel_size=3, stride=2, padding=3//2)

        self.stage_c_path = stage_c(2080)
        self.conv = ConvBNReLU(2080, 1536, kernel_size=1, stride=1, padding=0)


    def forward(self, x):
        x = self.head_path(x)

        # Branches 1
        b11 = self.b1_conv11(x)
        b12 = self.b1_conv22(self.b1_conv21(x))
        b13 = self.b1_conv33(self.b1_conv32(self.b1_conv31(x)))
        b14 = self.b1_conv42(self.b1_avgpool41(x))
        x = torch.cat([b11, b12, b13, b14], dim=1)

        x = self.stage_a_path(x)

        # Branches 2
        b21 = self.b2_conv11(x)
        b22 = self.b2_conv23(self.b2_conv22(self.b2_conv21(x)))
        b23 = self.b2_avgpool31(x)
        x = torch.cat([b21, b22, b23], dim=1)

        x = self.stage_b_path(x)

        # Branchex 3
        b31 = self.b3_conv12(self.b3_conv11(x))
        b32 = self.b3_conv22(self.b3_conv21(x))
        b33 = self.b3_conv33(self.b3_conv32(self.b3_conv31(x)))
        b34 = self.b3_maxpool31(x)
        x = torch.cat([b31, b32, b33, b34], dim=1)

        x = self.stage_c_path(x)

        x = self.conv(x)

        return x

class IRV2StageA(nn.Module):
    def __init__(self, in_channels):
        super(IRV2StageA, self).__init__()
        self.stage = nn.Sequential(
            *[InceptionModule35(in_channels=in_channels, scale=0.17, activation_fn=nn.ReLU) for _ in range(10)]
        )
    
    def forward(self, x):
        return self.stage(x)

class IRV2StageB(nn.Module):
    def __init__(self, in_channels):
        super(IRV2StageB, self).__init__()
        self.stage = nn.Sequential(
            *[InceptionModule17(in_channels=in_channels, scale=0.1, activation_fn=nn.ReLU) for _ in range(20)]
        )
    
    def forward(self, x):
        return self.stage(x)

class IRV2StageC(nn.Module):
    def __init__(self, in_channels):
        super(IRV2StageC, self).__init__()
        self.stage= nn.Sequential(
            *[InceptionModule8(in_channels=in_channels, scale=0.2, activation_fn=nn.ReLU) for _ in range(9)]
        )
        self.block = InceptionModule8(in_channels=in_channels, scale=1.0, activation_fn=None)
    
    def forward(self, x):
        x = self.stage(x)
        x = self.block(x)
        return x

class InceptionModule35(nn.Module):
    def __init__(self, in_channels, scale=1.0, activation_fn=nn.ReLU):
        super(InceptionModule35, self).__init__()

        self.scale = scale

        self.conv11 = ConvBNReLU(in_channels, 32, kernel_size=1, stride=1, padding=0)

        self.conv21 = ConvBNReLU(in_channels, 32, kernel_size=1, stride=1, padding=0)
        self.conv22 = ConvBNReLU(32, 32, kernel_size=3, stride=1, padding=3//2)

        self.conv31 = ConvBNReLU(in_channels, 32, kernel_size=1, stride=1, padding=0)
        self.conv32 = ConvBNReLU(32, 48, kernel_size=3, stride=1, padding=3//2)
        self.conv33 = ConvBNReLU(48, 64, kernel_size=3, stride=1, padding=3//2)

        self.conv_f = ConvBNReLU(32+32+64, in_channels, kernel_size=1, stride=1, padding=0)
        if activation_fn is not None:
            self.activation = activation_fn()
        else:
            self.activation = None
    
    def forward(self, x):
        identity = x
        b1 = self.conv11(x)
        b2 = self.conv22(self.conv21(x))
        b3 = self.conv33(self.conv32(self.conv31(x)))
        x = torch.cat([b1, b2, b3], dim=1)
        scaled_up = self.conv_f(x) * self.scale
        x = identity + scaled_up
        if self.activation is not None:
            x = self.activation(x)

        return x

class InceptionModule17(nn.Module):
    def __init__(self, in_channels, scale=1.0, activation_fn=nn.ReLU):
        super(InceptionModule17, self).__init__()

        self.scale = scale

        self.conv11 = ConvBNReLU(in_channels, 192, kernel_size=1, stride=1, padding=0)

        self.conv21 = ConvBNReLU(in_channels, 128, kernel_size=1, stride=1, padding=0)
        self.conv22 = ConvBNReLU(128, 160, kernel_size=(1, 7), stride=1, padding=(0, 7//2))
        self.conv23 = ConvBNReLU(160, 192, kernel_size=(7, 1), stride=1, padding=(7//2, 0))

        self.conv_f = ConvBNReLU(192+192, in_channels, kernel_size=1, stride=1, padding=0)
        if activation_fn is not None:
            self.activation = activation_fn()
        else:
            self.activation = None
    
    def forward(self, x):
        identity = x
        b1 = self.conv11(x)
        b2 = self.conv23(self.conv22(self.conv21(x)))
        x = torch.cat([b1, b2], dim=1)
        scaled_up = self.conv_f(x) * self.scale
        x = identity + scaled_up
        if self.activation is not None:
            x = self.activation(x)

        return x

class InceptionModule8(nn.Module):
    def __init__(self, in_channels, scale=1.0, activation_fn=nn.ReLU):
        super(InceptionModule8, self).__init__()
 
        self.scale = scale

        self.conv11 = ConvBNReLU(in_channels, 192, kernel_size=1, stride=1, padding=0)

        self.conv21 = ConvBNReLU(in_channels, 192, kernel_size=1, stride=1, padding=0)
        self.conv22 = ConvBNReLU(192, 224, kernel_size=(1, 3), stride=1, padding=(0, 3//2))
        self.conv23 = ConvBNReLU(224, 256, kernel_size=(3, 1), stride=1, padding=(3//2, 0))

        self.conv_f = ConvBNReLU(192+256, in_channels, kernel_size=1, stride=1, padding=0)
        if activation_fn is not None:
            self.activation = activation_fn()
        else:
            self.activation = None
    
    def forward(self, x):
        identity = x
        b1 = self.conv11(x)
        b2 = self.conv23(self.conv22(self.conv21(x)))
        x = torch.cat([b1, b2], dim=1)
        scaled_up = self.conv_f(x) * self.scale
        x = identity + scaled_up
        if self.activation is not None:
            x = self.activation(x)

        return x


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = InceptionResNetV2(in_shape=(3, 100, 100), n_classes=10).to(device)
    dummy = torch.zeros((50, 3, 100, 100), dtype=torch.float).to(device)

    out = net(dummy)

    print(out.size())

 