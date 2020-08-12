import torch
import torch.nn as nn

from .inception_resnet import InceptionResNetV2Base, InceptionModule35, InceptionModule17, InceptionModule8

class PolyNet(nn.Module):
    def __init__(self, in_shape, n_classes, dropout_rate=0.5):
        super(PolyNet, self).__init__()

        self.in_shape = in_shape
        in_channels = in_shape[0]

        stage_a = PolyInceptionModule_stage_a
        stage_b = PolyInceptionModule_stage_b
        stage_c = PolyInceptionModule_stage_c
        self.ir_base = InceptionResNetV2Base(in_channels, stage_a, stage_b, stage_c)
        self.aux = None

        in_ch = self._calc_last_feature_channels()
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
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

class PolyInceptionModule_2way(nn.Module):
    def __init__(self, module, is_activated=True, **kwargs):
        super(PolyInceptionModule_2way, self).__init__()
        self.mod1 = module(**kwargs)
        self.mod2 = module(**kwargs)
        if is_activated:
            self.activation = nn.ReLU(inplace=True)
        else:
            self.activation = None
    
    def forward(self, x):
        o1 = self.mod1(x)
        o2 = self.mod2(x)
        x = x + o1 + o2
        if self.activation is not None:
            x = self.activation(x)
        
        return x

class PolyInceptionModule_poly3(nn.Module):
    def __init__(self, module, is_activated=True, **kwargs):
        super(PolyInceptionModule_poly3, self).__init__()
        self.mod = module(**kwargs)
        if is_activated:
            self.activation = nn.ReLU(inplace=True)
        else:
            self.activation = None
    
    def forward(self, x):
        o1 = self.mod(x)
        o2 = self.mod(o1)
        o3 = self.mod(o2)
        x = x + o1 + o2 + o3 
        if self.activation is not None:
            x = self.activation(x)
        
        return x


class PolyInceptionModule_stage_a(nn.Module):
    def __init__(self, in_channels):
        super(PolyInceptionModule_stage_a, self).__init__()
        self.stage = nn.Sequential(
            *[PolyInceptionModule_2way(InceptionModule35, in_channels=in_channels, scale=0.17, activation_fn=nn.ReLU) for _ in range(10)]
        )
    
    def forward(self, x):
        return self.stage(x)

class PolyInceptionModule_stage_b(nn.Module):
    def __init__(self, in_channels):
        super(PolyInceptionModule_stage_b, self).__init__()
        modules = []
        for _ in range(10):
            modules += [
                PolyInceptionModule_poly3(InceptionModule17, in_channels=in_channels, scale=0.17, activation_fn=nn.ReLU),
                PolyInceptionModule_2way(InceptionModule17, in_channels=in_channels, scale=0.17, activation_fn=nn.ReLU),
            ]
        self.stage = nn.Sequential(*modules)
    
    def forward(self, x):
        return self.stage(x)

class PolyInceptionModule_stage_c(nn.Module):
    def __init__(self, in_channels):
        super(PolyInceptionModule_stage_c, self).__init__()
        modules = []
        for _ in range(4):
            modules += [
                PolyInceptionModule_poly3(InceptionModule8, in_channels=in_channels, scale=0.2, activation_fn=nn.ReLU),
                PolyInceptionModule_2way(InceptionModule8, in_channels=in_channels, scale=0.2, activation_fn=nn.ReLU),
            ]
        modules += [
            PolyInceptionModule_poly3(InceptionModule8, in_channels=in_channels, scale=0.2, activation_fn=nn.ReLU),
            PolyInceptionModule_2way(InceptionModule8, in_channels=in_channels, scale=1., activation_fn=None),
        ]
        self.stage = nn.Sequential(*modules)
    
    def forward(self, x):
        return self.stage(x)

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = PolyNet(in_shape=(3, 100), n_classes=10).to(device)
    dummy = torch.zeros((50, 3, 100), dtype=torch.float).to(device)

    out = net(dummy)

    print(out.size())

 