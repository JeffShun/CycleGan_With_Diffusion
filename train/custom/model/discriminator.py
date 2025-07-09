import torch.nn as nn
from torch.nn.utils import spectral_norm


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, use_act=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=True)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True) if use_act else nn.Identity()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))
    

"""
GlobalDiscriminator 一般用于图像整体转换，任务对全局信息以来较高
"""
class GlobalDiscriminator(nn.Module):
    def __init__(self, in_channels=1, base_channels=64, num_layers=4):
        super().__init__()

        # 每层两个卷积，共 2 * num_layers 个 block
        features = []
        for i in range(num_layers):
            features += [base_channels * (2 ** i)] * 2 

        blocks = []
        for idx, feature in enumerate(features):
            blocks.append(
                ConvBlock(
                    in_channels,
                    feature,
                    kernel_size=3,
                    stride=1 + idx % 2,  
                    padding=1,
                    use_act=True,
                )
            )
            in_channels = feature

        self.blocks = nn.Sequential(*blocks)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(features[-1], 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1)
        )


    def forward(self, x):
        x = self.blocks(x)
        x = self.classifier(x)
        return x

    def set_require_grad(self, flag=True):
        for p in self.parameters():
            p.requires_grad = flag

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.weight is not None:
                    nn.init.kaiming_normal_(m.weight.data, mode="fan_out", nonlinearity="leaky_relu")
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()                    
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, 0, 0.01)
                m.bias.data.zero_()


class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=1, base_channels=64, num_layers=5):
        super(PatchDiscriminator, self).__init__()
        layers = []

        # 第一层不加 InstanceNorm
        layers.append(nn.Conv2d(in_channels, base_channels, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        channels = base_channels
        for i in range(1, num_layers):
            next_channels = min(channels * 2, 512)
            stride = 1 if i == num_layers - 1 else 2  # 最后一层 stride=1 保留空间信息
            layers.append(nn.Conv2d(channels, next_channels, kernel_size=4, stride=stride, padding=1))
            layers.append(nn.InstanceNorm2d(next_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            channels = next_channels

        # 最后一层输出 1 个通道（PatchGAN）
        layers.append(nn.Conv2d(channels, 1, kernel_size=4, stride=1, padding=1))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

    def set_require_grad(self, flag=True):
        for p in self.parameters():
            p.requires_grad = flag

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.weight is not None:
                    nn.init.kaiming_normal_(m.weight.data, mode="fan_out", nonlinearity="leaky_relu")
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()                    
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, 0, 0.01)
                m.bias.data.zero_()
