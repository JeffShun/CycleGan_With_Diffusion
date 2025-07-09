import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionBlock(nn.Module):
    """
    Attention模块
    和Transformer中的multi-head attention原理及实现方式一致
    """

    def __init__(self, n_channels: int, n_heads: int = 1, n_groups: int = 32):
        """
        Params:
            n_channels：等待做attention操作的特征图的channel数
            n_heads：   attention头数
            d_k：       每一个attention头处理的向量维度
            n_groups：  Group Norm超参数
        """
        super().__init__()

        # 一般而言，d_k = n_channels // n_heads，需保证n_channels能被n_heads整除
        d_k = n_channels // n_heads
        self.norm = nn.GroupNorm(n_groups, n_channels)
        self.projection = nn.Linear(n_channels, n_heads * d_k * 3)
        self.output = nn.Linear(n_heads * d_k, n_channels)
        
        self.scale = d_k ** -0.5
        self.n_heads = n_heads
        self.d_k = d_k

    def forward(self, x):
        """
        Params:
            x: 输入数据xt，尺寸大小为（batch_size, in_channels, height, width）
            t: 输入数据t，尺寸大小为（batch_size, time_c）
        """

        batch_size, n_channels, height, width = x.shape
        x = x.view(batch_size, n_channels, -1).permute(0, 2, 1)
        qkv = self.projection(x).view(batch_size, -1, self.n_heads, 3 * self.d_k)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        attn = torch.einsum('bihd,bjhd->bijh', q, k) * self.scale
        attn = attn.softmax(dim=2)
        res = torch.einsum('bijh,bjhd->bihd', attn, v)
        res = res.reshape(batch_size, -1, self.n_heads * self.d_k)
        res = self.output(res)

        # 残差连接
        res += x

        res = res.permute(0, 2, 1).reshape(batch_size, n_channels, height, width)

        return res


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, t):
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, out_channels),
            DoubleConv(out_channels, out_channels)
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class UNet(nn.Module):
    def __init__(self, c_in=1, c_out=1, base_channel=32, attention_heads=8, time_dim=256):
        super().__init__()
        self.time_dim = time_dim
        self.inc = DoubleConv(c_in, base_channel)
        self.down1 = Down(base_channel, base_channel*2)
        self.down2 = Down(base_channel*2, base_channel*4)
        self.down3 = Down(base_channel*4, base_channel*8)

        self.bot1 = DoubleConv(base_channel*8, base_channel*8)
        self.bot2 = DoubleConv(base_channel*8, base_channel*8)
        self.sa4 = AttentionBlock(base_channel*8, attention_heads)

        self.up1 = Up(base_channel*12, base_channel*4)
        self.up2 = Up(base_channel*6, base_channel*2)
        self.up3 = Up(base_channel*3, base_channel*1)
        self.outc = nn.Conv2d(base_channel*1, c_out, kernel_size=1)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels)).to(t.device)
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, cond_img, t):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim).to(x.dtype).to(x.device)
        x = torch.cat([x, cond_img], dim=1)

        x1 = self.inc(x)
        x2 = self.down1(x1, t)  
        x3 = self.down2(x2, t)
        x4 = self.down3(x3, t)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.sa4(x4)

        x = self.up1(x4, x3, t)
        x = self.up2(x, x2, t)
        x = self.up3(x, x1, t)
        output = self.outc(x)
        return torch.tanh(output)
    
if __name__ == "__main__":
    model = UNet(c_in=2, c_out=1, base_channel=32, attention_heads=8, time_dim=256)
    x = torch.randn(1, 1, 256, 256)
    cond_img = torch.randn(1, 1, 256, 256)
    t = torch.randn(1)
    output = model(x, cond_img, t)
    print(output.shape)