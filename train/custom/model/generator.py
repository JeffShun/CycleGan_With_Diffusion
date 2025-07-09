import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

class DiffusionModel(nn.Module):

    def __init__(
        self,
        backbone,
        img_size,
        noise_steps=1000, 
        beta_schedule="linear"
        ):
        super(DiffusionModel, self).__init__()

        self.backbone = backbone
        self.img_size = img_size
        self.noise_steps = noise_steps
        self.beta_schedule = beta_schedule
        
        self.beta = self.prepare_noise_schedule()
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
        
        self.initialize_weights()

    def prepare_noise_schedule(self):
        if self.beta_schedule == "cosine":
            """
            cosine schedule as proposed in https://arxiv.org/abs/2102.09672
            """
            s = 0.008
            steps = self.noise_steps + 1
            x = torch.linspace(0, self.noise_steps, steps)
            alphas_cumprod = torch.cos(((x / self.noise_steps) + s) / (1 + s) * torch.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            return torch.clip(betas, 0.0001, 0.9999)
        
        elif self.beta_schedule == "quadratic":
            beta_start = 0.0001
            beta_end = 0.02
            return torch.linspace(beta_start**0.5, beta_end**0.5, self.noise_steps) ** 2
        
        elif self.beta_schedule == "sigmoid":
            beta_start = 0.0001
            beta_end = 0.02
            betas = torch.linspace(-6, 6, self.noise_steps)
            return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start
        
        else:
            beta_start = 0.0001
            beta_end = 0.02
            return torch.linspace(beta_start, beta_end, self.noise_steps)

    def noise_images(self, x, t):
        alpha_hat = self.alpha_hat.to(t.device)
        sqrt_alpha_hat = torch.sqrt(alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def ddim_sample(self, cond_img, ddim_timesteps=5, ddim_eta=0.0, discretization="quad"):
        device = cond_img.device
        self.img_size = cond_img.shape[-2:]
        # 计算DDIM时间步序列
        if discretization == 'uniform':
            c = self.noise_steps // ddim_timesteps
            ddim_timestep_seq = np.arange(0, self.noise_steps, c)
        elif discretization == 'quad':
            ddim_timestep_seq = (np.linspace(0, np.sqrt(self.noise_steps * 0.8), ddim_timesteps) ** 2).astype(int)
        else:
            raise NotImplementedError(f'未知的ddim离散方法: {discretization}')
        ddim_timestep_seq = ddim_timestep_seq + 1
        ddim_timestep_prev_seq = np.append(np.array([0]), ddim_timestep_seq[:-1])

        x = torch.randn((1, 1, self.img_size[0], self.img_size[1]), device=device)
        for i in tqdm(reversed(range(ddim_timesteps))):
            t = torch.full((1,), ddim_timestep_seq[i], device=device, dtype=torch.long)
            prev_t = torch.full((1,), ddim_timestep_prev_seq[i], device=device, dtype=torch.long)

            alpha_cumprod_t = self.alpha_hat.to(device).gather(0, t).float().reshape(1, 1, 1, 1)
            alpha_cumprod_prev = self.alpha_hat.to(device).gather(0, prev_t).float().reshape(1, 1, 1, 1)

            pred_x0 = self.forward(x, cond_img, t)

            sigma_t = ddim_eta * torch.sqrt((1 - alpha_cumprod_prev) / (1 - alpha_cumprod_t) * (1 - alpha_cumprod_t / alpha_cumprod_prev))
            dir_xt = torch.sqrt(1 - alpha_cumprod_prev - sigma_t ** 2) * (x - torch.sqrt(alpha_cumprod_t) * pred_x0) / torch.sqrt(1 - alpha_cumprod_t)
            noise = torch.randn_like(x) if i > 0 else 0

            x = torch.sqrt(alpha_cumprod_prev) * pred_x0 + dir_xt + sigma_t * noise

        x = x * 0.5 + 0.5
        return x

    def forward(self, x, cond, t):
        return self.backbone(x, cond, t)


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

