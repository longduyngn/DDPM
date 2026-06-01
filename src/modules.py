import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math
import matplotlib.pyplot as plt

class SinusoidalEmbeddings(nn.Module):
    def __init__(self, time_steps: int, embed_dim: int):
        super().__init__()
        position = torch.arange(time_steps).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        embeddings = torch.zeros(time_steps, embed_dim, requires_grad=False)
        embeddings[:, 0::2] = torch.sin(position * div)
        embeddings[:, 1::2] = torch.cos(position * div)
        # Sử dụng register_buffer để PyTorch tự động quản lý thiết bị (CPU/GPU)
        self.register_buffer('embeddings', embeddings)

    def forward(self, t):
        # Trả về tensor (Batch, embed_dim)
        return self.embeddings[t]

class ResBlock(nn.Module):
    def __init__(self, C: int, time_emb_dim: int, num_groups: int, dropout_prob: float):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.GroupNorm(num_groups=num_groups, num_channels=C),
            nn.SiLU(),
            nn.Conv2d(C, C, kernel_size=3, padding=1)
        )
        
        # Linear projection để ép chuẩn Time Embedding khớp với C
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, C)
        )
        
        self.block2 = nn.Sequential(
            nn.GroupNorm(num_groups=num_groups, num_channels=C),
            nn.SiLU(),
            nn.Dropout(p=dropout_prob, inplace=True),
            nn.Conv2d(C, C, kernel_size=3, padding=1)
        )

    def forward(self, x, t_emb):
        h = self.block1(x)
        # Phóng chiếu và broadcast Time Embedding
        t = self.time_mlp(t_emb)[:, :, None, None]
        h = h + t
        h = self.block2(h)
        return h + x

class Attention(nn.Module):
    def __init__(self, C: int, num_heads: int, dropout_prob: float):
        super().__init__()
        self.proj1 = nn.Linear(C, C * 3)
        self.proj2 = nn.Linear(C, C)
        self.num_heads = num_heads
        self.dropout_prob = dropout_prob

    def forward(self, x):
        b, c, h, w = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.proj1(x)
        d = c // self.num_heads
        x = rearrange(x, 'b L (K H d) -> K b H L d', K=3, H=self.num_heads, d=d)
        q, k, v = x[0], x[1], x[2]
        x = F.scaled_dot_product_attention(q, k, v, is_causal=False, dropout_p=self.dropout_prob)
        x = rearrange(x, 'b H (h w) d -> b (h w) (H d)', h=h, w=w, d=d)
        x = self.proj2(x)
        return rearrange(x, 'b (h w) C -> b C h w', h=h, w=w)

class DDPM_Scheduler(nn.Module):
    def __init__(self, num_time_steps: int = 1000, schedule_type: str = 'linear', beta_start: float = 1e-4, beta_end: float = 0.02):
        super().__init__()
        
        if schedule_type == 'linear' or schedule_type == 'nondecreasing':
            beta = torch.linspace(beta_start, beta_end, num_time_steps, requires_grad=False)
        elif schedule_type == 'non-increasing':
            beta = torch.linspace(beta_end, beta_start, num_time_steps, requires_grad=False)
        elif schedule_type == 'cosine':
            # Cosine schedule (Improved DDPM - Nichol & Dhariwal 2021)
            s = 0.008
            steps = num_time_steps + 1
            t = torch.linspace(0, num_time_steps, steps, dtype=torch.float64)
            alphas_cumprod = torch.cos(((t / num_time_steps) + s) / (1 + s) * math.pi / 2) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            beta = torch.clip(betas, 0, 0.999).float()
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")
            
        alpha = 1.0 - beta
        alpha_bar = torch.cumprod(alpha, dim=0) # Đã chuẩn hóa thành alpha_bar (tích lũy)
        
        self.register_buffer('beta', beta)
        self.register_buffer('alpha', alpha)
        self.register_buffer('alpha_bar', alpha_bar)

def display_reverse(images: list, save_path: str = None):
    fig, axes = plt.subplots(1, len(images), figsize=(len(images), 1))
    for i, ax in enumerate(axes.flat):
        x = images[i].squeeze(0).cpu()
        x = rearrange(x, 'c h w -> h w c').numpy()
        # Chuẩn hóa ngược từ [-1, 1] về [0, 1] để hiển thị
        x = (x + 1.0) / 2.0
        if x.shape[-1] == 1:
            ax.imshow(x.squeeze(-1), cmap='gray')
        else:
            ax.imshow(x)
        ax.axis('off')
    
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"[*] Saved reverse diffusion process to {save_path}")
    else:
        plt.show()
    plt.close(fig)