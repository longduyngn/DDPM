import torch
import torch.nn as nn
from src.modules import ResBlock, Attention, SinusoidalEmbeddings
import src.config

class UnetLayer(nn.Module):
    def __init__(self, upscale: bool, attention: bool, num_groups: int, 
                 dropout_prob: float, num_heads: int, C_in: int, C_out: int, time_emb_dim: int):
        super().__init__()
        # Thêm 1x1 conv projection nếu kênh đầu vào (sau khi concat) khác kênh xử lý đích
        if C_in != C_out:
            self.proj = nn.Conv2d(C_in, C_out, kernel_size=1)
        else:
            self.proj = nn.Identity()
            
        self.ResBlock1 = ResBlock(C=C_out, time_emb_dim=time_emb_dim, num_groups=num_groups, dropout_prob=dropout_prob)
        self.ResBlock2 = ResBlock(C=C_out, time_emb_dim=time_emb_dim, num_groups=num_groups, dropout_prob=dropout_prob)
        
        if upscale:
            self.conv = nn.ConvTranspose2d(C_out, C_out // 2, kernel_size=4, stride=2, padding=1)
        else:
            self.conv = nn.Conv2d(C_out, C_out * 2, kernel_size=3, stride=2, padding=1)
            
        if attention:
            self.attention_layer = Attention(C=C_out, num_heads=num_heads, dropout_prob=dropout_prob)

    def forward(self, x, t_emb):
        x = self.proj(x)
        x = self.ResBlock1(x, t_emb)
        if hasattr(self, 'attention_layer'):
            x = self.attention_layer(x)
        x = self.ResBlock2(x, t_emb)
        return self.conv(x), x

class UNET(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_layers = len(src.config.CHANNELS)
        self.shallow_conv = nn.Conv2d(src.config.INPUT_CHANNELS, src.config.CHANNELS[0], kernel_size=3, padding=1)
        
        # Tính toán động các kênh đầu vào của mỗi tầng
        C_in_list = []
        for i in range(self.num_layers):
            if i == 0:
                C_in = src.config.CHANNELS[0]
            elif i < self.num_layers // 2:
                C_in = src.config.CHANNELS[i-1] * 2
            elif i == self.num_layers // 2:
                C_in = src.config.CHANNELS[i-1] * 2
            else:
                C_in = (src.config.CHANNELS[i-1] // 2) + src.config.CHANNELS[self.num_layers - i]
            C_in_list.append(C_in)
            
        out_channels = (src.config.CHANNELS[-1] // 2) + src.config.CHANNELS[0]
        self.late_conv = nn.Conv2d(out_channels, out_channels // 2, kernel_size=3, padding=1)
        self.output_conv = nn.Conv2d(out_channels // 2, src.config.OUTPUT_CHANNELS, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        
        self.embeddings = SinusoidalEmbeddings(time_steps=src.config.NUM_TIME_STEPS, embed_dim=src.config.TIME_EMB_DIM)
        
        for i in range(self.num_layers):
            layer = UnetLayer(
                upscale=src.config.UPSCALES[i],
                attention=src.config.ATTENTIONS[i],
                num_groups=src.config.NUM_GROUPS,
                dropout_prob=src.config.DROPOUT_PROB,
                C_in=C_in_list[i],
                C_out=src.config.CHANNELS[i],
                num_heads=src.config.NUM_HEADS,
                time_emb_dim=src.config.TIME_EMB_DIM
            )
            setattr(self, f'Layer{i+1}', layer)

    def forward(self, x, t):
        x = self.shallow_conv(x)
        t_emb = self.embeddings(t) # (Batch, TIME_EMB_DIM)
        residuals = []
        
        # Nhánh Downsample
        for i in range(self.num_layers // 2):
            layer = getattr(self, f'Layer{i+1}')
            x, r = layer(x, t_emb)
            residuals.append(r)
            
        # Nhánh Upsample
        for i in range(self.num_layers // 2, self.num_layers):
            layer = getattr(self, f'Layer{i+1}')
            x = torch.concat((layer(x, t_emb)[0], residuals[self.num_layers - i - 1]), dim=1)
            
        return self.output_conv(self.relu(self.late_conv(x)))