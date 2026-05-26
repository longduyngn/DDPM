import torch
import torch.nn as nn
from src.modules import ResBlock, Attention, SinusoidalEmbeddings
import src.config

class UnetLayer(nn.Module):
    def __init__(self, upscale: bool, attention: bool, num_groups: int, 
                 dropout_prob: float, num_heads: int, C: int, time_emb_dim: int):
        super().__init__()
        self.ResBlock1 = ResBlock(C=C, time_emb_dim=time_emb_dim, num_groups=num_groups, dropout_prob=dropout_prob)
        self.ResBlock2 = ResBlock(C=C, time_emb_dim=time_emb_dim, num_groups=num_groups, dropout_prob=dropout_prob)
        
        if upscale:
            self.conv = nn.ConvTranspose2d(C, C // 2, kernel_size=4, stride=2, padding=1)
        else:
            self.conv = nn.Conv2d(C, C * 2, kernel_size=3, stride=2, padding=1)
            
        if attention:
            self.attention_layer = Attention(C=C, num_heads=num_heads, dropout_prob=dropout_prob)

    def forward(self, x, t_emb):
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
                C=src.config.CHANNELS[i],
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