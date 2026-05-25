import sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# Đưa thư mục gốc vào đường dẫn để import được các file ở cấp trên
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from timm.utils import ModelEmaV3
from tqdm import tqdm
import random
import numpy as np

import config
from network import UNET
from modules import DDPM_Scheduler

def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def train(resume_checkpoint=None):
    set_seed(config.SEED)
    
    # Chuẩn hóa về [-1, 1]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    train_dataset = datasets.MNIST(root=config.DATA_DIR, train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, 
                              drop_last=True, num_workers=config.NUM_WORKERS)

    scheduler = DDPM_Scheduler(num_time_steps=config.NUM_TIME_STEPS).to(config.DEVICE)
    model = UNET()
    
    # --- GIAI ĐOẠN 3: TỰ ĐỘNG ĐA GPU (Multi-GPU Allocation) ---
    if config.NUM_GPUS > 1:
        print(f"[*] Detected {config.NUM_GPUS} GPUs. Wrapping model with DataParallel.")
        model = nn.DataParallel(model)
    else:
        print(f"[*] Using {config.DEVICE}.")
        
    model = model.to(config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=config.LR)
    ema = ModelEmaV3(model, decay=config.EMA_DECAY)
    
    start_epoch = 0
    if resume_checkpoint is not None and os.path.exists(resume_checkpoint):
        checkpoint = torch.load(resume_checkpoint, map_location=config.DEVICE)
        # Xử lý an toàn tiền tố 'module.' nếu load checkpoint từ Multi-GPU sang Single-GPU
        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(checkpoint['weights'])
        else:
            model.load_state_dict(checkpoint['weights'])
        ema.load_state_dict(checkpoint['ema'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint.get('epoch', 0)
        print(f"[*] Resumed from epoch {start_epoch}")

    criterion = nn.MSELoss(reduction='mean')

    for i in range(start_epoch, config.NUM_EPOCHS):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {i+1}/{config.NUM_EPOCHS}")
        
        for bidx, (x, _) in enumerate(pbar):
            # Cấp phát thiết bị động
            x = x.to(config.DEVICE)
            x = F.pad(x, (2, 2, 2, 2))
            
            t = torch.randint(0, config.NUM_TIME_STEPS, (config.BATCH_SIZE,), device=config.DEVICE)
            e = torch.randn_like(x) # Phân phối chuẩn
            
            # Lấy alpha_bar (tích lũy)
            a_bar = scheduler.alpha_bar[t].view(config.BATCH_SIZE, 1, 1, 1)
            x_noisy = (torch.sqrt(a_bar) * x) + (torch.sqrt(1 - a_bar) * e)
            
            output = model(x_noisy, t)
            
            optimizer.zero_grad()
            loss = criterion(output, e)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            ema.update(model)
            
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        print(f'Epoch {i+1} | Avg Loss: {total_loss / len(train_loader):.5f}')

        # --- Trích xuất State Dict an toàn ---
        # Nếu model là DataParallel, ta chỉ lưu phần `module` bên trong để tương thích khi load
        model_state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
        
        checkpoint = {
            'epoch': i + 1,
            'weights': model_state,
            'optimizer': optimizer.state_dict(),
            'ema': ema.state_dict()
        }
        torch.save(checkpoint, os.path.join(config.CHECKPOINT_DIR, f'ddpm_epoch_{i+1}.pt'))

if __name__ == '__main__':
    train()