import sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# Đưa thư mục chứa train.py vào sys.path để import được gói src từ bất kỳ thư mục làm việc nào
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

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
from torch.amp import autocast, GradScaler
import matplotlib.pyplot as plt

import src.config
from src.network import UNET
from src.modules import DDPM_Scheduler

def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def train(resume_checkpoint=None):
    set_seed(src.config.SEED)
    
    # Chuẩn hóa về [-1, 1]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    train_dataset = datasets.MNIST(root=src.config.DATA_DIR, train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=src.config.BATCH_SIZE, shuffle=True, 
                              drop_last=True, num_workers=src.config.NUM_WORKERS, pin_memory=True)

    scheduler = DDPM_Scheduler(num_time_steps=src.config.NUM_TIME_STEPS).to(src.config.DEVICE)
    model = UNET()
    
    # --- GIAI ĐOẠN 3: TỰ ĐỘNG ĐA GPU (Multi-GPU Allocation) ---
    if src.config.NUM_GPUS > 1:
        print(f"[*] Detected {src.config.NUM_GPUS} GPUs. Wrapping model with DataParallel.")
        model = nn.DataParallel(model)
    else:
        print(f"[*] Using {src.config.DEVICE}.")
        
    model = model.to(src.config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=src.config.LR)
    
    # Khởi tạo EMA trên base model để tránh lưu checkpoint có tiền tố 'module.' trên Multi-GPU
    base_model = model.module if isinstance(model, nn.DataParallel) else model
    ema = ModelEmaV3(base_model, decay=src.config.EMA_DECAY)
    
    # Cấu hình AMP và best loss (chỉ kích hoạt scaler thực sự khi dùng CUDA)
    scaler = GradScaler(device=src.config.DEVICE.type, enabled=(src.config.DEVICE.type == 'cuda'))
    best_loss = float('inf')
    train_losses = []
    
    if resume_checkpoint == 'latest':
        latest_path = os.path.join(src.config.CHECKPOINT_DIR, 'latest_ddpm.pt')
        if os.path.exists(latest_path):
            resume_checkpoint = latest_path
        else:
            resume_checkpoint = None

    start_epoch = 0
    if resume_checkpoint is not None and os.path.exists(resume_checkpoint):
        checkpoint = torch.load(resume_checkpoint, map_location=src.config.DEVICE)
        # Xử lý an toàn tiền tố 'module.' nếu load checkpoint từ Multi-GPU sang Single-GPU
        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(checkpoint['weights'])
        else:
            model.load_state_dict(checkpoint['weights'])
        ema.load_state_dict(checkpoint['ema'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if 'scaler' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler'])
        best_loss = checkpoint.get('best_loss', float('inf'))
        train_losses = checkpoint.get('train_losses', [])
        start_epoch = checkpoint.get('epoch', 0)
        print(f"[*] Resumed from epoch {start_epoch} (Best Loss: {best_loss:.5f})")

    criterion = nn.MSELoss(reduction='mean')

    for i in range(start_epoch, src.config.NUM_EPOCHS):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {i+1}/{src.config.NUM_EPOCHS}")
        
        for bidx, (x, _) in enumerate(pbar):
            # Cấp phát thiết bị động
            x = x.to(src.config.DEVICE, non_blocking=True)
            x = F.pad(x, (2, 2, 2, 2))
            
            # Sử dụng x.size(0) thay cho config.BATCH_SIZE để an toàn nếu chạy batch cuối hoặc chia batch trên Multi-GPU
            curr_batch_size = x.size(0)
            t = torch.randint(0, src.config.NUM_TIME_STEPS, (curr_batch_size,), device=src.config.DEVICE)
            e = torch.randn_like(x) # Phân phối chuẩn
            
            # Lấy alpha_bar (tích lũy)
            a_bar = scheduler.alpha_bar[t].view(curr_batch_size, 1, 1, 1)
            x_noisy = (torch.sqrt(a_bar) * x) + (torch.sqrt(1 - a_bar) * e)
            
            optimizer.zero_grad()
            
            # Autocast phục vụ mixed precision (chỉ bật khi dùng CUDA)
            device_type = src.config.DEVICE.type
            enabled = (device_type == 'cuda')
            with autocast(device_type=device_type, enabled=enabled):
                output = model(x_noisy, t)
                loss = criterion(output, e)
            
            total_loss += loss.item()
            
            # Lan truyền ngược và tối ưu hóa sử dụng GradScaler
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            ema.update(model)
            
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {i+1} | Avg Loss: {avg_loss:.5f}')
        train_losses.append(avg_loss)

        # --- Trích xuất State Dict an toàn ---
        # Nếu model là DataParallel, ta chỉ lưu phần `module` bên trong để tương thích khi load
        model_state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
        
        checkpoint = {
            'epoch': i + 1,
            'weights': model_state,
            'optimizer': optimizer.state_dict(),
            'ema': ema.state_dict(),
            'scaler': scaler.state_dict(),
            'best_loss': best_loss,
            'train_losses': train_losses
        }
        
        # 1. Luôn lưu đè vào file latest_ddpm.pt để có thể resume
        latest_path = os.path.join(src.config.CHECKPOINT_DIR, 'latest_ddpm.pt')
        torch.save(checkpoint, latest_path)
        
        # 2. Chỉ lưu best_ddpm.pt nếu loss đạt mức thấp nhất mới
        if avg_loss < best_loss:
            best_loss = avg_loss
            checkpoint['best_loss'] = best_loss
            best_path = os.path.join(src.config.CHECKPOINT_DIR, 'best_ddpm.pt')
            torch.save(checkpoint, best_path)
            print(f"[*] New best loss ({best_loss:.5f}). Saved to {best_path}")
            
        # 3. Chỉ lưu checkpoint định kỳ mỗi 10 epoch hoặc ở epoch cuối cùng
        if (i + 1) % 10 == 0 or (i + 1) == src.config.NUM_EPOCHS:
            epoch_path = os.path.join(src.config.CHECKPOINT_DIR, f'ddpm_epoch_{i+1}.pt')
            torch.save(checkpoint, epoch_path)
            print(f"[*] Saved periodic checkpoint to {epoch_path}")

    # 4. Vẽ biểu đồ loss và lưu lại sau khi kết thúc huấn luyện
    if len(train_losses) > 0:
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Train Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('DDPM Training Loss Curve')
        plt.legend()
        plt.grid(True)
        loss_plot_path = os.path.join(src.config.REPORT_DIR, 'loss_plot.png')
        plt.savefig(loss_plot_path, bbox_inches='tight')
        plt.close()
        print(f"[*] Saved loss plot to {loss_plot_path}")

if __name__ == '__main__':
    train()