import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from timm.utils import ModelEmaV3

import config
from network import UNET
from modules import DDPM_Scheduler, display_reverse

def inference(checkpoint_path: str):
    print(f"[*] Loading checkpoint: {checkpoint_path} to {config.DEVICE}")
    checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE)
    
    model = UNET().to(config.DEVICE)
    model.load_state_dict(checkpoint['weights'])
    
    # Khôi phục EMA (Mô hình EMA cho chất lượng ảnh mượt mà hơn)
    ema = ModelEmaV3(model, decay=config.EMA_DECAY)
    ema.load_state_dict(checkpoint['ema'])
    eval_model = ema.module.eval()

    scheduler = DDPM_Scheduler(num_time_steps=config.NUM_TIME_STEPS).to(config.DEVICE)
    times = [0, 15, 50, 100, 200, 300, 400, 550, 700, 999]
    
    with torch.no_grad():
        for i in range(1): # Vẽ 1 mẫu
            z = torch.randn(1, 1, 32, 32, device=config.DEVICE)
            images = []
            
            for t in reversed(range(1, config.NUM_TIME_STEPS)):
                t_tensor = torch.tensor([t], device=config.DEVICE)
                
                alpha_t = scheduler.alpha[t_tensor]
                alpha_bar_t = scheduler.alpha_bar[t_tensor]
                beta_t = scheduler.beta[t_tensor]
                
                # Sửa công thức tính toán bước Reverse bằng biến đã được vectorize
                temp = beta_t / (torch.sqrt(1 - alpha_bar_t) * torch.sqrt(alpha_t))
                z = (1 / torch.sqrt(alpha_t)) * z - (temp * eval_model(z, t_tensor))
                
                if t in times:
                    images.append(z.clone())
                    
                e = torch.randn(1, 1, 32, 32, device=config.DEVICE)
                z = z + (e * torch.sqrt(beta_t))
            
            # Bước cuối cùng (t=0), không cộng thêm nhiễu
            t_tensor = torch.tensor([0], device=config.DEVICE)
            temp = scheduler.beta[0] / (torch.sqrt(1 - scheduler.alpha_bar[0]) * torch.sqrt(scheduler.alpha[0]))
            x = (1 / torch.sqrt(scheduler.alpha[0])) * z - (temp * eval_model(z, t_tensor))
            
            images.append(x)
            print("[*] Inference complete. Displaying sequence...")
            display_reverse(images)

if __name__ == '__main__':
    # Chạy inference với file checkpoint mong muốn (thường là best_ddpm.pt hoặc latest_ddpm.pt)
    ckpt_path = os.path.join(config.CHECKPOINT_DIR, 'best_ddpm.pt')
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(config.CHECKPOINT_DIR, 'latest_ddpm.pt')
        
    if os.path.exists(ckpt_path):
        inference(ckpt_path)
    else:
        print(f"[!] Checkpoint {ckpt_path} not found. Please train the model first.")