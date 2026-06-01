import sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
from timm.utils import ModelEmaV3

import src.config
from src.network import UNET
from src.modules import DDPM_Scheduler, display_reverse

def inference(checkpoint_path: str, num_samples: int = 5):
    print(f"[*] Loading checkpoint: {checkpoint_path} to {src.config.DEVICE}")
    checkpoint = torch.load(checkpoint_path, map_location=src.config.DEVICE)
    
    model = UNET().to(src.config.DEVICE)
    model.load_state_dict(checkpoint['weights'])
    
    # Khôi phục EMA (Mô hình EMA cho chất lượng ảnh mượt mà hơn)
    ema = ModelEmaV3(model, decay=src.config.EMA_DECAY)
    ema.load_state_dict(checkpoint['ema'])
    eval_model = ema.module.eval()
    s = sum([p.numel() for p in model.parameters()])
    print(f"[*] Number of parameters: {s}")

    scheduler = DDPM_Scheduler(
        num_time_steps=src.config.NUM_TIME_STEPS,
        schedule_type=src.config.BETA_SCHEDULE,
        beta_start=src.config.BETA_START,
        beta_end=src.config.BETA_END
    ).to(src.config.DEVICE)
    times = [0, 15, 50, 100, 200, 300, 400, 550, 700, 999]
    
    with torch.no_grad():
        z = torch.randn(num_samples, src.config.INPUT_CHANNELS, 32, 32, device=src.config.DEVICE)
        images = []
        
        # Khử nhiễu tuần tự từ NUM_TIME_STEPS - 1 về 0
        for t in reversed(range(0, src.config.NUM_TIME_STEPS)):
            t_tensor = torch.full((num_samples,), t, dtype=torch.long, device=src.config.DEVICE)
            
            alpha_t = scheduler.alpha[t_tensor].view(-1, 1, 1, 1)
            alpha_bar_t = scheduler.alpha_bar[t_tensor].view(-1, 1, 1, 1)
            beta_t = scheduler.beta[t_tensor].view(-1, 1, 1, 1)
            
            # Công thức tính toán bước Reverse bằng biến đã được vectorize
            temp = beta_t / (torch.sqrt(1 - alpha_bar_t) * torch.sqrt(alpha_t))
            z = (1 / torch.sqrt(alpha_t)) * z - (temp * eval_model(z, t_tensor))
            
            if t in times:
                images.append(z.clone())
            
            # Không cộng thêm nhiễu ở bước cuối cùng (t = 0)
            if t > 0:
                e = torch.randn(num_samples, src.config.INPUT_CHANNELS, 32, 32, device=src.config.DEVICE)
                z = z + (e * torch.sqrt(beta_t))
        
        print("[*] Inference complete. Saving sequences...")
        for s in range(num_samples):
            sample_images = [img[s:s+1] for img in images]
            save_img_path = os.path.join(src.config.REPORT_DIR, f'generated_sequence_{s+1}.png')
            display_reverse(sample_images, save_path=save_img_path)

if __name__ == '__main__':
    # Chạy inference với file checkpoint mong muốn (thường là best_ddpm.pt hoặc latest_ddpm.pt)
    ckpt_path = os.path.join(src.config.CHECKPOINT_DIR, 'best_ddpm.pt')
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(src.config.CHECKPOINT_DIR, 'latest_ddpm.pt')
        
    if os.path.exists(ckpt_path):
        inference(ckpt_path)
    else:
        print(f"[!] Checkpoint {ckpt_path} not found. Please train the model first.")