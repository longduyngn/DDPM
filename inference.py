import sys
import os
import numpy as np
import torch
import torchvision
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from timm.utils import ModelEmaV3

import src.config
from src.network import UNET
from src.modules import DDPM_Scheduler, display_reverse

def inference(checkpoint_path: str, num_samples: int = 25):
    # Set default values from config
    ddim_steps = src.config.DDIM_STEPS
    eta = src.config.DDIM_ETA

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
    
    # Tạo chuỗi các bước thời gian cho DDIM theo kiểu quadratic (tau = c * i^2)
    if ddim_steps > 1:
        i_space = np.arange(ddim_steps)
        c = (src.config.NUM_TIME_STEPS - 1) / ((ddim_steps - 1) ** 2)
        ddim_timesteps = np.round(c * (i_space ** 2)).astype(int)
    else:
        ddim_timesteps = np.array([src.config.NUM_TIME_STEPS - 1])
    
    # Chọn các bước thời gian cần lưu để hiển thị quá trình khử nhiễu (tối đa 10 bước)
    save_indices = np.linspace(0, ddim_steps - 1, min(10, ddim_steps), dtype=int)
    save_indices_set = set(save_indices)
    
    print(f"[*] Starting DDIM sampling with {ddim_steps} steps and eta={eta}...")
    with torch.no_grad():
        z = torch.randn(num_samples, src.config.INPUT_CHANNELS, 32, 32, device=src.config.DEVICE)
        images = []
        
        # Khử nhiễu tuần tự sử dụng công thức DDIM
        for i in reversed(range(ddim_steps)):
            t = ddim_timesteps[i]
            s = ddim_timesteps[i - 1] if i > 0 else -1
            
            t_tensor = torch.full((num_samples,), t, dtype=torch.long, device=src.config.DEVICE)
            
            # Dự đoán nhiễu bằng mô hình
            pred_noise = eval_model(z, t_tensor)
            
            # Lấy alpha_bar tích lũy tại bước t và s
            alpha_bar_t = scheduler.alpha_bar[t_tensor].view(-1, 1, 1, 1)
            if s >= 0:
                s_tensor = torch.full((num_samples,), s, dtype=torch.long, device=src.config.DEVICE)
                alpha_bar_s = scheduler.alpha_bar[s_tensor].view(-1, 1, 1, 1)
            else:
                alpha_bar_s = torch.ones_like(alpha_bar_t)
            
            # Dự đoán ảnh gốc x_0 ban đầu
            pred_x0 = (z - torch.sqrt(1.0 - alpha_bar_t) * pred_noise) / torch.sqrt(alpha_bar_t)
            
            # Tính độ lệch chuẩn sigma_t điều khiển mức độ ngẫu nhiên
            if eta > 0:
                sigma_t = eta * torch.sqrt((1.0 - alpha_bar_s) / (1.0 - alpha_bar_t)) * torch.sqrt(1.0 - alpha_bar_t / alpha_bar_s)
            else:
                sigma_t = torch.zeros_like(alpha_bar_t)
            
            # Hướng chỉ vào x_t
            pred_dir_coeff = torch.clamp(1.0 - alpha_bar_s - sigma_t**2, min=0.0).sqrt()
            pred_dir = pred_dir_coeff * pred_noise
            
            # Sinh nhiễu ngẫu nhiên nếu eta > 0 và chưa phải bước cuối
            if s >= 0 and eta > 0:
                noise = torch.randn_like(z)
            else:
                noise = torch.zeros_like(z)
            
            # Cập nhật z cho bước tiếp theo
            z = torch.sqrt(alpha_bar_s) * pred_x0 + pred_dir + sigma_t * noise
            
            # Lưu lại trạng thái ảnh tại các bước cấu hình để vẽ hình minh họa
            if i in save_indices_set:
                images.append(z.clone())
        
        print("[*] Inference complete. Saving sequences...")
        # Chỉ lưu sequence cho tối đa 5 ảnh để tránh tràn bộ nhớ và tạo quá nhiều file
        for s_idx in range(min(5, num_samples)):
            sample_images = [img[s_idx:s_idx+1] for img in images]
            save_img_path = os.path.join(src.config.REPORT_DIR, f'generated_sequence_{s_idx+1}.png')
            display_reverse(sample_images, save_path=save_img_path)
            
        # Lưu một lưới ảnh 5x5 từ các ảnh kết quả cuối cùng (nếu số mẫu >= 25)
        if num_samples >= 25:
            grid_img_path = os.path.join(src.config.REPORT_DIR, 'grid_5x5.png')
            final_images = z[:25]
            # Chuẩn hóa ngược từ [-1, 1] về [0, 1] để lưu
            grid = torchvision.utils.make_grid(final_images, nrow=5)
            grid = (grid + 1.0) / 2.0
            grid = torch.clamp(grid, 0.0, 1.0)
            torchvision.utils.save_image(grid, grid_img_path)
            print(f"[*] Saved 5x5 grid of final images to {grid_img_path}")

if __name__ == '__main__':
    # Chạy inference với file checkpoint mong muốn (thường là best_ddpm.pt hoặc latest_ddpm.pt)
    ckpt_path = os.path.join(src.config.CHECKPOINT_DIR, 'best_ddpm.pt')
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(src.config.CHECKPOINT_DIR, 'latest_ddpm.pt')
        
    if os.path.exists(ckpt_path):
        inference(ckpt_path)
    else:
        print(f"[!] Checkpoint {ckpt_path} not found. Please train the model first.")