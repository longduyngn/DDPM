import torch
import os

# --- Thiết lập Thiết bị (Giai đoạn 2) ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_GPUS = torch.cuda.device_count() if torch.cuda.is_available() else 0

# --- Siêu tham số Huấn luyện ---
BATCH_SIZE = 64
NUM_TIME_STEPS = 1000
NUM_EPOCHS = 10
LR = 1e-5
EMA_DECAY = 0.9999
SEED = 42
NUM_WORKERS = 0 # Đặt về 0 để tránh rò rỉ RAM khi chạy multiprocessing trên Kaggle/Linux và lỗi khi chạy win

# --- Cấu trúc U-Net ---
CHANNELS = [64, 128, 256, 512, 512, 384] #từ 1 lên 64 nhờ shallow conv,tăng dần attri và sau đó giảm dần về cuối cân nhắc concat nên có số chiều như vậy . Cuối cùng là late conv để chuyển về 1
ATTENTIONS = [False, True, False, False, False, True]
UPSCALES = [False, False, False, True, True, True]
NUM_GROUPS = 32
DROPOUT_PROB = 0.1
NUM_HEADS = 8
INPUT_CHANNELS = 1
OUTPUT_CHANNELS = 1
TIME_EMB_DIM = max(CHANNELS) # Chiều lớn nhất để làm vector thời gian chuẩn

# --- Đường dẫn ---
DATA_DIR = 'data'
CHECKPOINT_DIR = 'checkpoints'
REPORT_DIR = 'report'

# Đảm bảo thư mục lưu trữ tồn tại
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)