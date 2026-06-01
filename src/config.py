import torch
import os

# --- Thiết lập Thiết bị (Giai đoạn 2) ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_GPUS = torch.cuda.device_count() if torch.cuda.is_available() else 0

# --- Lựa chọn Bộ dữ liệu ---
DATASET = 'MNIST' # Lựa chọn: 'MNIST', 'CIFAR10'

# --- Siêu tham số Huấn luyện ---
BATCH_SIZE = 64
NUM_TIME_STEPS = 1000
NUM_EPOCHS = 2
LR = 2e-4
EMA_DECAY = 0.9999
SEED = 42
NUM_WORKERS = 4 # Đặt về 0 để tránh rò rỉ RAM khi chạy multiprocessing trên Kaggle/Linux và lỗi khi chạy win
BETA_SCHEDULE = 'cosine' # Lựa chọn: 'linear', 'cosine', 'nondecreasing', 'non-increasing'
BETA_START = 1e-4
BETA_END = 0.02

# --- Cấu trúc U-Net ---
CHANNELS = [32, 64, 128, 256, 512, 256, 128, 64] # Cấu trúc 8 tầng đối xứng hoàn toàn nhỏ gọn (~35.7M tham số)
ATTENTIONS = [False, True, True, False, False, False, True, True] # Chỉ chạy Attention ở độ phân giải 16x16 để tiết kiệm VRAM
UPSCALES = [False, False, False, False, True, True, True, True]
NUM_GROUPS = 32
DROPOUT_PROB = 0.1
NUM_HEADS = 8

if DATASET == 'MNIST':
    INPUT_CHANNELS = 1
    OUTPUT_CHANNELS = 1
elif DATASET == 'CIFAR10':
    INPUT_CHANNELS = 3
    OUTPUT_CHANNELS = 3
else:
    raise ValueError(f"Unknown dataset: {DATASET}")

TIME_EMB_DIM = max(CHANNELS) # Chiều lớn nhất để làm vector thời gian chuẩn

# --- Đường dẫn ---
DATA_DIR = 'data'
CHECKPOINT_DIR = 'checkpoints'
REPORT_DIR = 'report'
SAVE_EVERY = 25
CHECKPOINT = None 

# Đảm bảo thư mục lưu trữ tồn tại
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)