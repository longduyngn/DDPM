# Hướng Dẫn Tối Ưu Hóa Tốc Độ Huấn Luyện (Training Acceleration Guide)

Tài liệu này tổng hợp chi tiết các phương pháp tối ưu hóa hiệu năng đã được áp dụng vào dự án DDPM nhằm đẩy nhanh tốc độ huấn luyện trên GPU (cả môi trường local RTX 5060 và Multi-GPU trên Kaggle).

---

## 1. Tự Động Độ Chính Xác Hỗn Hợp (Automatic Mixed Precision - AMP)

### Nguyên lý hoạt động
Mặc định, PyTorch sử dụng kiểu dữ liệu số thực 32-bit (FP32) cho tất cả các trọng số và tính toán. Chế độ **Mixed Precision** sẽ tự động chuyển đổi các phép tính toán phù hợp (như nhân ma trận, tích chập) sang kiểu số thực 16-bit (FP16/BF16) trong khi giữ các tính toán nhạy cảm (như tính loss, batch normalization) ở kiểu FP32.

### Lý do giúp tăng tốc:
1. **Tận dụng Tensor Cores:** Các dòng card RTX (như RTX 5060) sở hữu phần cứng chuyên biệt gọi là Tensor Cores, giúp xử lý các phép toán FP16 nhanh gấp nhiều lần so với FP32.
2. **Tiết kiệm băng thông bộ nhớ:** Dữ liệu FP16 có kích thước chỉ bằng một nửa FP32, giúp giảm đáng kể thời gian truyền dữ liệu trong GPU.
3. **Giảm dung lượng VRAM:** Tiết kiệm khoảng 40% - 50% bộ nhớ VRAM, cho phép bạn tăng kích thước batch lớn hơn mà không bị lỗi Out-of-Memory (OOM).

### Cách triển khai trong mã nguồn:
Sử dụng bộ đôi `autocast` và `GradScaler` của PyTorch:
```python
from torch.amp import autocast, GradScaler

# Khởi tạo bộ cân bằng gradient (để tránh lỗi underflow của FP16)
scaler = GradScaler('cuda')

# Trong vòng lặp train
with autocast('cuda'):
    output = model(x_noisy, t)
    loss = criterion(output, e)

# Lan truyền ngược và tối ưu hóa qua Scaler
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

---

## 2. Tăng Kích Thước Batch (Batch Size Scaling)

### Nguyên lý hoạt động
Tăng tham số `BATCH_SIZE` trong `config.py` từ `64` lên `128` (hoặc `256` nếu chạy trên 2 GPU Kaggle).

### Lý do giúp tăng tốc:
* **Tăng hiệu suất song song (Hardware Occupancy):** GPU hoạt động tối đa hiệu năng khi nhận một lượng lớn dữ liệu tính toán cùng một lúc. Với các batch nhỏ, các vi xử lý (SMs) của GPU thường rơi vào trạng thái nhàn rỗi (idle) để chờ lệnh tiếp theo.
* **Giảm overhead của CPU:** Mỗi iteration (bước lặp) đòi hỏi CPU gửi lệnh điều khiển (kernel launch) đến GPU. Gom nhiều dữ liệu vào một batch giúp CPU gửi ít lệnh hơn trong một epoch.

> [!NOTE]
> Nhờ việc tích hợp **AMP** (ở mục 1), lượng VRAM chiếm dụng giảm đi rất nhiều, giúp chúng ta thoải mái tăng Batch Size lên 128 hoặc 256 mà không lo tràn bộ nhớ VRAM 8GB của máy.

---

## 3. Nạp Dữ Liệu Đa Tiến Trình (DataLoader `num_workers`)

### Nguyên lý hoạt động
Cấu hình `num_workers = 4` thay vì để mặc định là `0` trong `DataLoader`.

### Lý do giúp tăng tốc:
* **Khắc phục nghẽn cổ chai CPU (Data Loading Bottleneck):** Nếu `num_workers = 0`, CPU sẽ nạp dữ liệu một cách đồng bộ (synchronous): CPU nạp batch 1 -> GPU xử lý -> CPU lặp lại nạp batch 2 -> GPU xử lý. Lúc này GPU phải dừng lại chờ CPU chuẩn bị ảnh.
* **Nạp dữ liệu không đồng bộ:** Với `num_workers = 4`, PyTorch sẽ spawn (tạo) ra 4 tiến trình CPU chạy ngầm độc lập. Chúng sẽ đọc ảnh từ ổ đĩa, thực hiện các phép transform (ToTensor, Normalize) và xếp sẵn vào hàng đợi bộ nhớ. Khi GPU xử lý xong batch hiện tại, nó chỉ việc lấy ngay batch tiếp theo mà không phải chờ đợi.

---

## 4. Ghim Bộ Nhớ Hệ Thống (`pin_memory=True`)

### Nguyên lý hoạt động
Kích hoạt thuộc tính `pin_memory=True` trong `DataLoader` kết hợp với thuộc tính `non_blocking=True` khi chuyển dữ liệu lên thiết bị (`x.to(device, non_blocking=True)`).

### Lý do giúp tăng tốc:
* **RAM trang bộ nhớ ghim (Page-locked/Pinned Memory):** Mặc định, RAM hệ thống có thể bị hệ điều hành di chuyển dữ liệu sang phân vùng ảo (pagefile) trên ổ đĩa. Khi `pin_memory=True`, PyTorch yêu cầu hệ điều hành khóa cố định vùng nhớ chứa dữ liệu ảnh trên RAM vật lý.
* **Sao chép dữ liệu siêu tốc:** GPU có thể truy cập trực tiếp (Direct Memory Access - DMA) vào vùng nhớ đã ghim này để tải ảnh lên VRAM mà không cần thông qua sự can thiệp của CPU, giúp tốc độ chuyển mạch dữ liệu tăng lên đáng kể.
* **Không nghẽn luồng xử lý:** Kết hợp với `non_blocking=True` giúp luồng tính toán của GPU tiếp tục chạy song song trong lúc dữ liệu của batch kế tiếp đang được copy lên.

---

## 5. Song Song Hóa Đa GPU Cho Kaggle (`nn.DataParallel`)

### Nguyên lý hoạt động
Tự động bao bọc mô hình bằng `nn.DataParallel(model)` khi phát hiện hệ thống có nhiều hơn 1 GPU.

### Lý do giúp tăng tốc và độ ổn định:
* **Phân tách Batch tự động:** Giả sử bạn chạy trên Kaggle với 2 GPU và global `BATCH_SIZE = 256`. Lớp `DataParallel` sẽ tự động chia đôi dữ liệu, đẩy 128 mẫu sang GPU 0 và 128 mẫu sang GPU 1 để chạy lan truyền tiến (forward pass) song song cùng lúc, sau đó gom kết quả gradient lại để cập nhật trọng số.
* **Tối ưu mã nguồn với Batch động:**
  Trong vòng lặp huấn luyện, việc thay đổi từ tham số cấu hình tĩnh `config.BATCH_SIZE` sang lấy kích thước thực tế của tensor `x.size(0)` đảm bảo chương trình hoạt động bình thường trên môi trường Multi-GPU.
  *Tại sao:* Khi `DataParallel` chia nhỏ batch cho từng GPU, biến `x` đi vào luồng xử lý trên từng card sẽ chỉ có kích thước là `64` hoặc `128` chứ không phải `256`. Nếu dùng biến tĩnh `config.BATCH_SIZE` để khởi tạo vector thời gian `t`, PyTorch sẽ ném ra lỗi mismatch chiều ma trận lập tức.
