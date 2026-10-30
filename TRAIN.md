# Hướng Dẫn Huấn Luyện CO-SPY Trên Kaggle

Tài liệu này hướng dẫn chi tiết cách chuẩn bị dữ liệu và các cell cần chạy khi huấn luyện mô hình **CO-SPY** (phiên bản mới nhất với Adaptive Fusion v2/v3) trên môi trường Kaggle Notebook (GPU khuyến nghị: P100, T4 x 2 hoặc A100).

Từ phiên bản hiện tại, training tự lưu full-state checkpoint, tự resume khi
chạy lại, và dừng an toàn trước giới hạn 12 giờ. Không cần sửa epoch trong
notebook. Khi tạo Version mới, chỉ cần Add Output của Version trước làm Input.
Xem [TRAINING_RECOVERY_GUIDE.md](TRAINING_RECOVERY_GUIDE.md) để biết đầy đủ quy
trình Kaggle, Colab và chuyển checkpoint.

---

## 1. Nguồn Dữ Liệu & Link Tải (Datasets)

Theo cấu trúc tài liệu `README.md` và script `data/train/sd-v1_4/download.sh`, pipeline huấn luyện tiêu chuẩn (`sd-v1_4`) cần 2 bộ dữ liệu:

### A. Tập huấn luyện chính (`sd-v1_4` - Khuyến nghị)
1. **Ảnh AI (Synthetic - Stable Diffusion v1.4)**:
   - **Nguồn**: DRCT-2M (ICML 2024)
   - **Link tải trực tiếp**: `https://modelscope.cn/datasets/BokingChen/DRCT-2M/resolve/master/images/stable-diffusion-v1-4.zip`
   - **Đích giải nén**: `data/train/sd-v1_4/stable-diffusion-v1-4`
2. **Ảnh Thật (Real - MSCOCO 2017)**:
   - **Nguồn**: Co-Spy-Misc
   - **Link tải trực tiếp**: `https://huggingface.co/datasets/ruojiruoli/Co-Spy-Misc/resolve/main/mscoco2017.zip`
   - **Đích giải nén**: `data/train/sd-v1_4/mscoco2017`

### B. Tập huấn luyện phụ (`progan` / CNNDet - Tùy chọn)
Nếu muốn huấn luyện trên dữ liệu GAN:
- **Ảnh Train**: 7 phần nén từ `https://huggingface.co/datasets/sywang/CNNDetection/resolve/main/progan_train.7z.001` đến `.007`
- **Ảnh Val**: `https://huggingface.co/datasets/sywang/CNNDetection/resolve/main/progan_val.zip`

---

## 2. Cấu Trúc Thư Mục Dữ Liệu Sau Khi Giải Nén

```text
CO-SPY/
└── data/
    └── train/
        └── sd-v1_4/
            ├── mscoco2017/
            └── stable-diffusion-v1-4/
```

---

## 3. Các Cell Cần Chạy Trên Kaggle Notebook

> **Mẹo Tối Ưu T4 x 2:** Mặc định CO-SPY chạy đơn GPU (`cuda:0`). Nếu bạn bật môi trường **GPU T4 x 2** trên Kaggle, bạn có thể chạy **song song nhánh Artifact trên GPU 0 và nhánh Semantic trên GPU 1** ở Cell 4 (rút ngắn gần một nửa thời gian train). Hướng dẫn bên dưới cung cấp cả 2 lựa chọn (1 GPU hoặc 2 GPU song song).

### Cell 1: Clone Repo & Chuyển thư mục làm việc
```bash
!git clone https://github.com/datnguyentrung/CO-SPY.git cospy
%cd cospy
```

---

### Cell 2: Cài đặt Môi trường & Thư viện Phụ thuộc
```bash
!pip install -q diffusers open-clip-torch timm loguru accelerate ftfy
!apt-get update -qq && apt-get install -qq p7zip-full wget unzip
```

---

### Cell 3: Tải & Giải nén Dữ liệu Train (`sd-v1_4`)
```bash
%%bash
mkdir -p data/train/sd-v1_4
cd data/train/sd-v1_4

# 1. Tải ảnh AI (SD v1.4)
echo "Đang tải ảnh SD v1.4..."
wget -q --no-check-certificate https://modelscope.cn/datasets/BokingChen/DRCT-2M/resolve/master/images/stable-diffusion-v1-4.zip
unzip -q stable-diffusion-v1-4.zip && rm stable-diffusion-v1-4.zip

# 2. Tải ảnh Real (MSCOCO 2017)
echo "Đang tải ảnh Real MSCOCO 2017..."
wget -q https://huggingface.co/datasets/ruojiruoli/Co-Spy-Misc/resolve/main/mscoco2017.zip
unzip -q mscoco2017.zip && rm mscoco2017.zip

echo "Hoàn thành chuẩn bị dữ liệu train!"
```

---

### Cell 4 (LỰA CHỌN A - KHUYÊN DÙNG CHO GPU T4 x 2): Huấn luyện Song Song 2 Nhánh Cùng Lúc
*GPU 0 huấn luyện nhánh Artifact (20 epochs) trong khi GPU 1 đồng thời huấn luyện nhánh Semantic (10 epochs).*
```bash
%%bash
echo "=== Khởi chạy huấn luyện song song trên 2 GPU ==="

# 1. Chạy ngầm nhánh Artifact trên GPU 0
python main.py \
    --phase train \
    --gpu 0 \
    --mode branch \
    --branch artifact \
    --train_dataset sd-v1_4 \
    --epochs 20 > train_artifact.log 2>&1 &
PID_ART=$!
echo "Đã khởi chạy Artifact trên GPU 0 (PID: $PID_ART)"

# 2. Chạy ngầm nhánh Semantic trên GPU 1
python main.py \
    --phase train \
    --gpu 1 \
    --mode branch \
    --branch semantic \
    --train_dataset sd-v1_4 \
    --epochs 10 \
    --feat_interp > train_semantic.log 2>&1 &
PID_SEM=$!
echo "Đã khởi chạy Semantic trên GPU 1 (PID: $PID_SEM)"

# 3. Theo dõi tiến trình cho đến khi hoàn tất
wait $PID_ART $PID_SEM
echo "=== Cả 2 GPU đã hoàn thành huấn luyện! ==="
```
*(Bạn có thể mở một cell phụ với lệnh `!tail -n 20 train_artifact.log` hoặc `!tail -n 20 train_semantic.log` để theo dõi tiến độ log).*

---

### Cell 4 (LỰA CHỌN B - DÙNG KHI CHỈ CÓ 1 GPU P100 / T4 ĐƠN):
Nếu bạn chỉ chọn 1 GPU, chạy tuần tự 2 cell sau:

**Cell 4.1: Huấn luyện Artifact Branch (20 Epochs)**
```bash
!python main.py \
    --phase train \
    --gpu 0 \
    --mode branch \
    --branch artifact \
    --train_dataset sd-v1_4 \
    --epochs 20
```

**Cell 4.2: Huấn luyện Semantic Branch (10 Epochs)**
```bash
!python main.py \
    --phase train \
    --gpu 0 \
    --mode branch \
    --branch semantic \
    --train_dataset sd-v1_4 \
    --epochs 10 \
    --feat_interp
```

---

### Cell 5 (Tùy chọn): Huấn luyện Nhánh 3 - NPR Branch (20 Epochs)
*Chỉ chạy nếu bạn có ý định dùng cơ chế kết hợp Fusion Head `v3` hoặc `--npr`.*
```bash
!python main.py \
    --phase train \
    --gpu 0 \
    --mode branch \
    --branch npr \
    --train_dataset sd-v1_4 \
    --epochs 20
```

---

### Cell 6: Huấn luyện Bộ Kết Hợp Thích Ứng (Adaptive Fusion Head - 30 Epochs)
Đóng băng đặc trưng của các nhánh đã huấn luyện và tối ưu Fusion Head v2. Bước này nhẹ hơn rất nhiều nhờ cơ chế cache feature tự động.
```bash
!python main.py \
    --phase train \
    --gpu 0 \
    --mode fusion \
    --train_dataset sd-v1_4 \
    --epochs 30 \
    --fusion_head v2 \
    --artifact_views 1
```

---

### Cell 7: Đóng gói và Lưu Trữ Checkpoint
```bash
!zip -r cospy_checkpoints.zip ckpt/
```

`checkpoint_latest.pth`, `checkpoint_recovery.pth`, `checkpoint_best.pth` và
`training_state.json` nằm trong thư mục của từng mode. Trainer không còn tạo
file checkpoint theo mọi epoch. Nếu Version mới Add Output cũ làm Input,
Trainer tự tìm checkpoint dưới `/kaggle/input` và tiếp tục từ batch tiếp theo.

