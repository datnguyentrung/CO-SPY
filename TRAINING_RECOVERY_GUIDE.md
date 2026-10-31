# CO-SPY Training Recovery Guide

Tài liệu này mô tả hệ thống auto checkpoint, auto resume và phục hồi training
đã được tích hợp vào entrypoint hiện tại. Không cần sửa `start_epoch`,
`current_epoch` hoặc chạy lại từ đầu sau khi Kaggle/Colab hết thời gian.

## 1. Hành vi mặc định

Mỗi lần chạy:

```bash
python main.py --phase train ...
```

Trainer sẽ:

1. Phát hiện Kaggle, Colab hoặc môi trường local.
2. Tìm checkpoint trong thư mục output và các input đã cấu hình.
3. Kiểm tra checkpoint có đúng experiment, kiến trúc, dataset và cấu hình train.
4. Khôi phục model, optimizer, scheduler, RNG, global step, best metric và vị trí
   DataLoader.
5. Tiếp tục đúng batch tiếp theo. Không chạy lại các batch đã cập nhật optimizer.

Có checkpoint tương thích:

```text
[AUTO RESUME] Checkpoint found.
[AUTO RESUME] Last completed epoch: 35
[AUTO RESUME] Training will continue at epoch 36, batch 1.
```

Không có checkpoint:

```text
[AUTO RESUME] No checkpoint found.
[TRAINING] Starting from epoch 1.
```

## 2. Checkpoint được lưu ở đâu

Mặc định, checkpoint nằm trong thư mục train hiện có:

```text
ckpt/
└── sd-v1_4/
    ├── artifact/
    │   ├── checkpoint_latest.pth
    │   ├── checkpoint_recovery.pth
    │   ├── checkpoint_best.pth
    │   ├── training_state.json
    │   └── training.log
    ├── semantic/
    └── fusion_v2/
```

Ba checkpoint có vai trò khác nhau:

| File | Nội dung | Mục đích |
| --- | --- | --- |
| `checkpoint_latest.pth` | Full training state sau epoch hoàn tất | Resume |
| `checkpoint_recovery.pth` | Full training state tại batch an toàn gần nhất | Resume/fallback |
| `checkpoint_best.pth` | Weights của model có metric tốt nhất | Evaluation/inference |

File `.json` nằm cạnh `checkpoint_latest.pth` và
`checkpoint_recovery.pth` chỉ là metadata phụ để tìm checkpoint nhanh.

`best_model.pth` vẫn được đọc như checkpoint cũ để bảo đảm tương thích, nhưng
lần train mới chỉ ghi `checkpoint_best.pth`.

## 3. Schedule checkpoint

Các mặc định:

- Recovery checkpoint: mỗi 15 phút và sau mỗi batch khi gần timeout hoặc có signal.
- Latest checkpoint: sau mỗi epoch hoàn tất.
- Best checkpoint: chỉ khi validation metric tăng thực sự.
- Session timeout: 12 giờ.
- Dừng an toàn trước timeout: 15 phút.

Có thể đổi bằng CLI:

```bash
python main.py \
  --phase train \
  --mode branch \
  --branch artifact \
  --train_dataset sd-v1_4 \
  --epochs 20 \
  --checkpoint-interval-minutes 15 \
  --session-timeout-minutes 720 \
  --pre-timeout-minutes 15
```

Đặt `--checkpoint-interval-minutes 0` để tắt checkpoint theo thời gian.
Checkpoint theo epoch, timeout và graceful shutdown vẫn hoạt động.

## 4. Resume giữa epoch

Checkpoint full chứa:

- Model state.
- AdamW state, gồm moment buffers và learning rate.
- Cosine scheduler state hoặc trạng thái LR step-decay.
- `global_step`.
- Best metric và lịch sử early stopping.
- Python, NumPy, PyTorch CPU/CUDA RNG.
- Epoch, batch position và thứ tự sampler.
- Trạng thái seed ban đầu của DataLoader workers.
- Các state tùy chọn cho GradScaler/EMA nếu training engine dùng.

Khi resume giữa epoch, Trainer tạo lại cùng sampler order và worker seeds,
replay qua số batch đã hoàn tất nhưng bỏ qua optimizer update. Batch tiếp theo
được train giống như phiên cũ chưa từng bị ngắt.

Checkpoint chỉ được ghi sau một batch hoàn tất. Nếu process bị kill đột ngột
giữa batch, batch dở được chạy lại; checkpoint hợp lệ trước đó không bị sửa.

## 5. Atomic save và checkpoint hỏng

Một checkpoint được ghi vào file tạm cùng thư mục, flush/fsync, rồi thay thế
file chính bằng `os.replace`. Vì vậy process bị ngắt khi đang ghi không làm
hỏng checkpoint hợp lệ trước đó.

Khi restore:

1. Ưu tiên latest/recovery có `global_step` cao nhất.
2. Nếu file hỏng, ghi lý do rồi thử checkpoint kế tiếp.
3. Không dùng checkpoint có experiment/kiến trúc/dataset/cấu hình khác.
4. Nếu không còn checkpoint hợp lệ, training bắt đầu mới và in cảnh báo rõ ràng.

Với `--resume-path` được chỉ định trực tiếp, checkpoint hỏng hoặc không tương
thích sẽ dừng với lỗi có kiểm soát thay vì âm thầm bắt đầu experiment mới.

## 6. Chọn chế độ resume

Resume full, mặc định:

```bash
python main.py --phase train ... --resume-mode auto
```

Dùng `--resume-mode full` khi bắt buộc phải resume; nếu không tìm thấy
checkpoint, lệnh dừng với lỗi rõ ràng thay vì bắt đầu training mới.

Chỉ load model weights và tạo optimizer/scheduler mới:

```bash
python main.py --phase train ... --resume-mode weights
```

Trong chế độ `weights`, epoch và global step bắt đầu từ đầu. Đây không phải
resume full và không giữ optimizer momentum.

Chủ động tạo training mới và bỏ qua checkpoint:

```bash
python main.py --phase train ... --resume-mode new
# tương đương
python main.py --phase train ... --no-resume
```

Resume đúng file:

```bash
python main.py \
  --phase train \
  --mode branch \
  --branch semantic \
  --train_dataset sd-v1_4 \
  --resume-path /kaggle/input/cospy-v5/ckpt/sd-v1_4/semantic/checkpoint_latest.pth
```

## 7. Kaggle Notebook Version mới

### Lần chạy đầu

Chạy lệnh train bình thường. Checkpoint được ghi dưới
`/kaggle/working/ckpt/...`.

### Khi gần hết 12 giờ

Notebook nên dừng an toàn trước giới hạn:

```bash
python main.py \
  --phase train \
  --mode branch \
  --branch artifact \
  --train_dataset sd-v1_4 \
  --epochs 20 \
  --session-timeout-minutes 705 \
  --pre-timeout-minutes 15
```

Khi tới ngưỡng, Trainer hoàn tất batch hiện tại, lưu recovery checkpoint và
thoát có kiểm soát.

### Tạo Version tiếp theo

1. Bấm **Save Version** để tạo output của lần chạy hiện tại.
2. Tạo Notebook Version mới.
3. Trong **Add Input**, chọn output của Version trước.
4. Giữ nguyên lệnh train và chạy lại cell.

Trainer tự quét `/kaggle/input/` và tìm `checkpoint_latest.pth` hoặc
`checkpoint_recovery.pth`, kể cả khi thư mục nằm sâu trong output cũ.

Không cần sửa epoch trong notebook. Checkpoint trong `/kaggle/working` chỉ là
output tạm của session hiện tại; cần Save Version để chuyển nó thành input của
version sau.

## 8. Google Colab

Ưu tiên ghi checkpoint trực tiếp vào Google Drive:

```python
from google.colab import drive
drive.mount("/content/drive")
```

```bash
python main.py \
  --phase train \
  --mode branch \
  --branch artifact \
  --train_dataset sd-v1_4 \
  --epochs 20 \
  --ckpt /content/drive/MyDrive/CO-SPY/ckpt
```

Có thể dùng thư mục checkpoint tuyệt đối riêng:

```bash
python main.py ... \
  --checkpoint-dir /content/drive/MyDrive/CO-SPY/recovery/artifact
```

Nếu Drive mount bị hạn chế atomic rename, có thể cấu hình thêm input roots:

```bash
export CO_SPY_CHECKPOINT_ROOTS="/content/drive/MyDrive/CO-SPY/ckpt"
```

Local temporary checkpoint có thể dùng để giảm I/O tới Drive; truyền thư mục
local ngắn hạn bằng `--checkpoint-dir`. Khi chuyển sang phiên mới, copy/tải thư
mục đó vào Drive rồi truyền lại bằng `--resume-path`.

## 9. Chuyển Kaggle sang Colab

1. Tải thư mục checkpoint của mode đang train từ Kaggle Output.
2. Upload nguyên thư mục vào Google Drive.
3. Dùng cùng dataset, model mode, batch size và seed. Có thể giảm `epochs`,
   patience hoặc bật/tắt feature cache khi resume; kiến trúc, dataset,
   optimizer và batch size vẫn phải khớp.
4. Trỏ `--ckpt` tới thư mục cha hoặc dùng `--resume-path` tới file full state.

Ví dụ:

```bash
python main.py \
  --phase train \
  --mode fusion \
  --train_dataset sd-v1_4 \
  --fusion_head v2 \
  --epochs 30 \
  --ckpt /content/drive/MyDrive/CO-SPY/ckpt
```

Có thể đổi `epochs`, `early_stopping_patience` và `cache_features` khi resume
full. Không đổi `batch_size`, `seed`, dataset path set, optimizer hoặc kiến
trúc nếu muốn tiếp tục đúng quỹ đạo checkpoint.

Nếu chỉ muốn chuyển weights:

```bash
python main.py ... --resume-mode weights
```

## 10. Training metadata

Mỗi thư mục checkpoint có `training_state.json`:

```text
NOT_STARTED -> RUNNING -> CHECKPOINTED
                         -> RESUMABLE
                         -> INTERRUPTED
                         -> COMPLETED
                         -> FAILED
```

File này chứa experiment, epoch, global step, best metric, checkpoint gần nhất,
environment, model config, dataset fingerprint và checkpoint format version.

`training.log` ghi cấu hình khởi chạy, runtime, model/optimizer, dataset
fingerprint, checkpoint search roots, resume, progress, lỗi kèm traceback và
trạng thái kết thúc. Đổi tần suất progress bằng:

```bash
python main.py --phase train ... --log-interval-batches 100
```

## 11. Output size

Trainer không còn tạo `epoch_*.pth` và `final_model.pth`. Mỗi mode giữ ba role
checkpoint chính. Feature cache vẫn dùng memmap dưới `features/` và có thể tái
tạo nếu có hai branch weights.

Mặc định chỉ có một `checkpoint_recovery.pth`. Có thể giữ thêm bản recovery cũ:

```bash
python main.py --phase train ... --recovery-retention 3
```

Lúc đó hệ thống quản lý `checkpoint_recovery.1.pth`,
`checkpoint_recovery.2.pth`, ... và chỉ xoay vòng/xóa các file recovery do chính
CheckpointManager tạo. Latest và best vẫn chỉ có một bản.

Cuối training, Trainer in:

- Dung lượng checkpoint.
- Log.
- Generated samples.
- Cache.
- Các file khác.
- Tổng dung lượng.
- Top 20 file lớn nhất.

Hệ thống không tự xóa `epoch_*.pth`, cache hoặc artifact cũ nếu chưa xác định
được vai trò. Sau khi xác nhận checkpoint mới resume được, có thể xóa thủ công
các checkpoint legacy trong đúng thư mục model.

Trên PowerShell:

```powershell
Get-ChildItem .\ckpt\sd-v1_4\artifact -Filter 'epoch_*.pth' |
  Remove-Item -WhatIf
```

Chỉ bỏ `-WhatIf` sau khi kiểm tra danh sách. Không xóa dataset gốc hoặc
`pretrained/`.

## 12. Kiểm thử

Chạy test không cần dataset CO-SPY đầy đủ:

```bash
python -m unittest discover -s tests -v
```

Các test hiện có kiểm tra:

- Full-state round trip.
- Resume giữa epoch cho optimizer khớp với train không bị ngắt.
- Fallback khi latest bị hỏng.
- Không resume checkpoint không tương thích.
- Atomic save và retention.
- Timeout request.
- Best checkpoint chỉ đổi khi metric tăng thực sự.
- Trainer thật (với model/dataset tiny) resume giữa epoch và cho trọng số cuối
  khớp hoàn toàn với một lần train không bị ngắt.
- Discovery root cấu hình và root Kaggle.

Kết quả gần nhất: 9 test pass trên Python 3.14 với PyTorch CPU.

Kiểm tra cú pháp:

```bash
python -m py_compile checkpoint_manager.py train.py main.py evaluate.py dataSets/dataset.py
```

## 13. Giới hạn

- Bitwise-exact reproducibility trên GPU còn phụ thuộc driver, cuDNN và phiên
  bản PyTorch. Trạng thái RNG được khôi phục, nhưng kernel không đảm bảo bitwise
  equality giữa hai loại GPU.
- Augmentation tùy biến mới cần dùng RNG deterministic trong worker. Các
  transform hiện tại dựa trên Python/NumPy/Torch RNG đã được worker seed.
- Nếu validation bị ngắt, validation sẽ chạy lại từ đầu. Optimizer không bị cập
  nhật thêm.
- `torch.save`/`torch.load` giả định checkpoint là file tin cậy do chính project
  tạo ra. Không load checkpoint không rõ nguồn gốc.
- Atomic replace cần thư mục đích cùng filesystem với file tạm. Hệ thống đã tạo
  file tạm ngay cạnh checkpoint để bảo đảm điều này.
- GradScaler và EMA được CheckpointManager hỗ trợ nếu training engine được mở
  rộng sử dụng chúng; pipeline CO-SPY hiện tại không bật AMP/EMA.
