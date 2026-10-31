# Hướng Dẫn Huấn Luyện CO-SPY Trên Kaggle

Tài liệu này khớp với pipeline hiện tại:

- Full-state checkpoint và auto-resume.
- Early stopping mặc định sau 5 epoch không cải thiện.
- Branch và adaptive fusion theo dõi validation AP.
- Semantic branch cache CLIP feature một lần.
- Fusion v2/v3 cache đặc trưng của các branch.
- `print(..., flush=True)` trong `train.py` hiển thị ngay trên Kaggle terminal.

Luôn chạy lệnh từ thư mục gốc repository:

```bash
cd /kaggle/working/cospy
```

---

## 1. Dataset Có Sẵn Trên Kaggle

Dataset `Co-Spy_v1` chỉ cần được Add vào notebook qua **Add Input**. Không tải
lại và không giải nén.

Cấu trúc input Kaggle:

```text
/kaggle/input/datasets/datnguyentrung/co-spy-v1/
├── mscoco2017/
│   └── mscoco2017/
│       ├── train2017/
│       └── val2017/
└── stable-diffusion-v1-4/
    └── stable-diffusion-v1-4/
        ├── train2017/
        └── val2017/
```

`dataSets/dataset.py` hiện vẫn đọc bằng đường dẫn tương đối. Cell 3 chỉ tạo
symlink từ cấu trúc trên vào:

```text
data/train/sd-v1_4/
├── mscoco2017/
│   ├── train2017/
│   └── val2017/
└── stable-diffusion-v1-4/
    ├── train2017/
    └── val2017/
```

Symlink trỏ trực tiếp về `/kaggle/input`, không copy ảnh và không thêm dữ liệu
vào `/kaggle/working`.

---

## 2. Các Cell Trên Kaggle

### Cell 1: Clone Repository

```bash
%%bash
set -euo pipefail

cd /kaggle/working
if [[ ! -d cospy/.git ]]; then
    git clone https://github.com/datnguyentrung/CO-SPY.git cospy
fi
```

### Cell 2: Cài Thư Viện

```python
%cd /kaggle/working/cospy

%pip install -q \
    "diffusers==0.32.2" \
    "open-clip-torch==2.31.0" \
    "timm==1.0.15" \
    "loguru==0.7.3" \
    "accelerate==1.0.1" \
    "ftfy==6.2.3" \
    "tqdm==4.67.1" \
    "safetensors==0.5.3"

```

Không cài lại `torch`/`torchvision` nếu phiên Kaggle hiện tại đã chạy ổn; việc
thay Torch có thể làm hỏng CUDA runtime của notebook.

### Cell 3: Liên Kết Dataset Có Sẵn Vào Working Directory

```bash
%%bash
set -euo pipefail

cd /kaggle/working/cospy

DATASET_ROOT="${CO_SPY_DATASET_ROOT:-/kaggle/input/datasets/datnguyentrung/co-spy-v1}"

if [[ ! -d "$DATASET_ROOT" ]]; then
    DATASET_ROOT="$(
        find /kaggle/input -type d -name "co-spy-v1" -print -quit
    )"
fi

if [[ -z "${DATASET_ROOT:-}" || ! -d "$DATASET_ROOT" ]]; then
    echo "[DATA] ERROR: Cannot find Co-Spy_v1 under /kaggle/input." >&2
    echo "[DATA] Add it with Add Input or override CO_SPY_DATASET_ROOT." >&2
    find /kaggle/input -maxdepth 4 -type d | sort | head -n 80 >&2
    exit 1
fi

echo "[DATA] Using existing Kaggle input: $DATASET_ROOT"

SD_ROOT="data/train/sd-v1_4"
FAKE_DIR="$SD_ROOT/stable-diffusion-v1-4"
REAL_DIR="$SD_ROOT/mscoco2017"
mkdir -p "$FAKE_DIR" "$REAL_DIR"

link_split() {
    local source_parent="$1"
    local target_parent="$2"
    local split="$3"
    local expected="${split}2017"
    local source_dir
    local target="$target_parent/$expected"

    source_dir="$(
        find "$source_parent" -type d \
            \( -name "$expected" -o -name "$split" \) \
            -print -quit
    )"

    if [[ -z "$source_dir" ]]; then
        echo "[DATA] ERROR: Cannot find $expected under $source_parent" >&2
        find "$source_parent" -maxdepth 4 -type d | sort | head -n 40 >&2
        exit 1
    fi

    if [[ -L "$target" ]]; then
        rm -f "$target"
    elif [[ -d "$target" ]]; then
        echo "[DATA] Keeping existing directory: $target"
        return
    fi

    ln -s "$source_dir" "$target"
    echo "[DATA] Linked $target -> $source_dir"
}

link_split "$DATASET_ROOT/mscoco2017" "$REAL_DIR" train
link_split "$DATASET_ROOT/mscoco2017" "$REAL_DIR" val
link_split "$DATASET_ROOT/stable-diffusion-v1-4" "$FAKE_DIR" train
link_split "$DATASET_ROOT/stable-diffusion-v1-4" "$FAKE_DIR" val

count_images() {
    find -L "$1" -type f \
        \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' \) \
        | wc -l
}

for path in \
    "$FAKE_DIR/train2017" \
    "$FAKE_DIR/val2017" \
    "$REAL_DIR/train2017" \
    "$REAL_DIR/val2017"
do
    count="$(count_images "$path")"
    printf '[DATA] %-70s %8s images\n' "$path" "$count"
    if [[ "$count" -eq 0 ]]; then
        echo "[DATA] ERROR: no images found in $path" >&2
        exit 1
    fi
done

echo "[DATA] Existing Kaggle dataset is ready; no download or extraction needed."
```

---

## 3. Train Và Log Trực Tiếp Trên Kaggle

`train.py` dùng `print(..., flush=True)` cho:

- Batch progress: `[TRAIN] Epoch ... | Batch ... | AP ... | Accuracy ...`
- Validation: `[VAL] Epoch ... | Test AP ... | Test Accuracy ...`
- Checkpoint tốt nhất: `[CKPT] Best checkpoint saved ...`
- Kết thúc hoặc early stopping: `[TRAIN] ...`

Hàm `run_job` dưới đây chạy `python -u`, không buffer stdout/stderr, hiển thị log
ngay trên Kaggle và đồng thời lưu vào file bằng `tee -a`.

### Cell 4A: Train Song Song Trên T4 x 2

GPU 0 chạy Artifact, GPU 1 chạy Semantic. Cả hai dùng tối đa 10 epoch và dừng
sớm sau 5 epoch không tăng AP.

```bash
%%bash
set -o pipefail
cd /kaggle/working/cospy

run_job() {
    local tag="$1"
    local log_file="$2"
    shift 2

    PYTHONUNBUFFERED=1 python -u "$@" 2>&1 \
        | tee -a "$log_file" \
        | sed -u "s/^/[$tag] /"
}

echo "[BOOT] Starting Artifact on GPU 0 and Semantic on GPU 1..."

run_job ART train_artifact.log main.py \
    --phase train \
    --gpu 0 \
    --mode branch \
    --branch artifact \
    --train_dataset sd-v1_4 \
    --epochs 10 \
    --early-stopping-patience 5 &
PID_ART=$!

run_job SEM train_semantic.log main.py \
    --phase train \
    --gpu 1 \
    --mode branch \
    --branch semantic \
    --train_dataset sd-v1_4 \
    --epochs 10 \
    --early-stopping-patience 5 \
    --feat_interp &
PID_SEM=$!

echo "[BOOT] Artifact PID=$PID_ART | Semantic PID=$PID_SEM"

STATUS=0
wait "$PID_ART" || STATUS=1
wait "$PID_SEM" || STATUS=1

if [[ "$STATUS" -ne 0 ]]; then
    echo "[ERROR] At least one branch failed. Check the terminal output above." >&2
    exit "$STATUS"
fi

echo "[DONE] Both branches completed."
```

### Cell 4B: Train Tuần Tự Trên Một GPU

Chạy riêng Artifact:

```bash
%%bash
set -o pipefail
cd /kaggle/working/cospy

PYTHONUNBUFFERED=1 python -u main.py \
    --phase train \
    --gpu 0 \
    --mode branch \
    --branch artifact \
    --train_dataset sd-v1_4 \
    --epochs 10 \
    --early-stopping-patience 5 \
    2>&1 | tee -a train_artifact.log
```

Artifact không cache đặc trưng cuối vì ResNet trên VAE residual vẫn trainable.
VAE và ResNet vẫn chạy mỗi epoch.

Chạy riêng Semantic:

```bash
%%bash
set -o pipefail
cd /kaggle/working/cospy

PYTHONUNBUFFERED=1 python -u main.py \
    --phase train \
    --gpu 0 \
    --mode branch \
    --branch semantic \
    --train_dataset sd-v1_4 \
    --epochs 10 \
    --early-stopping-patience 5 \
    --feat_interp \
    2>&1 | tee -a train_semantic.log
```

Lần đầu Semantic sẽ tạo cache CLIP dưới:

```text
ckpt/sd-v1_4/semantic/features/semantic/
```

Cache khoảng `191807 × 1152 × float16 ≈ 0.4 GB`. Các epoch sau và lần resume
tiếp theo chỉ cần chạy lớp `fc`. Dùng `--no-cache-features` nếu muốn áp dụng
augmentation mới cho từng epoch.

### Cell 5: NPR Branch Cho Fusion v3

Chỉ chạy nếu dùng `--fusion_head v3`.

```bash
%%bash
set -o pipefail
cd /kaggle/working/cospy

PYTHONUNBUFFERED=1 python -u main.py \
    --phase train \
    --gpu 0 \
    --mode branch \
    --branch npr \
    --train_dataset sd-v1_4 \
    --epochs 10 \
    --early-stopping-patience 5 \
    2>&1 | tee -a train_npr.log
```

### Cell 6: Adaptive Fusion

Fusion v2 cần `checkpoint_best.pth` của Artifact và Semantic:

```bash
%%bash
set -o pipefail
cd /kaggle/working/cospy

PYTHONUNBUFFERED=1 python -u main.py \
    --phase train \
    --gpu 0 \
    --mode fusion \
    --train_dataset sd-v1_4 \
    --epochs 20 \
    --early-stopping-patience 5 \
    --fusion_head v2 \
    --artifact_views 1 \
    2>&1 | tee -a train_fusion_v2.log
```

Fusion v3 cần thêm NPR branch:

```bash
%%bash
set -o pipefail
cd /kaggle/working/cospy

PYTHONUNBUFFERED=1 python -u main.py \
    --phase train \
    --gpu 0 \
    --mode fusion \
    --train_dataset sd-v1_4 \
    --epochs 20 \
    --early-stopping-patience 5 \
    --fusion_head v3 \
    --artifact_views 1 \
    2>&1 | tee -a train_fusion_v3.log
```

Fusion v2/v3 tự cache đặc trưng branch. Cache được tái sử dụng khi resume.

---

## 4. Early Stopping

Mặc định:

```text
--early-stopping-patience 5
```

Theo dõi:

| Mode | Metric |
| --- | --- |
| Branch Artifact | Validation AP |
| Branch Semantic | Validation AP |
| Branch NPR | Validation AP |
| Fusion v2/v3 | Validation AP |
| Fusion v1 hoặc End-to-End | Validation Accuracy |

Tắt early stopping:

```bash
--early-stopping-patience 0
```

`checkpoint_best.pth` chỉ được cập nhật khi metric tăng thực sự.

---

## 5. Resume Trên Kaggle Version Mới

Trainer tự tìm `checkpoint_latest.pth` và `checkpoint_recovery.pth` dưới
`/kaggle/working` và `/kaggle/input`.

Quy trình:

1. Để Trainer tự dừng trước giới hạn phiên.
2. Bấm **Save Version** để lưu output.
3. Tạo Version mới.
4. Add output của Version trước trong **Add Input**.
5. Chạy lại đúng cell train.

Ví dụ Artifact resume với log trực tiếp:

```bash
%%bash
set -o pipefail
cd /kaggle/working/cospy

PYTHONUNBUFFERED=1 python -u main.py \
    --phase train \
    --gpu 0 \
    --mode branch \
    --branch artifact \
    --train_dataset sd-v1_4 \
    --epochs 10 \
    --early-stopping-patience 5 \
    --resume-mode auto \
    2>&1 | tee -a train_artifact_resume.log
```

Có thể thay đổi các cấu hình điều khiển sau khi resume:

```text
epochs
early_stopping_patience
cache_features
num_workers
```

Không đổi khi resume full:

```text
dataset
model mode/branch/fusion head
architecture
batch size
seed
optimizer
```

`--resume-mode full` bắt buộc phải tìm được checkpoint. `--no-resume` bỏ qua
checkpoint và bắt đầu training mới.

Chi tiết đầy đủ nằm trong [TRAINING_RECOVERY_GUIDE.md](TRAINING_RECOVERY_GUIDE.md).

---

## 6. Save, Kiểm Tra Và Đóng Gói

Mỗi mode lưu:

```text
checkpoint_latest.pth
checkpoint_recovery.pth
checkpoint_best.pth
training_state.json
training.log
```

`training.log` vẫn nhận log từ `logger`; terminal nhận `print()` và output được
`tee` sang `train_*.log`.

Kiểm tra checkpoint nhanh:

```bash
!find ckpt -maxdepth 4 -type f \
    \( -name 'checkpoint_*.pth' -o -name 'training_state.json' \) \
    -printf '%p %k KB\n' | sort
```

Đóng gói checkpoint nhưng bỏ feature cache:

```bash
!zip -r cospy_checkpoints.zip ckpt/ -x '*/features/*'
```

Feature cache có thể tái tạo nếu có branch weights; không cần đưa vào archive
cuối cùng.

---

## 7. Lỗi Thường Gặp

### `Train size 0` hoặc `Val size 0`

Chạy lại Cell 3 và kiểm tra:

```bash
!find -L data/train/sd-v1_4 -maxdepth 4 -type d | sort | head -n 40
```

### Không thấy `[TRAIN]` trên Kaggle

Không chạy:

```bash
python main.py ... > log.txt 2>&1
```

Cách đó chỉ ghi file. Dùng:

```bash
PYTHONUNBUFFERED=1 python -u main.py ... 2>&1 | tee -a log.txt
```

### Fusion không tìm thấy branch weights

Chạy đủ Artifact và Semantic trước Fusion v2. Chạy thêm NPR trước Fusion v3.
Kiểm tra:

```bash
!find ckpt/sd-v1_4 -name checkpoint_best.pth -printf '%p\n' | sort
```
