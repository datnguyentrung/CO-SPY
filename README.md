<div align="center">
  <img src="imgs/logo.png" width="220" alt="CO-SPY logo" />
  <h1>CO-SPY — Fusion v3 Web Demo</h1>
  <p><strong>Phát hiện ảnh do AI tạo bằng semantic, pixel-forensics và NPR signals.</strong></p>
</div>

[![Python](https://img.shields.io/badge/Python-3.8-blue)](environment.yml)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4.1-ee4c2c)](environment.yml)
[![React](https://img.shields.io/badge/React-19-149eca)](co-spy/package.json)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

## Mục lục

- [Dự án này là gì?](#dự-án-này-là-gì)
- [Khác biệt với CO-SPY gốc](#khác-biệt-với-co-spy-gốc)
- [Kiến trúc và pipeline](#kiến-trúc-và-pipeline)
- [Cài đặt và chạy web](#cài-đặt-và-chạy-web)
- [Checkpoint, API và kiểm thử](#checkpoint-api-và-kiểm-thử)
- [Nghiên cứu, giới hạn và trích dẫn](#nghiên-cứu-giới-hạn-và-trích-dẫn)

## Dự án này là gì?

Repository này bắt đầu từ mã nguồn chính thức của bài báo CVPR 2025 [CO-SPY: Combining Semantic and Pixel Features to Detect Synthetic Images by AI](https://openaccess.thecvf.com/content/CVPR2025/html/Cheng_CO-SPY_Combining_Semantic_and_Pixel_Features_to_Detect_Synthetic_Images_CVPR_2025_paper.html). Bản hiện tại mở rộng code nghiên cứu thành ứng dụng web cục bộ để so sánh trực tiếp:

- **CO-SPY Original** — Semantic + Artifact + fusion gốc.
- **CO-SPY Fusion v3** — Semantic + Artifact-VAE + NPR + adaptive fusion từ checkpoint_best.pth.

Người dùng tải JPG, PNG hoặc WEBP lên UI. Backend FastAPI chuẩn hoá ảnh và chạy cả hai kiến trúc; UI React hiển thị nhãn Real/AI, confidence, thời gian inference và ba tín hiệu branch của Fusion v3.

> Đây là công cụ hỗ trợ forensic, không phải bằng chứng xác quyết về nguồn gốc hay quyền sở hữu ảnh.

## Khác biệt với CO-SPY gốc

Các mục dưới đây là mở rộng của repository này, không phải tuyên bố benchmark/kết quả gốc của bài báo.

| Hạng mục | CO-SPY gốc | Bản hiện tại |
| --- | --- | --- |
| Cách dùng | CLI cho training, evaluation, single-image inference | Web app React + FastAPI, upload và phân tích trực tiếp |
| Mô hình | Chạy detector theo lệnh CLI | Chạy song song Original và Fusion v3 để đối chiếu |
| Fusion v3 | Pipeline training nghiên cứu | Nạp checkpoint tổng hợp theo đúng kiến trúc và xác thực state bắt buộc |
| Giải thích | Chủ yếu là xác suất cuối | Hiện Semantic, Artifact-VAE, NPR; đây là dự đoán độc lập, không phải fusion weights |
| Input inference | Transform eval của branch | EXIF transpose + RGB + hai forensic views: center, high_frequency |
| Ảnh AI khó | Có thể nghiêng Real khi ngữ nghĩa rất tự nhiên | NPR consensus guard chỉ đổi thành AI khi cả hai forensic views đều rất chắc chắn |
| Quan sát | Log/CLI | Health endpoint, timing, device, raw score và forensic metadata |
| UI | Không có dashboard web | Preview, hai panel so sánh, ba signals luôn hiện trên desktop |

### Tại sao thêm hai forensic views?

Ảnh ChatGPT/DALL·E có thể hợp lý về nội dung nên Semantic dễ nghiêng Real. Crop trung tâm cũng có thể chỉ chứa da, trời hoặc nền mượt. Fusion v3 vì vậy xem:

1. **center** — crop tâm 224×224, căn toạ độ theo block 8 px.
2. **high_frequency** — crop 224×224 có năng lượng biên cao nhất.

HighFrequencyCrop dùng proxy high-pass kiểu DCT: chuyển grayscale, tính sai phân ngang/dọc |dx| + |dy|, tổng hợp năng lượng theo vùng cục bộ rồi chọn vùng có điểm cao nhất. Nó không làm đẹp hay thay đổi nội dung; nó chọn vùng giàu chi tiết hơn cho Artifact-VAE/NPR.

Fusion chọn evidence AI mạnh nhất giữa hai view. Sau đó NPR guard chỉ đổi nhãn khi cả hai NPR predictions đạt 95%:

~~~text
if minimum(NPR(center).P(AI), NPR(high_frequency).P(AI)) >= 0.95:
    final = AI_GENERATED
else:
    final = Fusion v3 ban đầu
~~~

Dùng xác suất nhỏ hơn của hai view thay vì xác suất lớn nhất, để một crop nhiễu đơn lẻ không tự đảo nhãn. Xem [FUSION_V3_INPUT_PREPROCESSING.md](FUSION_V3_INPUT_PREPROCESSING.md).

## Kiến trúc và pipeline

~~~text
Browser (React + Vite :5173)
  │ POST /api/predict (multipart image)
  ▼
FastAPI (:8000)
  ├─ kiểm tra MIME/kích thước
  ├─ EXIF transpose → RGB
  ├─ CO-SPY Original
  └─ Fusion v3
       ├─ Semantic (1 lần)
       ├─ Artifact-VAE (center + high_frequency)
       ├─ NPR (center + high_frequency)
       ├─ chọn Fusion evidence AI cao nhất
       └─ NPR consensus guard
  ▼
JSON: original + Fusion v3 + branches + forensic metadata
~~~

| Thành phần | Vị trí | Vai trò |
| --- | --- | --- |
| Frontend | co-spy/src | Upload/preview, health polling, hiển thị kết quả |
| API | co-spy/backend/main.py | Endpoints, kiểm tra upload, EXIF/RGB normalization |
| Runtime | co-spy/backend/inference/runtime.py | Load model, tuần tự hoá CUDA, inference, guard, response |
| Fusion detector | detectors/sd_v1_4/fusion.py | High-frequency crop, multi-view transform, fusion head |
| Weights | pretrained | Weights Original và Fusion v3 |

Runtime tuần tự hoá inference để tránh hai graph VAE/CLIP lớn chạy đan xen trên CUDA. Model load ở background; health endpoint vẫn phản hồi trong lúc khởi tạo.

| Trường kết quả | Ý nghĩa |
| --- | --- |
| prediction | AI_GENERATED khi ai_probability > 0.5; còn lại REAL |
| confidence | Giá trị lớn hơn giữa xác suất AI và Real |
| raw_score | Logit trước sigmoid; dùng debug/calibration, không phải xác suất |
| processing_time_ms | Thời gian inference của model |
| total_processing_time_ms | Thời gian chạy cả Original và Fusion v3 |

## Cài đặt và chạy web

### Yêu cầu

- Python theo [environment.yml](environment.yml): Python 3.8, PyTorch 2.4.1, Torchvision 0.19.1.
- Node.js và npm.
- Checkpoint hợp lệ dưới thư mục pretrained.
- CUDA là tuỳ chọn. Nếu không có, runtime tự dùng CPU nhưng sẽ chậm hơn.

### 1. Môi trường Python

Tại root repository:

~~~bash
conda env create -f environment.yml
conda activate cospy
python -m pip install -r co-spy/backend/requirements.txt
~~~

File environment chứa stack model/research. Requirements backend bổ sung FastAPI, Uvicorn, multipart upload và Pillow.

### 2. Frontend và khởi động

~~~bash
cd co-spy
npm install
npm run dev
~~~

Lệnh mở Vite tại http://localhost:5173 và FastAPI tại http://127.0.0.1:8000. Vite proxy API requests tới backend.

Nếu Python trên PATH không đúng môi trường, PowerShell:

~~~powershell
$env:CO_SPY_PYTHON = "C:\path\to\env\python.exe"
npm run dev
~~~

Chạy riêng từng service:

~~~bash
# Terminal 1 — co-spy/
npm run dev:frontend

# Terminal 2 — co-spy/, Python environment đã kích hoạt
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
~~~

## Checkpoint, API và kiểm thử

Files bắt buộc:

~~~text
pretrained/
├── checkpoint_best.pth                 # Fusion v3
└── sd-v1_4/
    ├── semantic_weights.pth            # CO-SPY Original
    ├── artifact_weights.pth            # CO-SPY Original
    └── fusion_weights.pth              # CO-SPY Original
~~~

Semantic branch có thể cần OpenCLIP/SigLIP weights. Runtime ưu tiên .model-cache/openclip, sau đó pretrained và Hugging Face cache. Có thể chỉ định file qua CO_SPY_OPENCLIP_WEIGHTS.

Runtime dựng graph v3 đúng kiến trúc: Semantic + Artifact-VAE + NPR + v3 fusion head, kiểm tra state bắt buộc rồi mới nạp weights. Điều này thay thế lỗi nạp v3 theo layout v1 với strict=False, vốn có thể để head không khớp ở trạng thái random và tạo dự đoán không đáng tin.

Không nạp file PTH không rõ nguồn gốc: PyTorch deserialization có rủi ro thực thi mã.

| Biến môi trường | Công dụng |
| --- | --- |
| CO_SPY_PYTHON | Python executable dùng bởi npm run dev |
| CO_SPY_MODEL_CACHE | Cache model; mặc định .model-cache |
| CO_SPY_OPENCLIP_CACHE | Cache OpenCLIP; mặc định thư mục openclip trong model cache |
| CO_SPY_OPENCLIP_WEIGHTS | Đường dẫn trực tiếp OpenCLIP weights |

### GET /api/health

~~~json
{
  "status": "ok",
  "device": "cuda",
  "original_loaded": true,
  "fusion_v3_loaded": true,
  "errors": {}
}
~~~

Status là loading, ok hoặc degraded. Khi startup lỗi, xem errors và backend log.

### POST /api/predict

Gửi multipart/form-data với field image.

- Nhận JPEG, PNG, WEBP; tối đa 15 MB.
- 415: MIME không hỗ trợ; 413: rỗng/quá lớn; 422: không giải mã được; 503: model chưa sẵn sàng/lỗi inference.

~~~bash
curl -X POST http://127.0.0.1:8000/api/predict \
  -F "image=@path/to/image.png"
~~~

Response có original, fusion_v3, total_processing_time_ms, device. fusion_v3.branches có semantic, artifact, npr. fusion_v3.forensic trả views, xác suất NPR theo view, threshold, cờ guard, selected fusion view và model_prediction. model_prediction luôn giữ Fusion thô trước guard, nên phù hợp để debug/calibration.

### Kiểm thử

~~~bash
# Root repository
python -m pytest test_inference_runtime.py -q

# Frontend
cd co-spy
npm run build
npm run lint
~~~

Runtime tests gồm guard áp dụng (hai views ≥ 0.95) và guard không áp dụng (một view < 0.95). UI desktop hiện preview, hai panel Original/Fusion v3 và ba cards Semantic/Artifact-VAE/NPR; ở viewport thấp/hẹp vẫn có thể scroll để bảo toàn khả năng đọc.

## Nghiên cứu, giới hạn và trích dẫn

Web app là lớp inference. Workflow nghiên cứu gốc vẫn ở root:

~~~bash
# Evaluate pretrained trên CO-SPY-Bench
python main.py --gpu 0 --phase eval --train_dataset sd-v1_4 --pretrain

# Single-image CLI
python main.py --gpu 0 --phase test --train_dataset sd-v1_4 --pretrain

# Train NPR rồi train Fusion v3
python main.py --phase train --gpu 0 --mode branch --branch npr --train_dataset sd-v1_4 --epochs 20
python main.py --phase train --gpu 0 --mode fusion --train_dataset sd-v1_4 --fusion_head v3 --epochs 30
~~~

Xem [TRAIN.md](TRAIN.md), [TRAINING_RECOVERY_GUIDE.md](TRAINING_RECOVERY_GUIDE.md) và [CO-SPY-Bench](https://huggingface.co/datasets/ruojiruoli/Co-Spy-Bench).

- Xác suất model không phải bằng chứng pháp lý hoặc xác nhận nguồn ảnh.
- Resize, screenshot, recompress, hậu kỳ mạnh và mạng xã hội có thể làm đổi forensic signal.
- Có thể có false positive/false negative, nhất là với generator ngoài validation data.
- NPR guard đã giúp xử lý trường hợp ảnh ChatGPT gen đã quan sát, nhưng không phải benchmark mới. Cần đánh giá tập real/AI độc lập trước khi đổi ngưỡng hoặc dùng production.
- Không nên blur, denoise, sharpen, JPEG recompress hoặc resize ngẫu nhiên trước inference: chúng có thể xoá artifact Artifact-VAE/NPR cần.

~~~bibtex
@inproceedings{Cheng_2025_CVPR,
  author    = {Cheng, Siyuan and Lyu, Lingjuan and Wang, Zhenting and Zhang, Xiangyu and Sehwag, Vikash},
  title     = {CO-SPY: Combining Semantic and Pixel Features to Detect Synthetic Images by AI},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month     = {June},
  year      = {2025},
  pages     = {13455--13465}
}
~~~

Ghi nhận: [CNNDetection](https://github.com/PeterWang512/CNNDetection), [UniversalFakeDetect](https://github.com/WisconsinAIVision/UniversalFakeDetect), [NPR-DeepfakeDetection](https://github.com/chuangchuangtan/NPR-DeepfakeDetection), [DRCT](https://github.com/beibuwandeluori/DRCT).
