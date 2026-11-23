# Tiền xử lý ảnh trước khi suy luận Fusion v3

Tài liệu này mô tả pipeline **đang được chạy ở ứng dụng** trước và trong lúc đưa một ảnh vào `CO-SPY Fusion v3`. Đây là xử lý ở thời điểm dự đoán (inference), **không phải huấn luyện lại** checkpoint `pretrained/checkpoint_best.pth`.

Mục tiêu của thay đổi là tăng tín hiệu forensic cho ảnh do ChatGPT/DALL·E hoặc công cụ sinh ảnh tạo ra. Với ảnh đã kiểm thử, bước này giúp Fusion v3 nhận diện đúng là **AI generated** thay vì bị hai nhánh Semantic/Artifact nghiêng về ảnh thật.

> Kết quả không phải là bảo đảm tuyệt đối với mọi ảnh AI. Ngưỡng và hiệu quả cần tiếp tục được kiểm chứng bằng bộ ảnh thật/AI độc lập.

## Luồng xử lý

```text
Ảnh tải lên
  → sửa hướng EXIF + giải mã RGB
  → Semantic: một ảnh chuẩn hoá
  → Artifact/NPR: hai forensic views
       1. center crop căn theo block 8 px
       2. crop có năng lượng chi tiết cao nhất
  → chạy Fusion v3 cho từng forensic view
  → chọn Fusion evidence nghiêng AI nhất
  → NPR consensus guard (chỉ ép AI khi cả hai view cùng rất chắc chắn)
  → kết quả cuối + metadata forensic
```

## Các thao tác và thuật toán

| Bước | Tên trong mã nguồn | Mục đích | Cách hoạt động |
| --- | --- | --- | --- |
| 1 | `ImageOps.exif_transpose` | Tránh ảnh bị phân tích sai chiều vì metadata EXIF. | Xoay/lật pixel theo cờ Orientation của ảnh JPEG trước khi suy luận. |
| 2 | `convert("RGB")` | Tạo đầu vào màu thống nhất cho tất cả detector. | Loại bỏ khác biệt giữa RGBA, grayscale, palette… và đưa ảnh về ba kênh RGB. |
| 3 | `AlignedCrop` | Giữ crop ổn định, phù hợp tín hiệu nén/block của ảnh. | Lấy crop 224×224 ở tâm ảnh; toạ độ crop được căn theo bội số 8 px. Ảnh nhỏ hơn được phóng bằng bicubic trước khi crop. |
| 4 | `HighFrequencyCrop` | Chủ động tìm vùng có nhiều chi tiết pixel/biên, nơi thường chứa artefact sinh ảnh, resize hoặc nén. | Chuyển ảnh thành grayscale, thu nhỏ tối đa 256 px, tính sai phân tuyệt đối theo ngang và dọc (`|dx| + |dy|`), cộng năng lượng trong các ô cục bộ và chọn crop có điểm cao nhất. Đây là proxy high-pass kiểu DCT, không cần OpenCV. |
| 5 | Hai forensic views | Không để quyết định phụ thuộc duy nhất vào vùng trung tâm vốn có thể rất mượt. | Nhánh Artifact-VAE và NPR lần lượt xem `center` và `high_frequency`; nhánh Semantic chỉ chạy một lần để giữ chi phí hợp lý. |
| 6 | `max` theo xác suất AI | Không bỏ qua bằng chứng AI mạnh xuất hiện ở một vùng ảnh. | Kết quả Fusion có xác suất AI cao hơn giữa hai view được chọn làm kết quả Fusion ban đầu. Với thẻ Branch signals, Artifact-VAE/NPR cũng hiển thị evidence AI mạnh hơn trong hai view. |
| 7 | NPR consensus guard | Chặn trường hợp Fusion cuối vẫn nói Real trong khi cả hai phân tích forensic đều rất chắc ảnh là AI. | Nếu **cả `center` lẫn `high_frequency`** có xác suất AI từ NPR ≥ `0.95`, kết quả cuối bị nâng thành AI; xác suất trả về là giá trị thấp hơn trong hai xác suất NPR. Nếu một view dưới 95%, guard không can thiệp. |

## Vì sao cách này có thể bắt ảnh ChatGPT tốt hơn

Ảnh sinh thường có thể rất thuyết phục ở ngữ nghĩa: người, vật, phong cảnh hợp lý, vì vậy Semantic dễ đánh giá là ảnh thật. Crop tâm cũng có thể nằm ở vùng bầu trời/da/phông nền mượt, khiến Artifact-VAE ít tín hiệu.

`HighFrequencyCrop` không làm đẹp hay biến đổi nội dung ảnh. Nó chỉ chọn một vùng khác có mật độ biên/chi tiết cao hơn để các nhánh forensic tìm các dấu vết như cấu trúc texture bất thường, chi tiết tổng hợp, nội suy/resampling hoặc đặc trưng tần số. Khi NPR nhìn thấy cùng một bằng chứng ở **hai view độc lập**, consensus guard mới cho phép đảo kết quả; nhờ vậy giảm rủi ro một crop nhiễu đơn lẻ làm sai nhãn.

## Quy tắc quyết định chính xác

Gọi `p_center` và `p_high_frequency` là xác suất AI của NPR:

```text
if min(p_center, p_high_frequency) >= 0.95:
    final_label = "AI generated"
    final_ai_probability = min(p_center, p_high_frequency)
else:
    final_label = Fusion v3 ban đầu
    final_ai_probability = Fusion v3 ban đầu
```

Ngưỡng `0.95` được khai báo tại `NPR_GUARD_AI_THRESHOLD`. Điều kiện dùng `min(...)`, thay vì `max(...)`, nên một view nhận diện AI mạnh là chưa đủ để guard đổi kết quả cuối.

## Khả năng truy vết kết quả

API vẫn trả kết quả Fusion gốc, chưa bị guard điều chỉnh, tại:

```text
fusion_v3.forensic.model_prediction
```

Metadata forensic còn chứa:

- `views`: `center`, `high_frequency`
- `npr_ai_probabilities`: xác suất AI của NPR trên từng view
- `npr_guard_threshold`: `0.95`
- `npr_guard_applied`: guard có đổi nhãn cuối hay không
- `selected_fusion_view`: view có Fusion evidence AI lớn nhất

Nhờ đó UI hoặc log có thể giải thích vì sao ảnh bị kết luận là AI, thay vì chỉ hiển thị nhãn cuối.

## Ảnh hưởng hiệu năng và giới hạn

- Semantic chạy một lần; Artifact/NPR/Fusion cần xử lý thêm một forensic view, do đó độ trễ có thể tăng gần gấp đôi phần forensic.
- Pipeline không thêm watermark detector, OCR, metadata classifier hoặc mô hình mới; nó chỉ tận dụng các nhánh đã có trong checkpoint Fusion v3.
- Không nên áp dụng blur, denoise, sharpen, JPEG recompression hoặc resize ngẫu nhiên trước model: chúng có thể xoá chính forensic evidence mà Artifact-VAE và NPR cần đọc.
- Cần đánh giá ngưỡng 95% trên một tập validation gồm ảnh camera/ảnh chỉnh sửa thật và ảnh sinh từ nhiều phiên bản ChatGPT/DALL·E trước khi thay đổi ngưỡng trong production.

## Mã nguồn liên quan

- `co-spy/backend/main.py` — chuẩn hoá EXIF và RGB khi nhận upload.
- `detectors/sd_v1_4/fusion.py` — `HighFrequencyCrop`, `ArtifactViewTransform` và `AlignedCrop`.
- `co-spy/backend/inference/runtime.py` — chạy hai view, chọn evidence và áp dụng `NPR consensus guard`.
- `test_inference_runtime.py` — kiểm thử trường hợp guard được áp dụng và không được áp dụng.
