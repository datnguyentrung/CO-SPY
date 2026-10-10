"""
=============================================================================
FILE: detectors/progan/__init__.py
CHỨC NĂNG & NHIỆM VỤ CỦA FILE:
    - Đóng vai trò là package initializer cho module phát hiện ảnh do mạng đối nghịch
      tạo sinh ProGAN (Progressive Growing of GANs) tạo ra.
    - Xuất khẩu (export) và tập hợp toàn bộ các lớp Detector chủ chốt của nhánh ProGAN:
        1. BaseDetector: Lớp cơ sở trừu tượng định nghĩa khung sườn (blueprint) cho các detector thuộc ProGAN.
        2. SemanticDetector: Bộ phát hiện dựa trên đặc trưng ngữ nghĩa trích xuất từ CLIP ViT-L/14 của OpenAI.
        3. ArtifactDetector: Bộ phát hiện dấu vết nội suy điểm ảnh lân cận (NPR - Neighboring Pixel Relationships)
           thông qua mạng ResNet chuyên dụng.
        4. CoSpyFusionDetector: Bộ phát hiện kết hợp (Late Fusion) giữa đặc trưng ngữ nghĩa CLIP và dấu vết tần số NPR.
        5. End2EndDetector: Bộ phát hiện huấn luyện đồng bộ cả 2 nhánh từ đầu đến cuối có tích hợp Dropout nhánh.

DANH SÁCH CÁC PHƯƠNG THỨC / HÀM TRONG FILE:
    - File này là module export package, không định nghĩa hàm trực tiếp.
    - Cung cấp danh sách các lớp thông qua `__all__`:
        + ArtifactDetector: Phát hiện sai lệch do phép nội suy upsampling của kiến trúc mạng GAN.
        + BaseDetector: Khung lớp cơ sở cho suy luận (predict), transforms, và lưu/nạp trọng số.
        + CoSpyFusionDetector: Ghép nối logits của 2 nhánh và đưa ra phân loại kết hợp.
        + End2EndDetector: Huấn luyện liên hợp và cập nhật toàn bộ tham số cùng lúc.
        + SemanticDetector: Trích xuất vector đặc trưng ngữ nghĩa 768 chiều từ OpenAI CLIP.
=============================================================================
"""

from .artifact import ArtifactDetector
from .base import BaseDetector
from .end2end import End2EndDetector
from .fusion import CoSpyFusionDetector
from .semantic import SemanticDetector

__all__ = [
    "ArtifactDetector",
    "BaseDetector",
    "CoSpyFusionDetector",
    "End2EndDetector",
    "SemanticDetector",
]
