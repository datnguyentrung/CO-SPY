"""
=============================================================================
FILE: detectors/sd_v1_4/__init__.py
CHỨC NĂNG & NHIỆM VỤ CỦA FILE:
    - Đóng vai trò là package initializer cho module phát hiện ảnh giả tạo bởi
      mô hình khuếch tán tiềm ẩn Stable Diffusion v1.4 (sd-v1_4).
    - Xuất khẩu (export) và tập hợp toàn bộ các lớp Detector chủ chốt của nhánh SD:
        1. BaseDetector: Lớp cơ sở trừu tượng định nghĩa khung sườn (blueprint) cho các detector.
        2. SemanticDetector: Bộ phát hiện dựa trên đặc trưng ngữ nghĩa trích xuất từ OpenCLIP ViT-H/14.
        3. ArtifactDetector: Bộ phát hiện dấu vết nhân tạo/sai lệch tái tạo ảnh thông qua AutoencoderKL (VAE).
        4. CoSpyFusionDetector: Bộ phát hiện kết hợp (Late Fusion) giữa nhánh ngữ nghĩa và nhánh dấu vết VAE.
        5. End2EndDetector: Bộ phát hiện huấn luyện đồng bộ từ đầu đến cuối (End-to-End) có tích hợp Feature Dropout.

DANH SÁCH CÁC PHƯƠNG THỨC / HÀM TRONG FILE:
    - File này là module export package, không định nghĩa hàm trực tiếp.
    - Cung cấp danh sách các lớp thông qua `__all__`:
        + ArtifactDetector: Phát hiện sai lệch tái tạo ảnh từ Stable Diffusion VAE.
        + BaseDetector: Khung lớp cơ sở cho suy luận (predict), transforms, và lưu/nạp trọng số.
        + CoSpyFusionDetector: Ghép nối đặc trưng và phân loại tối ưu sau khi 2 nhánh đã huấn luyện.
        + End2EndDetector: Cập nhật trọng số toàn bộ hệ thống cùng lúc.
        + SemanticDetector: Phân tích đặc trưng ngữ nghĩa cấp cao từ mô hình OpenCLIP.
=============================================================================
"""

from .artifact import ArtifactDetector
from .base import BaseDetector
from .end2end import End2EndDetector
from .fusion import CoSpyFusionDetector
from .npr import NPRDetector
from .semantic import SemanticDetector

__all__ = [
    "ArtifactDetector",
    "BaseDetector",
    "CoSpyFusionDetector",
    "End2EndDetector",
    "NPRDetector",
    "SemanticDetector",
]
