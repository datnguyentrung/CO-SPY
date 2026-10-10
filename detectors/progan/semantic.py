"""
=============================================================================
FILE: detectors/progan/semantic.py
CHỨC NĂNG & NHIỆM VỤ CỦA FILE:
    - Định nghĩa lớp SemanticDetector cho nhánh mô hình ProGAN:
        1. Sử dụng thị giác máy tính của OpenAI CLIP ("openai/clip-vit-large-patch14")
           để trích xuất các đặc trưng ngữ nghĩa mức cao (High-level Semantic Features).
        2. Đóng băng toàn bộ trọng số của CLIP encoder để giữ nguyên tri thức tổng quát.
        3. Huấn luyện lớp tuyến tính Linear (fc) nhận đặc trưng 768 chiều để phân loại ảnh Thật/Giả.
        4. Cung cấp cơ chế lưu và nạp trọng số tối ưu (chỉ lưu các tham số của lớp fc).

DANH SÁCH CÁC PHƯƠNG THỨC / HÀM TRONG FILE:
    1. SemanticDetector.__init__(self, dim_clip, num_classes): Nạp CLIP, đóng băng encoder, khởi tạo fc và transforms.
    2. SemanticDetector.forward(self, x, return_feat): Trích xuất đặc trưng CLIP và tính logits qua lớp fc.
    3. SemanticDetector.save_weights(self, weights_path): Lưu trọng số fc.weight và fc.bias ra file checkpoint.
    4. SemanticDetector.load_weights(self, weights_path): Nạp trọng số lớp fc từ file checkpoint.
=============================================================================
"""

import torch
from transformers import CLIPModel
from .base import BaseDetector


# =============================================================================
# CLASS: SemanticDetector
# =============================================================================
class SemanticDetector(BaseDetector):
    """Bộ phát hiện ảnh AI dựa trên đặc trưng ngữ nghĩa cấp cao từ mô hình nền tảng CLIP."""

    # -------------------------------------------------------------------------
    # CHỨC NĂNG:
    #     Khởi tạo SemanticDetector: nạp CLIP ViT-L/14, đóng băng trọng số CLIP,
    #     tạo lớp phân loại fc, thiết lập chuẩn hóa theo chuẩn CLIP và xây dựng transforms.
    # INPUT:
    #     - dim_clip (int): Số chiều vector đặc trưng CLIP (mặc định 768).
    #     - num_classes (int): Số nhãn phân loại đầu ra (mặc định 1).
    # OUTPUT:
    #     - Đối tượng SemanticDetector.
    # -------------------------------------------------------------------------
    def __init__(self, dim_clip=768, num_classes=1):
        super(SemanticDetector, self).__init__()

        # Get the pre-trained CLIP
        model_name = "openai/clip-vit-large-patch14"
        self.clip = CLIPModel.from_pretrained(model_name)

        # Freeze the CLIP visual encoder
        self.clip.requires_grad_(False)

        # Classifier
        self.fc = torch.nn.Linear(dim_clip, num_classes)

        # CLIP normalization
        self.mean = [0.48145466, 0.45782750, 0.40821073]
        self.std  = [0.26862954, 0.26130258, 0.27577711]

        # Data augmentation (override defaults)
        self.blur_prob = 0.5
        self.jpg_prob  = 0.5

        # Build transforms
        self._build_transforms()

    # -------------------------------------------------------------------------
    # CHỨC NĂNG:
    #     Lan truyền xuôi: đưa ảnh qua CLIP visual encoder để trích xuất đặc trưng
    #     ngữ nghĩa feat (B, 768) và đưa qua lớp phân loại fc.
    # INPUT:
    #     - x (torch.Tensor): Tensor ảnh kích thước (B, 3, H, W).
    #     - return_feat (bool): Trả về cả vector đặc trưng lẫn logits nếu True.
    # OUTPUT:
    #     - out (torch.Tensor): Logits phân loại nhị phân.
    #     - (feat, out) nếu return_feat=True.
    # -------------------------------------------------------------------------
    def forward(self, x, return_feat=False):
        feat = self.clip.get_image_features(x)
        out = self.fc(feat)
        if return_feat:
            return feat, out
        return out

    # -------------------------------------------------------------------------
    # CHỨC NĂNG:
    #     Lưu trọng số của lớp phân loại fc về CPU và ghi vào file .pth.
    # INPUT:
    #     - weights_path (str): Đường dẫn file lưu trữ.
    # OUTPUT:
    #     - Không trả về giá trị (None).
    # -------------------------------------------------------------------------
    def save_weights(self, weights_path):
        save_params = {"fc.weight": self.fc.weight.cpu(), "fc.bias": self.fc.bias.cpu()}
        torch.save(save_params, weights_path)

    # -------------------------------------------------------------------------
    # CHỨC NĂNG:
    #     Nạp trọng số cho lớp fc từ file checkpoint.
    # INPUT:
    #     - weights_path (str): Đường dẫn file checkpoint.
    # OUTPUT:
    #     - Không trả về giá trị (None).
    # -------------------------------------------------------------------------
    def load_weights(self, weights_path):
        weights = torch.load(weights_path, map_location='cpu')
        self.fc.weight.data = weights["fc.weight"]
        self.fc.bias.data = weights["fc.bias"]
