"""
=============================================================================
FILE: detectors/sd-v1_4/base.py
CHỨC NĂNG & NHIỆM VỤ CỦA FILE:
    - Định nghĩa lớp cơ sở trừu tượng (BaseDetector) kế thừa từ torch.nn.Module và abc.ABC.
    - Đóng vai trò là khung mẫu (blueprint) thống nhất cho tất cả các bộ dò phát hiện
      (SemanticDetector, ArtifactDetector, CoSpyFusionDetector, End2EndDetector).
    - Cung cấp sẵn các phương thức dùng chung phục vụ: suy luận (predict), lưu trọng số
      (save_weights) và tải trọng số (load_weights).

DANH SÁCH CÁC PHƯƠNG THỨC / HÀM TRONG FILE:
    1. __init__(): Khởi tạo lớp cơ sở nơ-ron.
    2. _build_transforms(): Phương thức trừu tượng khởi tạo pipeline xử lý ảnh.
    3. forward(x, return_feat): Phương thức trừu tượng lan truyền xuôi tính toán output.
    4. predict(inputs): Suy luận và chuyển đổi logit sang xác suất phần trăm (0.0 đến 1.0).
    5. save_weights(weights_path): Lưu trọng số sang định dạng file .pth trên CPU.
    6. load_weights(weights_path): Nạp trọng số từ file .pth vào mô hình.
=============================================================================
"""

from abc import ABC, abstractmethod

import torch


class BaseDetector(torch.nn.Module, ABC):
    """
    Lớp cơ sở trừu tượng (Abstract Base Class) cho tất cả các bộ phát hiện (Detector).
    Cung cấp các hàm dùng chung: dự đoán (predict), lưu trọng số (save_weights), tải trọng số (load_weights).
    """

    # -------------------------------------------------------------------------
    # CHỨC NĂNG: Khởi tạo mô hình cơ sở
    # INPUT: Không có
    # OUTPUT: Đối tượng BaseDetector
    # -------------------------------------------------------------------------
    def __init__(self):
        super().__init__()

    # -------------------------------------------------------------------------
    # CHỨC NĂNG: Phương thức trừu tượng định nghĩa các bước tiền xử lý ảnh (transforms)
    # INPUT: Không có
    # OUTPUT: None (gán vào các thuộc tính self.train_transform, self.val_transform, self.test_transform)
    # -------------------------------------------------------------------------
    @abstractmethod
    def _build_transforms(self):
        """Xây dựng các bước tiền xử lý ảnh (transforms) cho tập train, val và test."""

    # -------------------------------------------------------------------------
    # CHỨC NĂNG: Phương thức trừu tượng tính toán lan truyền xuôi (Forward pass)
    # INPUT:
    #   - x (torch.Tensor): Batch ảnh đầu vào có kích thước (B, C, H, W).
    #   - return_feat (bool, optional): Có trả về vector đặc trưng hay không. Mặc định là False.
    # OUTPUT:
    #   - outputs (torch.Tensor): Logit đầu ra kích thước (B, 1) hoặc tuple (feat, outputs).
    # -------------------------------------------------------------------------
    @abstractmethod
    def forward(self, x, return_feat=False):
        """Lan truyền xuôi (Forward pass) của mạng nơ-ron."""

    # -------------------------------------------------------------------------
    # CHỨC NĂNG: Dự đoán xác suất ảnh do AI tạo ra (0.0 đến 1.0)
    # INPUT:
    #   - inputs (torch.Tensor): Tensor ảnh kích thước (B, C, H, W)
    # OUTPUT:
    #   - prediction (list[float]): Danh sách xác suất (giá trị từ 0.0 đến 1.0)
    #     Trong đó: điểm > 0.5 là AI sinh (AI-Generated), <= 0.5 là Ảnh thật (Real).
    # -------------------------------------------------------------------------
    def predict(self, inputs):
        """
        Hàm dự đoán xác suất ảnh do AI tạo ra (0.0 đến 1.0).
        - Đưa ảnh về cùng thiết bị (CPU/GPU) với mô hình.
        - Áp dụng hàm sigmoid lên output logit.
        - Trả về danh sách xác suất: điểm > 0.5 là AI, < 0.5 là Real.
        """
        inputs = inputs.to(next(self.parameters()).device)
        outputs = self.forward(inputs)
        prediction = outputs.sigmoid().flatten().tolist()
        return prediction

    # -------------------------------------------------------------------------
    # CHỨC NĂNG: Lưu toàn bộ trọng số (weights/state_dict) của mô hình ra file đĩa
    # INPUT:
    #   - weights_path (str): Đường dẫn file đích (ví dụ: 'ckpt/sd-v1_4/best_model.pth')
    # OUTPUT:
    #   - None (file trọng số được ghi xuống ổ đĩa)
    # -------------------------------------------------------------------------
    def save_weights(self, weights_path):
        """Lưu trọng số mô hình (state_dict) ra file .pth trên CPU."""
        save_params = {k: v.cpu() for k, v in self.state_dict().items()}
        torch.save(save_params, weights_path)

    # -------------------------------------------------------------------------
    # CHỨC NĂNG: Tải trọng số từ file lưu trữ vào kiến trúc mô hình hiện tại
    # INPUT:
    #   - weights_path (str): Đường dẫn file .pth cần nạp
    # OUTPUT:
    #   - None (trọng số của self được cập nhật)
    # -------------------------------------------------------------------------
    def load_weights(self, weights_path):
        """Nạp trọng số từ file .pth vào mô hình."""
        weights = torch.load(weights_path, map_location="cpu")
        self.load_state_dict(weights, strict=False)
