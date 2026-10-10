"""
=============================================================================
FILE: detectors/sd-v1_4/fusion.py
CHỨC NĂNG & NHIỆM VỤ CỦA FILE:
    - Cài đặt bộ phát hiện tổng hợp CO-SPY Fusion (CoSpyFusionDetector).
    - Đây là kiến trúc đề xuất chính (Best Practice) của bài báo CVPR 2025:
        + Tích hợp 2 nhánh phát hiện bổ trợ cho nhau: SemanticDetector (đặc trưng ngữ nghĩa toàn cục)
          và ArtifactDetector (đặc trưng khiếm khuyết tần số / điểm ảnh cục bộ).
        + Đóng băng (Freeze) toàn bộ tham số của cả 2 nhánh con để giữ vững biểu diễn đặc trưng.
        + Chỉ sử dụng 1 lớp Linear (fc) có trọng số học Linear(2, 1) để hiệu chuẩn (calibrate)
          và kết hợp 2 điểm số thành xác suất tổng hợp duy nhất.

DANH SÁCH CÁC PHƯƠNG THỨC / HÀM TRONG FILE:
    1. __init__(semantic_weights_path, artifact_weights_path, num_classes):
       Nạp 2 nhánh con, đóng băng tham số, tạo lớp kết hợp Linear(2, 1).
    2. _build_transforms(): Xây dựng pipeline tiền xử lý ảnh chung (train/val/test).
    3. forward(x, return_feat): Lan truyền xuôi qua 2 nhánh, ghép nối và phân loại.
    4. save_weights(weights_path): Lưu trọng số của lớp Linear fc.
    5. load_weights(weights_path): Nạp trọng số cho lớp Linear fc.
=============================================================================
"""

import torch
from torchvision import transforms

from utils import data_augment

from .artifact import ArtifactDetector
from .base import BaseDetector
from .semantic import SemanticDetector


class CoSpyFusionDetector(BaseDetector):
    """
    Bộ phát hiện CO-SPY Fusion (Hiệu chỉnh và kết hợp 2 nhánh Semantic và Artifact).
    Ý tưởng cốt lõi của bài báo:
    - Đóng băng (Freeze) 2 nhánh Semantic (CLIP) và Artifact (VAE).
    - Dùng 1 lớp tuyến tính Linear(2, 1) để học cách kết hợp tối ưu xác suất từ 2 nguồn đặc trưng.
    """

    # -------------------------------------------------------------------------
    # CHỨC NĂNG: Khởi tạo mô hình Fusion, nạp trọng số 2 nhánh con và đóng băng
    # INPUT:
    #   - semantic_weights_path (str): Đường dẫn file trọng số của nhánh Semantic
    #   - artifact_weights_path (str): Đường dẫn file trọng số của nhánh Artifact
    #   - num_classes (int): Số lớp phân loại đầu ra (mặc định: 1)
    # OUTPUT:
    #   - Đối tượng CoSpyFusionDetector
    # -------------------------------------------------------------------------
    def __init__(self, semantic_weights_path, artifact_weights_path, num_classes=1):
        super().__init__()

        # Khởi tạo và nạp trọng số đã huấn luyện cho nhánh Semantic
        self.sem = SemanticDetector()
        self.sem.load_weights(semantic_weights_path)

        # Khởi tạo và nạp trọng số đã huấn luyện cho nhánh Artifact
        self.art = ArtifactDetector()
        self.art.load_weights(artifact_weights_path)

        # Đóng băng (Freeze) toàn bộ tham số của 2 nhánh con, không cho cập nhật gradient
        for param in self.sem.parameters():
            param.requires_grad = False
        for param in self.art.parameters():
            param.requires_grad = False

        # Lớp phân loại kết hợp (Fusion Classifier): Nhận 2 giá trị đầu ra từ 2 nhánh -> đưa ra kết quả cuối cùng
        self.fc = torch.nn.Linear(2, num_classes)

        # Các phép biến đổi dữ liệu thực hiện bên trong hàm forward
        # Chuẩn hóa riêng theo mean/std của từng nhánh; nhánh artifact cần resize theo cropSize riêng
        self.sem_transform = transforms.Compose(
            [transforms.Normalize(self.sem.mean, self.sem.std)]
        )
        self.art_transform = transforms.Compose(
            [
                transforms.Resize(self.art.cropSize, antialias=False),
                transforms.Normalize(self.art.mean, self.art.std),
            ]
        )

        # Xây dựng các phép tiền xử lý ảnh chung (không normalize ở đây vì đã làm trong forward)
        self._build_transforms()

    # -------------------------------------------------------------------------
    # CHỨC NĂNG: Xây dựng pipeline tiền xử lý ảnh chung cho train, val và test
    # INPUT: Không có
    # OUTPUT: None (cập nhật self.train_transform, self.val_transform, self.test_transform)
    # -------------------------------------------------------------------------
    def _build_transforms(self):
        """Xây dựng pipeline tiền xử lý ảnh chung cho cả tập train, validation và test."""

        # Kích thước ảnh đầu vào: load 384x384 và crop 384x384
        self.loadSize = 384
        self.cropSize = 384

        # Cấu hình tăng cường dữ liệu (Data Augmentation): Làm mờ (Blur) và nén JPEG ngẫu nhiên
        self.blur_prob = 0.0
        self.blur_sig = [0.0, 3.0]
        self.jpg_prob = 0.5
        self.jpg_method = ["cv2", "pil"]
        self.jpg_qual = list(range(70, 96))

        # Đóng gói cấu hình augment
        self.aug_config = {
            "blur_prob": self.blur_prob,
            "blur_sig": self.blur_sig,
            "jpg_prob": self.jpg_prob,
            "jpg_method": self.jpg_method,
            "jpg_qual": self.jpg_qual,
        }

        random_crop = transforms.RandomCrop(self.cropSize)
        center_crop = transforms.CenterCrop(self.cropSize)
        flip_func = transforms.RandomHorizontalFlip()
        rz_func = transforms.Resize(self.loadSize)
        aug_func = transforms.Lambda(lambda x: data_augment(x, self.aug_config))

        # Pipeline cho tập huấn luyện (Train): Lật ngang, augment, resize, crop ngẫu nhiên, chuyển tensor
        self.train_transform = transforms.Compose(
            [
                flip_func,
                aug_func,
                rz_func,
                random_crop,
                transforms.ToTensor(),
            ]
        )

        # Pipeline cho tập kiểm thử validation (Val)
        self.val_transform = transforms.Compose(
            [
                rz_func,
                random_crop,
                transforms.ToTensor(),
            ]
        )

        # Pipeline cho tập kiểm thử thực tế (Test): Resize và cắt chính giữa (Center Crop)
        self.test_transform = transforms.Compose(
            [
                rz_func,
                center_crop,
                transforms.ToTensor(),
            ]
        )

    # -------------------------------------------------------------------------
    # CHỨC NĂNG: Lan truyền xuôi qua 2 nhánh con, ghép nối và phân loại
    # INPUT:
    #   - x (torch.Tensor): Batch ảnh đầu vào (B, 3, 384, 384) chưa chuẩn hóa.
    #   - return_feat (bool, optional): Có trả về vector ghép nối [pred_sem, pred_art] hay không.
    # OUTPUT:
    #   - out (torch.Tensor): Logit đầu ra kích thước (B, 1) (hoặc tuple (feat, out)).
    # -------------------------------------------------------------------------
    def forward(self, x, return_feat=False):
        """
        Lan truyền xuôi (Forward):
        1. Đưa ảnh x qua biến đổi riêng của từng nhánh (sem_transform, art_transform).
        2. Nhánh Semantic tính toán pred_sem.
        3. Nhánh Artifact tính toán pred_art.
        4. Nối 2 kết quả lại (torch.cat) thành vector [pred_sem, pred_art].
        5. Đưa qua lớp tuyến tính self.fc để ra kết quả logit cuối cùng.
        """
        x_sem = self.sem_transform(x)
        x_art = self.art_transform(x)
        pred_sem = self.sem(x_sem)
        pred_art = self.art(x_art)
        feat = torch.cat([pred_sem, pred_art], dim=1)
        out = self.fc(feat)
        if return_feat:
            return feat, out
        return out

    # -------------------------------------------------------------------------
    # CHỨC NĂNG: Lưu riêng trọng số của lớp Linear fc ra file .pth
    # INPUT:
    #   - weights_path (str): Đường dẫn lưu file
    # OUTPUT:
    #   - None
    # -------------------------------------------------------------------------
    def save_weights(self, weights_path):
        """Lưu riêng trọng số của lớp Linear fc (vì 2 nhánh con đã đóng băng)."""
        save_params = {"fc.weight": self.fc.weight.cpu(), "fc.bias": self.fc.bias.cpu()}
        torch.save(save_params, weights_path)

    # -------------------------------------------------------------------------
    # CHỨC NĂNG: Nạp trọng số cho riêng lớp Linear fc từ file .pth
    # INPUT:
    #   - weights_path (str): Đường dẫn file .pth cần nạp
    # OUTPUT:
    #   - None
    # -------------------------------------------------------------------------
    def load_weights(self, weights_path):
        """Nạp trọng số cho riêng lớp Linear fc."""
        weights = torch.load(weights_path, map_location="cpu")
        self.fc.weight.data = weights["fc.weight"]
        self.fc.bias.data = weights["fc.bias"]
