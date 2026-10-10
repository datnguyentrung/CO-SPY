"""
=============================================================================
FILE: detectors/sd-v1_4/semantic.py
CHỨC NĂNG & NHIỆM VỤ CỦA FILE:
    - Cài đặt bộ phát hiện nhánh Ngữ nghĩa (Semantic Detector) cho mô hình CO-SPY.
    - Sử dụng kiến trúc Vision Transformer từ mô hình OpenCLIP (bản SigLIP: ViT-SO400M-14-SigLIP-384).
    - Trích xuất các vector đặc trưng ngữ nghĩa mức cao (high-level semantic representation)
      nhằm phát hiện các điểm bất hợp lý về mặt vật lý, logic cấu trúc của ảnh do AI sinh ra.
    - Đóng băng (Freeze) toàn bộ trọng số của Vision Encoder, chỉ huấn luyện một lớp Linear (fc)
      ở tầng cuối để phân loại nhị phân.

DANH SÁCH CÁC PHƯƠNG THỨC / HÀM TRONG FILE:
    1. __init__(dim_clip, num_classes): Khởi tạo mô hình CLIP, đóng băng trọng số, tạo lớp classifier fc.
    2. _build_transforms(): Thiết lập pipeline tiền xử lý ảnh và tăng cường dữ liệu (train/val/test).
    3. forward(x, return_feat): Trích xuất vector đặc trưng CLIP và tính toán logit dự đoán.
    4. save_weights(weights_path): Lưu trọng số của riêng lớp Linear fc ra file .pth.
    5. load_weights(weights_path): Nạp trọng số cho lớp Linear fc từ file .pth.
=============================================================================
"""

import open_clip
import torch
from torchvision import transforms

from utils import data_augment

from .base import BaseDetector


class SemanticDetector(BaseDetector):
    """
    Bộ phát hiện ngữ nghĩa (Semantic Detector):
    - Sử dụng mô hình thị giác lớn Vision Transformer OpenCLIP (bản SigLIP: ViT-SO400M-14-SigLIP-384).
    - Trích xuất đặc trưng ngữ nghĩa mức cao (nhận biết tính hợp lý về logic cấu trúc của ảnh).
    - Đóng băng CLIP encoder, chỉ huấn luyện lớp tuyến tính Linear (fc) ở đầu ra.
    """

    # -------------------------------------------------------------------------
    # CHỨC NĂNG: Khởi tạo nhánh Semantic Detector, nạp OpenCLIP và đóng băng encoder
    # INPUT:
    #   - dim_clip (int): Số chiều vector đặc trưng đầu ra của CLIP (mặc định: 1152).
    #   - num_classes (int): Số lớp phân loại đầu ra (mặc định: 1 cho phân loại nhị phân).
    # OUTPUT:
    #   - Đối tượng SemanticDetector đã sẵn sàng huấn luyện hoặc đánh giá.
    # -------------------------------------------------------------------------
    def __init__(self, dim_clip=1152, num_classes=1):
        super().__init__()

        # Tải mô hình OpenCLIP pretrained (SigLIP ViT-SO400M-14, resolution 384x384)
        model_name = "ViT-SO400M-14-SigLIP-384"
        version = "webli"
        self.clip, _, _ = open_clip.create_model_and_transforms(
            model_name, pretrained=version
        )

        # Đóng băng visual encoder của CLIP, không tính toán gradient để tiết kiệm VRAM và giữ nguyên tri thức
        self.clip.requires_grad_(False)

        # Lớp phân loại nhị phân (Binary Classifier): Chuyển vector đặc trưng 1152 chiều về 1 logit
        self.fc = torch.nn.Linear(dim_clip, num_classes)

        # Xây dựng các bước tiền xử lý ảnh
        self._build_transforms()

    # -------------------------------------------------------------------------
    # CHỨC NĂNG: Xây dựng pipeline tiền xử lý ảnh và các phép biến đổi tăng cường (Augmentation)
    # INPUT: Không có
    # OUTPUT: None (cập nhật self.train_transform, self.val_transform, self.test_transform)
    # -------------------------------------------------------------------------
    def _build_transforms(self):
        """Thiết lập pipeline tiền xử lý ảnh và tăng cường dữ liệu (Augmentation)."""
        # Chuẩn hóa ảnh về khoảng [-1, 1]
        self.mean = [0.5, 0.5, 0.5]
        self.std = [0.5, 0.5, 0.5]

        # Độ phân giải yêu cầu của SigLIP là 384x384
        self.loadSize = 384
        self.cropSize = 384

        # Tăng cường dữ liệu: Làm mờ (Gaussian Blur) và nén ảnh JPEG với xác suất 50%
        self.blur_prob = 0.5
        self.blur_sig = [0.0, 3.0]
        self.jpg_prob = 0.5
        self.jpg_method = ["cv2", "pil"]
        self.jpg_qual = list(range(30, 101))

        # Cấu hình augmentation cho tập huấn luyện (Train)
        self.train_aug_config = {
            "blur_prob": self.blur_prob,
            "blur_sig": self.blur_sig,
            "jpg_prob": self.jpg_prob,
            "jpg_method": self.jpg_method,
            "jpg_qual": self.jpg_qual,
        }

        # Cấu hình augmentation cố định cho tập kiểm thử (Val)
        self.val_aug_config = {
            "blur_prob": self.blur_prob,
            "blur_sig": [self.blur_sig[0] + self.blur_sig[1] / 2],  # [1.5]
            "jpg_prob": self.jpg_prob,
            "jpg_method": ["pil"],
            "jpg_qual": [int((self.jpg_qual[0] + self.jpg_qual[-1]) / 2)],  # [65]
        }

        # Định nghĩa các phép biến đổi hình học
        random_crop = transforms.RandomCrop(self.cropSize)
        center_crop = transforms.CenterCrop(self.cropSize)
        flip_func = transforms.RandomHorizontalFlip()
        rz_func = transforms.Resize(self.loadSize)
        train_aug_func = transforms.Lambda(
            lambda x: data_augment(x, self.train_aug_config)
        )
        val_aug_func = transforms.Lambda(lambda x: data_augment(x, self.val_aug_config))

        # Pipeline cho tập Train: Resize -> Augment -> Cắt ngẫu nhiên -> Lật ngang -> ToTensor -> Normalize
        self.train_transform = transforms.Compose(
            [
                rz_func,
                train_aug_func,
                random_crop,
                flip_func,
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std),
            ]
        )

        # Pipeline cho tập Val
        self.val_transform = transforms.Compose(
            [
                rz_func,
                val_aug_func,
                random_crop,
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std),
            ]
        )

        # Pipeline cho tập Test: Resize -> Cắt trung tâm -> ToTensor -> Normalize
        self.test_transform = transforms.Compose(
            [
                rz_func,
                center_crop,
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std),
            ]
        )

    # -------------------------------------------------------------------------
    # CHỨC NĂNG: Lan truyền xuôi trích xuất đặc trưng CLIP và phân loại
    # INPUT:
    #   - x (torch.Tensor): Batch ảnh đầu vào kích thước (B, 3, 384, 384) đã chuẩn hóa.
    #   - return_feat (bool, optional): Nếu True, trả về cả vector đặc trưng CLIP (1152 chiều).
    # OUTPUT:
    #   - out (torch.Tensor): Logit đầu ra kích thước (B, 1) (hoặc tuple (feat, out) nếu return_feat=True).
    # -------------------------------------------------------------------------
    def forward(self, x, return_feat=False):
        """
        Lan truyền xuôi:
        1. Dùng CLIP visual encoder trích xuất vector đặc trưng ngữ nghĩa feat (1152 chiều).
        2. Đưa qua lớp Linear self.fc để tính toán điểm số out.
        """
        feat = self.clip.encode_image(x)
        out = self.fc(feat)
        if return_feat:
            return feat, out
        return out

    # -------------------------------------------------------------------------
    # CHỨC NĂNG: Lưu trọng số của lớp Linear fc ra file .pth
    # INPUT:
    #   - weights_path (str): Đường dẫn lưu file
    # OUTPUT:
    #   - None
    # -------------------------------------------------------------------------
    def save_weights(self, weights_path):
        """Chỉ lưu trọng số của lớp Linear fc (vì thân mạng CLIP đã đóng băng)."""
        save_params = {"fc.weight": self.fc.weight.cpu(), "fc.bias": self.fc.bias.cpu()}
        torch.save(save_params, weights_path)

    # -------------------------------------------------------------------------
    # CHỨC NĂNG: Nạp trọng số cho lớp Linear fc từ file .pth
    # INPUT:
    #   - weights_path (str): Đường dẫn file .pth cần nạp
    # OUTPUT:
    #   - None
    # -------------------------------------------------------------------------
    def load_weights(self, weights_path):
        """Nạp trọng số cho lớp Linear fc."""
        weights = torch.load(weights_path, map_location="cpu")
        self.fc.weight.data = weights["fc.weight"]
        self.fc.bias.data = weights["fc.bias"]
