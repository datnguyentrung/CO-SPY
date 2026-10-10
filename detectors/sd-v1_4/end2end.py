"""
=============================================================================
FILE: detectors/sd-v1_4/end2end.py
CHỨC NĂNG & NHIỆM VỤ CỦA FILE:
    - Định nghĩa lớp End2EndDetector kết hợp nhánh Semantic và nhánh Artifact 
      để huấn luyện đồng bộ từ đầu đến cuối (End-to-End Training).
    - Tích hợp kỹ thuật Feature Dropout ngẫu nhiên giữa 2 nhánh (xác suất 30%) trong khi train
      nhằm tránh hiện tượng mạng bị thiên lệch hoặc phụ thuộc hoàn toàn vào một nhánh duy nhất.
    - Cung cấp cơ chế lưu trữ và nạp trọng số đồng bộ cho tất cả các thành phần: 
      sem.fc, art.fc, art.artifact_encoder và classifier.

DANH SÁCH CÁC PHƯƠNG THỨC / HÀM TRONG FILE:
    1. End2EndDetector.__init__(self, num_classes): Khởi tạo 2 nhánh SemanticDetector, ArtifactDetector, lớp Linear tổng hợp, và transforms.
    2. End2EndDetector._build_transforms(self): Xây dựng pipeline tiền xử lý ảnh chung (Data Augmentation, Resize, Crop).
    3. End2EndDetector.forward(self, x, return_feat, dropout_rate): Lan truyền xuôi qua 2 nhánh, áp dụng nhánh dropout nếu train, và phân loại qua fc.
    4. End2EndDetector.save_weights(self, weights_path): Lưu state_dict của sem_fc, art_fc, art_encoder và classifier vào file checkpoint.
    5. End2EndDetector.load_weights(self, weights_path): Nạp trọng số từ file checkpoint vào các module tương ứng.
=============================================================================
"""

import torch
import random
from torchvision import transforms
from utils import data_augment, weights2cpu
from .base import BaseDetector
from .semantic import SemanticDetector
from .artifact import ArtifactDetector


# =============================================================================
# CLASS: End2EndDetector
# =============================================================================
class End2EndDetector(BaseDetector):
    """
    Bộ phát hiện CO-SPY huấn luyện từ đầu đến cuối (End-to-End Detector):
    - Khác với chế độ Fusion (đóng băng 2 nhánh con), chế độ End-to-End cho phép cập nhật đồng thời
      cả phần biểu diễn đặc trưng và lớp phân loại kết hợp.
    - Áp dụng kỹ thuật Feature Dropout ngẫu nhiên trong lúc train (xác suất 30%) để tránh mô hình
      bị phụ thuộc quá mức vào chỉ 1 trong 2 nhánh.
    """

    # -------------------------------------------------------------------------
    # CHỨC NĂNG:
    #     Khởi tạo mạng End-to-End: tích hợp đồng thời SemanticDetector và ArtifactDetector,
    #     tạo lớp Linear kết hợp với số chiều vào là tổng kích thước đặc trưng của cả 2 nhánh.
    # INPUT:
    #     - num_classes (int): Số lớp phân loại đầu ra (mặc định 1 cho nhị phân).
    # OUTPUT:
    #     - Đối tượng End2EndDetector.
    # -------------------------------------------------------------------------
    def __init__(self, num_classes=1):
        super(End2EndDetector, self).__init__()

        # Khởi tạo nhánh Semantic
        self.sem = SemanticDetector()
        self.sem_dim = self.sem.fc.in_features

        # Khởi tạo nhánh Artifact
        self.art = ArtifactDetector()
        self.art_dim = self.art.fc.in_features

        # Lớp phân loại kết hợp nhận toàn bộ vector đặc trưng của cả 2 nhánh (sem_dim + art_dim)
        self.fc = torch.nn.Linear(self.sem_dim + self.art_dim, num_classes)

        # Chuẩn hóa riêng theo từng nhánh
        self.sem_transform = transforms.Compose([
            transforms.Normalize(self.sem.mean, self.sem.std)
        ])
        self.art_transform = transforms.Compose([
            transforms.Resize(self.art.cropSize, antialias=False),
            transforms.Normalize(self.art.mean, self.art.std)
        ])

        # Xây dựng các bước tiền xử lý ảnh chung
        self._build_transforms()

    # -------------------------------------------------------------------------
    # CHỨC NĂNG:
    #     Xây dựng các pipeline biến đổi dữ liệu ảnh (train_transform, val_transform, test_transform)
    #     cho toàn bộ mô hình End-to-End.
    # INPUT:
    #     - Không có tham số ngoài self.
    # OUTPUT:
    #     - Không trả về giá trị (khởi tạo các thuộc tính transform).
    # -------------------------------------------------------------------------
    def _build_transforms(self):
        self.loadSize = 384
        self.cropSize = 384

        self.blur_prob = 0.0
        self.blur_sig = [0.0, 3.0]
        self.jpg_prob = 0.5
        self.jpg_method = ['cv2', 'pil']
        self.jpg_qual = list(range(70, 96))

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

        self.train_transform = transforms.Compose([
            flip_func,
            aug_func,
            rz_func,
            random_crop,
            transforms.ToTensor(),
        ])

        self.val_transform = transforms.Compose([
            rz_func,
            random_crop,
            transforms.ToTensor(),
        ])

        self.test_transform = transforms.Compose([
            rz_func,
            center_crop,
            transforms.ToTensor(),
        ])

    # -------------------------------------------------------------------------
    # CHỨC NĂNG:
    #     Lan truyền xuôi (Forward pass):
    #     - Đưa ảnh x qua sem_transform và art_transform.
    #     - Trích xuất đặc trưng và hệ số tự kích hoạt coeff từ cả hai nhánh con.
    #     - Trong lúc train, áp dụng dropout ngẫu nhiên một trong hai nhánh với tỷ lệ dropout_rate.
    #     - Ghép nối vector đặc trưng và đưa qua lớp tuyến tính fc.
    # INPUT:
    #     - x (torch.Tensor): Tensor ảnh đầu vào kích thước (B, 3, H, W).
    #     - return_feat (bool): Nếu True, trả về cả vector đặc trưng lẫn logits đầu ra.
    #     - dropout_rate (float): Xác suất dropout ngẫu nhiên một nhánh trong khi train (mặc định 0.3).
    # OUTPUT:
    #     - out (torch.Tensor): Logits phân loại nhị phân (B, 1).
    #     - (feat, out) nếu return_feat=True.
    # -------------------------------------------------------------------------
    def forward(self, x, return_feat=False, dropout_rate=0.3):
        x_sem = self.sem_transform(x)
        x_art = self.art_transform(x)

        sem_feat, sem_coeff = self.sem(x_sem, return_feat=True)
        art_feat, art_coeff = self.art(x_art, return_feat=True)

        # Kỹ thuật Dropout nhánh đặc trưng trong lúc huấn luyện
        if self.training:
            if random.random() < dropout_rate:
                idx_drop = random.randint(0, 1)
                if idx_drop == 0:
                    sem_coeff = torch.zeros_like(sem_coeff)
                else:
                    art_coeff = torch.zeros_like(art_coeff)

        # Nối vector đặc trưng 2 nhánh sau khi nhân với hệ số điều chế
        feat = torch.cat([sem_coeff * sem_feat, art_coeff * art_feat], dim=1)
        out = self.fc(feat)

        if return_feat:
            return feat, out
        return out

    # -------------------------------------------------------------------------
    # CHỨC NĂNG:
    #     Lưu trữ toàn bộ tham số của mô hình End-to-End ra file: 
    #     gồm sem.fc, art.fc, art.artifact_encoder và classifier (self.fc).
    # INPUT:
    #     - weights_path (str): Đường dẫn file .pth cần ghi.
    # OUTPUT:
    #     - Không trả về giá trị (None).
    # -------------------------------------------------------------------------
    def save_weights(self, weights_path):
        save_params = {
            "sem_fc": weights2cpu(self.sem.fc.state_dict()),
            "art_fc": weights2cpu(self.art.fc.state_dict()),
            "art_encoder": weights2cpu(self.art.artifact_encoder.state_dict()),
            "classifier": weights2cpu(self.fc.state_dict()),
        }
        torch.save(save_params, weights_path)

    # -------------------------------------------------------------------------
    # CHỨC NĂNG:
    #     Nạp trọng số từ file checkpoint cho tất cả các thành phần trong End2EndDetector.
    # INPUT:
    #     - weights_path (str): Đường dẫn file .pth chứa trọng số đã lưu.
    # OUTPUT:
    #     - Không trả về giá trị (None).
    # -------------------------------------------------------------------------
    def load_weights(self, weights_path):
        weights = torch.load(weights_path, map_location='cpu')
        self.sem.fc.load_state_dict(weights["sem_fc"])
        self.art.fc.load_state_dict(weights["art_fc"])
        self.art.artifact_encoder.load_state_dict(weights["art_encoder"])
        self.fc.load_state_dict(weights["classifier"])
