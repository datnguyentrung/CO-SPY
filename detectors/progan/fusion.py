"""
=============================================================================
FILE: detectors/progan/fusion.py
CHỨC NĂNG & NHIỆM VỤ CỦA FILE:
    - Định nghĩa lớp CoSpyFusionDetector kết hợp (Late Fusion) 2 nhánh phát hiện độc lập
      của nhánh ProGAN: SemanticDetector và ArtifactDetector.
    - Nạp trọng số đã huấn luyện trước của cả 2 nhánh và đóng băng (freeze) hoàn toàn tham số của chúng.
    - Huấn luyện một lớp phân loại tuyến tính Linear 2 chiều (self.fc) nhận đầu vào là logits 
      từ cả 2 nhánh [pred_sem, pred_art] để đưa ra phán đoán tổng hợp cuối cùng.
    - Cung cấp hàm _build_fusion_transforms xây dựng pipeline tiền xử lý ảnh chung (không chuẩn hóa ở DataLoader 
      mà chuyển bước chuẩn hóa về từng nhánh trong hàm forward).

DANH SÁCH CÁC PHƯƠNG THỨC / HÀM TRONG FILE:
    1. CoSpyFusionDetector.__init__(self, semantic_weights_path, artifact_weights_path, num_classes):
       Khởi tạo bộ phát hiện Fusion, nạp trọng số 2 nhánh con, đóng băng tham số và tạo lớp fc.
    2. CoSpyFusionDetector._build_fusion_transforms(self):
       Xây dựng các pipeline biến đổi ảnh mà không có bước Normalize (chuẩn hóa được làm riêng trong forward).
    3. CoSpyFusionDetector.forward(self, x, return_feat):
       Lan truyền xuôi qua 2 nhánh đã đóng băng, ghép nối logits của 2 nhánh và phân loại qua fc.
    4. CoSpyFusionDetector.save_weights(self, weights_path):
       Lưu trọng số của lớp phân loại kết hợp fc ra file checkpoint.
    5. CoSpyFusionDetector.load_weights(self, weights_path):
       Nạp trọng số cho lớp phân loại fc từ file checkpoint.
=============================================================================
"""

import torch
from torchvision import transforms
from .base import BaseDetector
from .semantic import SemanticDetector
from .artifact import ArtifactDetector


# =============================================================================
# CLASS: CoSpyFusionDetector
# =============================================================================
class CoSpyFusionDetector(BaseDetector):
    """Mô hình phát hiện Fusion kết hợp nhánh ngữ nghĩa (Semantic) và nhánh dấu vết (Artifact) cho ProGAN."""

    # -------------------------------------------------------------------------
    # CHỨC NĂNG:
    #     Khởi tạo mạng Fusion: nạp mô hình SemanticDetector và ArtifactDetector,
    #     nạp trọng số cho cả 2 nhánh từ đường dẫn chỉ định, đóng băng gradient của 2 nhánh con,
    #     và khởi tạo lớp phân loại tuyến tính fc nhận vector logits 2 chiều.
    # INPUT:
    #     - semantic_weights_path (str): Đường dẫn file weights của nhánh Semantic.
    #     - artifact_weights_path (str): Đường dẫn file weights của nhánh Artifact.
    #     - num_classes (int): Số lớp phân loại đầu ra (mặc định 1).
    # OUTPUT:
    #     - Đối tượng CoSpyFusionDetector.
    # -------------------------------------------------------------------------
    def __init__(self, semantic_weights_path, artifact_weights_path, num_classes=1):
        super(CoSpyFusionDetector, self).__init__()

        # Load the semantic detector
        self.sem = SemanticDetector()
        self.sem.load_weights(semantic_weights_path)

        # Load the artifact detector
        self.art = ArtifactDetector()
        self.art.load_weights(artifact_weights_path)

        # Freeze the two pre-trained models
        for param in self.sem.parameters():
            param.requires_grad = False
        for param in self.art.parameters():
            param.requires_grad = False

        # Classifier
        self.fc = torch.nn.Linear(2, num_classes)

        # Transformations inside the forward function
        self.sem_transform = transforms.Compose([
            transforms.Normalize(self.sem.mean, self.sem.std)
        ])
        self.art_transform = transforms.Compose([
            transforms.Normalize(self.art.mean, self.art.std)
        ])

        # Build transforms (no normalization, done in forward)
        self._build_fusion_transforms()

    # -------------------------------------------------------------------------
    # CHỨC NĂNG:
    #     Xây dựng các bộ biến đổi ảnh (train, val, test) không kèm bước Normalize,
    #     do mỗi nhánh con có bộ chuẩn hóa riêng biệt được xử lý trong hàm forward.
    # INPUT:
    #     - Không có tham số ngoài self.
    # OUTPUT:
    #     - Không trả về giá trị (gán trực tiếp vào các thuộc tính transform).
    # -------------------------------------------------------------------------
    def _build_fusion_transforms(self):
        from utils import data_augment

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
    #     Lan truyền xuôi: chuẩn hóa ảnh đầu vào theo đặc tính từng nhánh,
    #     chạy dự đoán song song trên 2 nhánh con, ghép 2 logits lại và đưa qua lớp fc.
    # INPUT:
    #     - x (torch.Tensor): Tensor ảnh đầu vào (B, 3, H, W).
    #     - return_feat (bool): Tùy chọn trả về vector đặc trưng (logits 2 nhánh) hay không.
    # OUTPUT:
    #     - out (torch.Tensor): Logits phân loại nhị phân cuối cùng (B, 1).
    #     - (feat, out) nếu return_feat=True.
    # -------------------------------------------------------------------------
    def forward(self, x, return_feat=False):
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
    # CHỨC NĂNG:
    #     Lưu trọng số của lớp phân loại kết hợp fc về CPU và ghi vào file .pth.
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
