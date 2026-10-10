"""
=============================================================================
FILE: detectors/progan/base.py
CHỨC NĂNG & NHIỆM VỤ CỦA FILE:
    - Định nghĩa lớp cơ sở trừu tượng BaseDetector (Abstract Base Class) cho tất cả các mô hình 
      phát hiện ảnh AI trong nhánh huấn luyện ProGAN.
    - Cấu hình các thông số chuẩn hóa mặc định (chuẩn ImageNet: mean, std).
    - Thiết lập độ phân giải chuẩn (loadSize=256, cropSize=224).
    - Cung cấp hàm _build_transforms xây dựng pipeline tiền xử lý ảnh (Data Augmentation, Resize, Crop).
    - Cung cấp phương thức predict() tính toán xác suất dự đoán AI.
    - Quản lý việc lưu (save_weights) và tải (load_weights) trọng số mô hình.

DANH SÁCH CÁC PHƯƠNG THỨC / HÀM TRONG FILE:
    1. BaseDetector.__init__(self): Khởi tạo các siêu tham số kích thước, chuẩn hóa và augmentation.
    2. BaseDetector.aug_config(self): Thuộc tính (property) tổng hợp cấu hình Data Augmentation.
    3. BaseDetector._build_transforms(self): Xây dựng 3 pipeline tiền xử lý: train_transform, val_transform, test_transform.
    4. BaseDetector.forward(self, x, return_feat): Phương thức trừu tượng lan truyền xuôi, các lớp con bắt buộc phải cài đặt.
    5. BaseDetector.predict(self, inputs): Chuyển đổi tensor đầu vào lên thiết bị, tính forward pass và trả về xác suất sau Sigmoid.
    6. BaseDetector.save_weights(self, weights_path): Chuyển trọng số state_dict về CPU và lưu vào file .pth.
    7. BaseDetector.load_weights(self, weights_path): Nạp trọng số từ file .pth vào mô hình.
=============================================================================
"""

import torch
from abc import ABC, abstractmethod
from torchvision import transforms
from utils import data_augment


# =============================================================================
# CLASS: BaseDetector
# =============================================================================
class BaseDetector(torch.nn.Module, ABC):
    """Lớp trừu tượng cơ sở cho toàn bộ các detector thuộc nhánh ProGAN."""

    # -------------------------------------------------------------------------
    # CHỨC NĂNG:
    #     Khởi tạo các tham số tiền xử lý chuẩn: giá trị chuẩn hóa ImageNet, 
    #     kích thước ảnh loadSize (256) và cropSize (224), cùng các xác suất augmentation.
    # INPUT:
    #     - Không có tham số đầu vào.
    # OUTPUT:
    #     - Đối tượng BaseDetector.
    # -------------------------------------------------------------------------
    def __init__(self):
        super(BaseDetector, self).__init__()

        # Default normalization (ImageNet)
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        # Default resolution
        self.loadSize = 256
        self.cropSize = 224

        # Default data augmentation config
        self.blur_prob = 0.0
        self.blur_sig = [0.0, 3.0]
        self.jpg_prob = 0.0
        self.jpg_method = ['cv2', 'pil']
        self.jpg_qual = list(range(30, 101))

    # -------------------------------------------------------------------------
    # CHỨC NĂNG:
    #     Property đóng gói toàn bộ thông số tăng cường dữ liệu thành dictionary.
    # INPUT:
    #     - Không có tham số ngoài self.
    # OUTPUT:
    #     - (dict): Chứa blur_prob, blur_sig, jpg_prob, jpg_method, jpg_qual.
    # -------------------------------------------------------------------------
    @property
    def aug_config(self):
        return {
            "blur_prob": self.blur_prob,
            "blur_sig": self.blur_sig,
            "jpg_prob": self.jpg_prob,
            "jpg_method": self.jpg_method,
            "jpg_qual": self.jpg_qual,
        }

    # -------------------------------------------------------------------------
    # CHỨC NĂNG:
    #     Xây dựng 3 bộ transforms (train_transform, val_transform, test_transform) 
    #     sử dụng torchvision transforms kết hợp augmentation tùy biến.
    # INPUT:
    #     - Không có tham số ngoài self.
    # OUTPUT:
    #     - Không trả về giá trị (gán trực tiếp vào các thuộc tính transform).
    # -------------------------------------------------------------------------
    def _build_transforms(self):
        random_crop = transforms.RandomCrop(self.cropSize)
        center_crop = transforms.CenterCrop(self.cropSize)
        flip_func = transforms.RandomHorizontalFlip()
        rz_func = transforms.Resize(self.loadSize)
        aug_func = transforms.Lambda(lambda x: data_augment(x, self.aug_config))

        self.train_transform = transforms.Compose([
            rz_func,
            aug_func,
            random_crop,
            flip_func,
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
        ])

        self.val_transform = transforms.Compose([
            rz_func,
            random_crop,
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
        ])

        self.test_transform = transforms.Compose([
            rz_func,
            center_crop,
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
        ])

    # -------------------------------------------------------------------------
    # CHỨC NĂNG:
    #     Phương thức trừu tượng tính toán forward pass qua mạng.
    # INPUT:
    #     - x (torch.Tensor): Tensor ảnh đầu vào.
    #     - return_feat (bool): Tùy chọn trả về đặc trưng hay không.
    # OUTPUT:
    #     - torch.Tensor logits hoặc tuple (đặc trưng, logits).
    # -------------------------------------------------------------------------
    @abstractmethod
    def forward(self, x, return_feat=False):
        pass

    # -------------------------------------------------------------------------
    # CHỨC NĂNG:
    #     Thực hiện suy luận dự đoán xác suất mẫu ảnh là do AI sinh ra.
    # INPUT:
    #     - inputs (torch.Tensor): Batch ảnh đầu vào (B, C, H, W).
    # OUTPUT:
    #     - prediction (list[float]): Danh sách xác suất dự đoán (trong khoảng [0, 1]).
    # -------------------------------------------------------------------------
    def predict(self, inputs):
        inputs = inputs.to(next(self.parameters()).device)
        outputs = self.forward(inputs)
        prediction = outputs.sigmoid().flatten().tolist()
        return prediction

    # -------------------------------------------------------------------------
    # CHỨC NĂNG:
    #     Lưu trọng số state_dict của mô hình ra file checkpoint (.pth).
    # INPUT:
    #     - weights_path (str): Đường dẫn file lưu trữ.
    # OUTPUT:
    #     - Không trả về giá trị (None).
    # -------------------------------------------------------------------------
    def save_weights(self, weights_path):
        save_params = {k: v.cpu() for k, v in self.state_dict().items()}
        torch.save(save_params, weights_path)

    # -------------------------------------------------------------------------
    # CHỨC NĂNG:
    #     Nạp trọng số từ file .pth vào mô hình hiện tại.
    # INPUT:
    #     - weights_path (str): Đường dẫn file checkpoint.
    # OUTPUT:
    #     - Không trả về giá trị (None).
    # -------------------------------------------------------------------------
    def load_weights(self, weights_path):
        weights = torch.load(weights_path, map_location='cpu')
        self.load_state_dict(weights, strict=False)
