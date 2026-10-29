"""
=============================================================================
FILE: detectors/sd-v1_4/artifact.py
CHỨC NĂNG & NHIỆM VỤ CỦA FILE:
    - Cài đặt bộ phát hiện nhánh Khiếm khuyết điểm ảnh (Artifact Detector).
    - Khai thác đặc trưng tần số cao và sai số tái tạo của Autoencoder:
        + Đưa ảnh x qua Autoencoder VAE của Stable Diffusion để nén xuống không gian ẩn (latent)
          rồi giải mã tái tạo lại ảnh x_recon.
        + Tính sai số tái tạo: residual = x - x_recon. Ảnh AI và ảnh thật có phân bố nhiễu tái tạo
          rất khác nhau.
        + Đưa sai số này qua một mạng ResNet rút gọn (VAEReconEncoder) để trích xuất vector đặc trưng 512 chiều.
        + Phân loại bằng lớp Linear (fc).

DANH SÁCH CÁC PHƯƠNG THỨC / HÀM TRONG FILE:
    1. ArtifactDetector.__init__(dim_artifact, num_classes): Khởi tạo VAE, VAEReconEncoder, lớp Linear fc.
    2. ArtifactDetector._build_transforms(): Xây dựng pipeline tiền xử lý ảnh cho nhánh artifact.
    3. ArtifactDetector.forward(x, return_feat): Lan truyền xuôi trích xuất đặc trưng artifact và phân loại.
    4. conv3x3(in_planes, out_planes, stride): Hàm tiện ích tạo lớp Conv2D 3x3 kèm padding=1.
    5. conv1x1(in_planes, out_planes, stride): Hàm tiện ích tạo lớp Conv2D 1x1.
    6. Bottleneck.forward(x): Khối tích chập Bottleneck của ResNet.
    7. VAEReconEncoder.__init__(vae, block): Khởi tạo mạng ResNet rút gọn gắn kèm VAE.
    8. VAEReconEncoder.reconstruct(x): Nén và giải mã tái tạo ảnh qua VAE không tính gradient.
    9. VAEReconEncoder.forward(x): Tính sai số (x - x_recon), khuếch đại tín hiệu và trích xuất vector 512 chiều.
=============================================================================
"""

import math
import random

import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
from torch import nn
from torchvision import transforms

from utils import data_augment

from .base import BaseDetector


class AlignedCrop:
    """Crop ảnh ở độ phân giải gốc, căn tọa độ crop theo bội số 8."""

    def __init__(self, size=224, train=True):
        self.size = size
        self.train = train

    def __call__(self, img):
        w, h = img.size

        # Nếu ảnh quá nhỏ thì phóng lên vừa đủ để crop
        if min(w, h) < self.size:
            scale = self.size / min(w, h)
            new_w = math.ceil(w * scale)
            new_h = math.ceil(h * scale)

            resampling = getattr(Image, "Resampling", Image)
            img = img.resize(
                (new_w, new_h),
                resample=resampling.BICUBIC,
            )
            w, h = img.size

        max_x = w - self.size
        max_y = h - self.size

        if self.train:
            # Chọn tọa độ ngẫu nhiên là bội số của 8
            x = random.randrange(0, max_x // 8 + 1) * 8
            y = random.randrange(0, max_y // 8 + 1) * 8
        else:
            # Crop giữa ảnh, sau đó căn lại theo bội số 8
            x = (max_x // 2) // 8 * 8
            y = (max_y // 2) // 8 * 8

        return img.crop((x, y, x + self.size, y + self.size))


class ArtifactDetector(BaseDetector):
    """
    Bộ phát hiện khiếm khuyết điểm ảnh (Artifact Detector):
    Cơ chế hoạt động cốt lõi:
    1. Đưa ảnh x qua VAE của Stable Diffusion để nén xuống latent rồi giải mã tái tạo lại x_recon.
    2. Tính sai số tái tạo: residual = x - x_recon (Ảnh do AI sinh sẽ có dấu vết phân bố nhiễu khác ảnh thật).
    3. Đưa phần sai số này qua một mạng ResNet rút gọn để trích xuất đặc trưng điểm ảnh (512 chiều).
    4. Phân loại bằng lớp Linear(512, 1).
    """

    # -------------------------------------------------------------------------
    # CHỨC NĂNG: Khởi tạo mô hình VAE, encoder sai số tái tạo và classifier fc
    # INPUT:
    #   - dim_artifact (int): Số chiều vector đặc trưng artifact (mặc định: 512).
    #   - num_classes (int): Số lớp phân loại (mặc định: 1 cho binary classification).
    # OUTPUT:
    #   - Đối tượng ArtifactDetector hoàn chỉnh.
    # -------------------------------------------------------------------------
    def __init__(self, dim_artifact=512, num_classes=1):
        super().__init__()

        # Tải mô hình VAE pretrained từ Stable Diffusion v1.4
        model_id = "CompVis/stable-diffusion-v1-4"
        vae = StableDiffusionPipeline.from_pretrained(model_id).vae
        vae.requires_grad_(False)  # Đóng băng VAE
        self.artifact_encoder = VAEReconEncoder(vae)

        # Lớp phân loại tuyến tính (Linear Classifier)
        self.fc = torch.nn.Linear(dim_artifact, num_classes)

        # Xây dựng các bước tiền xử lý ảnh
        self._build_transforms()

    # -------------------------------------------------------------------------
    # CHỨC NĂNG: Xây dựng pipeline tiền xử lý ảnh cho Artifact Detector
    # INPUT: Không có
    # OUTPUT: None (cập nhật self.train_transform, self.val_transform, self.test_transform)
    # -------------------------------------------------------------------------
    def _build_transforms(self):
        """Xây dựng pipeline tiền xử lý ảnh cho Artifact Detector."""
        # Mean/Std chuẩn cho nhánh Artifact
        self.mean = [0.0, 0.0, 0.0]
        self.std = [1.0, 1.0, 1.0]

        # Kích thước ảnh: load 256x256, crop 224x224
        self.loadSize = 256
        self.cropSize = 224

        # Cấu hình tăng cường dữ liệu: Nén JPEG ngẫu nhiên với xác suất 50%
        self.blur_prob = 0.0
        self.blur_sig = [0.0, 3.0]
        self.jpg_prob = 0.5
        self.jpg_method = ["cv2", "pil"]
        self.jpg_qual = list(range(70, 96))

        self.aug_config = {
            "blur_prob": self.blur_prob,
            "blur_sig": self.blur_sig,
            "jpg_prob": self.jpg_prob,
            "jpg_method": self.jpg_method,
            "jpg_qual": self.jpg_qual,
        }

        aligned_train_crop = AlignedCrop(224, train=True)
        aligned_eval_crop = AlignedCrop(224, train=False)

        flip_func = transforms.RandomHorizontalFlip()
        aug_func = transforms.Lambda(lambda x: data_augment(x, self.aug_config))

        # Pipeline cho tập Train
        self.train_transform = transforms.Compose(
            [
                aug_func,
                aligned_train_crop,
                flip_func,
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std),
            ]
        )

        # Pipeline cho tập Val
        self.val_transform = transforms.Compose(
            [
                aligned_eval_crop,
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std),
            ]
        )

        # Pipeline cho tập Test
        self.test_transform = transforms.Compose(
            [
                aligned_eval_crop,
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std),
            ]
        )

    # -------------------------------------------------------------------------
    # CHỨC NĂNG: Lan truyền xuôi trích xuất đặc trưng artifact và phân loại
    # INPUT:
    #   - x (torch.Tensor): Batch ảnh kích thước (B, 3, 224, 224).
    #   - return_feat (bool, optional): Có trả về vector đặc trưng artifact hay không.
    # OUTPUT:
    #   - out (torch.Tensor): Logit đầu ra kích thước (B, 1) (hoặc tuple (feat, out)).
    # -------------------------------------------------------------------------
    def forward(self, x, return_feat=False):
        """
        Lan truyền xuôi:
        1. Trích xuất đặc trưng artifact qua VAEReconEncoder.
        2. Đưa qua lớp Linear self.fc để tính điểm số đầu ra.
        """
        feat = self.artifact_encoder(x)
        out = self.fc(feat)
        if return_feat:
            return feat, out
        return out


# -----------------------------------------------------------------------------
# CÁC HÀM TRỢ GIÚP XÂY DỰNG KHỐI TÍCH CHẬP RESNET
# -----------------------------------------------------------------------------
# CHỨC NĂNG: Tạo lớp tích chập Conv2D 3x3 kèm padding=1
# INPUT: in_planes (int), out_planes (int), stride (int)
# OUTPUT: nn.Conv2d
# -----------------------------------------------------------------------------
def conv3x3(in_planes, out_planes, stride=1):
    """Tích chập 3x3 kèm padding=1 để giữ nguyên kích thước không gian."""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


# -----------------------------------------------------------------------------
# CHỨC NĂNG: Tạo lớp tích chập Conv2D 1x1
# INPUT: in_planes (int), out_planes (int), stride (int)
# OUTPUT: nn.Conv2d
# -----------------------------------------------------------------------------
def conv1x1(in_planes, out_planes, stride=1):
    """Tích chập 1x1 để thay đổi số kênh đặc trưng."""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    """Khối Bottleneck chuẩn của kiến trúc ResNet (1x1 conv -> 3x3 conv -> 1x1 conv + residual)."""

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    # -------------------------------------------------------------------------
    # CHỨC NĂNG: Lan truyền xuôi qua khối Bottleneck
    # INPUT: x (torch.Tensor) kích thước (B, C_in, H, W)
    # OUTPUT: Tensor sau tích chập + residual connection (B, C_out, H', W')
    # -------------------------------------------------------------------------
    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class VAEReconEncoder(nn.Module):
    """
    Bộ mã hóa sai số tái tạo VAE (VAE Reconstruction Error Encoder):
    Tính toán hiệu số giữa ảnh gốc và ảnh tái tạo từ VAE, sau đó đưa qua mạng ResNet rút gọn.
    """

    def __init__(self, vae, block=Bottleneck):
        super().__init__()

        # Khởi tạo các tầng đầu vào của ResNet
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Dùng 2 tầng đầu tiên của cấu hình ResNet-50
        self.layer1 = self._make_layer(block, 64, 3)
        self.layer2 = self._make_layer(block, 128, 4, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Khởi tạo trọng số Kaiming cho mạng tích chập
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Lưu đối tượng VAE
        self.vae = vae

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    # -------------------------------------------------------------------------
    # CHỨC NĂNG: Tái tạo ảnh thông qua không gian ẩn (latent) của VAE
    # INPUT: x (torch.Tensor): Ảnh đầu vào (B, 3, H, W)
    # OUTPUT: decoded (torch.Tensor): Ảnh được giải mã tái tạo từ VAE (B, 3, H, W)
    # -------------------------------------------------------------------------
    def reconstruct(self, x):
        """Tái tạo ảnh thông qua không gian ẩn (latent space) của VAE mà không tính đạo hàm."""
        with torch.no_grad():
            latent = self.vae.encode(x).latent_dist.mean
            decoded = self.vae.decode(latent).sample
        return decoded

    # -------------------------------------------------------------------------
    # CHỨC NĂNG: Lan truyền xuôi trích xuất đặc trưng khiếm khuyết điểm ảnh
    # INPUT: x (torch.Tensor): Batch ảnh đầu vào (B, 3, 224, 224)
    # OUTPUT: x (torch.Tensor): Vector đặc trưng artifact 512 chiều (B, 512)
    # -------------------------------------------------------------------------
    def forward(self, x):
        # 1. Tái tạo ảnh bằng VAE
        x_recon = self.reconstruct(x)
        # 2. Tính sai số tái tạo (khiếm khuyết mức pixel / artifact)
        x = x - x_recon

        # 3. Khuếch đại sai số để mạng dễ bắt được tín hiệu
        x = x / 7.0 * 100.0

        # 4. Trích xuất đặc trưng qua các lớp tích chập ResNet
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        return x
