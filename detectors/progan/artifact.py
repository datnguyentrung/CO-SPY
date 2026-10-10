"""
=============================================================================
FILE: detectors/progan/artifact.py
CHỨC NĂNG & NHIỆM VỤ CỦA FILE:
    - Định nghĩa lớp ArtifactDetector và kiến trúc mạng ResNet trích xuất đặc trưng dấu vết nội suy 
      (dựa trên phương pháp NPR - Neighboring Pixel Relationships) cho ProGAN.
    - Tính toán dư sai nội suy (artifact = x - interpolate(x, 0.5)) để làm nổi bật các dấu vết tần số cao
      do thuật toán nội suy/upsampling của mạng GAN gây ra.
    - Cung cấp các khối cơ bản tích chập: conv1x1, conv3x3, khối Bottleneck residual và mạng ResNet trích xuất đặc trưng.

DANH SÁCH CÁC PHƯƠNG THỨC / HÀM TRONG FILE:
    1. ArtifactDetector.__init__(self, dim_artifact, num_classes): Khởi tạo encoder ResNet trích xuất đặc trưng lỗi GAN và lớp fc.
    2. ArtifactDetector.forward(self, x, return_feat): Lan truyền xuôi qua artifact_encoder và lớp fc.
    3. conv1x1(in_planes, out_planes, stride): Lớp tích chập 2D kích thước kernel 1x1.
    4. conv3x3(in_planes, out_planes, stride): Lớp tích chập 2D kích thước kernel 3x3 kèm padding 1.
    5. Bottleneck.__init__(self, inplanes, planes, stride, downsample): Khởi tạo khối thắt cổ chai ResNet.
    6. Bottleneck.forward(self, x): Lan truyền xuôi qua khối Bottleneck có kết nối tắt residual.
    7. ResNet.__init__(self, block, layers, num_classes): Khởi tạo mạng ResNet trích xuất đặc trưng NPR.
    8. ResNet._make_layer(self, block, planes, blocks, stride): Tạo chuỗi khối Bottleneck liên tiếp.
    9. ResNet.interpolate(self, img, factor): Phóng to thu nhỏ ảnh bằng Nearest Interpolation để mô phỏng tái lập ảnh.
    10. ResNet.forward(self, x): Tính toán sai lệch nội suy x - interpolate(x, 0.5) và trích xuất vector đặc trưng.
=============================================================================
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
from .base import BaseDetector


# =============================================================================
# CLASS: ArtifactDetector
# =============================================================================
class ArtifactDetector(BaseDetector):
    """Bộ phát hiện dấu vết nhân tạo (Artifact) trên ảnh ProGAN dựa trên kiến trúc ResNet NPR."""

    # -------------------------------------------------------------------------
    # CHỨC NĂNG:
    #     Khởi tạo ArtifactDetector: tạo artifact_encoder (ResNet 4 tầng), 
    #     lớp phân loại tuyến tính fc và xây dựng pipeline tiền xử lý ảnh.
    # INPUT:
    #     - dim_artifact (int): Số chiều vector đặc trưng artifact (mặc định 512).
    #     - num_classes (int): Số nhãn phân loại đầu ra (mặc định 1).
    # OUTPUT:
    #     - Đối tượng ArtifactDetector.
    # -------------------------------------------------------------------------
    def __init__(self, dim_artifact=512, num_classes=1):
        super(ArtifactDetector, self).__init__()

        # Load the artifact encoder based on NPR
        self.artifact_encoder = ResNet(Bottleneck, [3, 4, 6, 3])

        # Classifier
        self.fc = torch.nn.Linear(dim_artifact, num_classes)

        # Build transforms
        self._build_transforms()

    # -------------------------------------------------------------------------
    # CHỨC NĂNG:
    #     Lan truyền xuôi qua artifact_encoder và lớp phân loại fc.
    # INPUT:
    #     - x (torch.Tensor): Tensor ảnh đầu vào (B, 3, H, W).
    #     - return_feat (bool): Tùy chọn trả về đặc trưng hay không.
    # OUTPUT:
    #     - out (torch.Tensor): Logits phân loại.
    #     - (feat, out) nếu return_feat=True.
    # -------------------------------------------------------------------------
    def forward(self, x, return_feat=False):
        feat = self.artifact_encoder(x)
        out = self.fc(feat)
        if return_feat:
            return feat, out
        return out


# -------------------------------------------------------------------------
# CHỨC NĂNG:
#     Tạo một lớp tích chập Conv2d kích thước 1x1 không có bias.
# INPUT:
#     - in_planes (int): Số kênh đầu vào.
#     - out_planes (int): Số kênh đầu ra.
#     - stride (int): Bước trượt của kernel (mặc định 1).
# OUTPUT:
#     - nn.Conv2d: Lớp tích chập 1x1.
# -------------------------------------------------------------------------
def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


# -------------------------------------------------------------------------
# CHỨC NĂNG:
#     Tạo một lớp tích chập Conv2d kích thước 3x3 kèm padding=1 không có bias.
# INPUT:
#     - in_planes (int): Số kênh đầu vào.
#     - out_planes (int): Số kênh đầu ra.
#     - stride (int): Bước trượt của kernel (mặc định 1).
# OUTPUT:
#     - nn.Conv2d: Lớp tích chập 3x3.
# -------------------------------------------------------------------------
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


# =============================================================================
# CLASS: Bottleneck
# =============================================================================
class Bottleneck(nn.Module):
    """Khối Bottleneck chuẩn với hệ số mở rộng kênh expansion=4."""
    expansion = 4

    # -------------------------------------------------------------------------
    # CHỨC NĂNG:
    #     Khởi tạo 3 lớp tích chập liên tiếp (1x1 -> 3x3 -> 1x1) cùng BatchNorm và ReLU.
    # INPUT:
    #     - inplanes (int): Số kênh đầu vào.
    #     - planes (int): Số kênh trung gian.
    #     - stride (int): Bước trượt của lớp conv 3x3.
    #     - downsample (nn.Module): Nhánh điều chỉnh kích thước nếu cần.
    # OUTPUT:
    #     - Đối tượng Bottleneck.
    # -------------------------------------------------------------------------
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
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
    # CHỨC NĂNG:
    #     Lan truyền xuôi qua khối Bottleneck, cộng nhánh residual shortcut và kích hoạt ReLU.
    # INPUT:
    #     - x (torch.Tensor): Feature map đầu vào.
    # OUTPUT:
    #     - out (torch.Tensor): Feature map đầu ra sau khi kết hợp residual.
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


# =============================================================================
# CLASS: ResNet
# =============================================================================
class ResNet(nn.Module):
    """Mạng ResNet chuyên biệt trích xuất đặc trưng dư sai nội suy điểm ảnh lân cận (NPR)."""

    # -------------------------------------------------------------------------
    # CHỨC NĂNG:
    #     Khởi tạo cấu trúc mạng ResNet với các khối tích chập ban đầu và các tầng residual layer.
    # INPUT:
    #     - block (Type[nn.Module]): Kiểu khối cơ bản (Bottleneck).
    #     - layers (list[int]): Số lượng block trong mỗi tầng [3, 4, 6, 3].
    #     - num_classes (int): Số lớp phân loại.
    # OUTPUT:
    #     - Đối tượng ResNet.
    # -------------------------------------------------------------------------
    def __init__(self, block, layers, num_classes=1):
        super(ResNet, self).__init__()

        self.unfoldSize = 2
        self.unfoldIndex = 0
        assert self.unfoldSize > 1
        assert -1 < self.unfoldIndex and self.unfoldIndex < self.unfoldSize*self.unfoldSize
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64 , layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(512, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    # -------------------------------------------------------------------------
    # CHỨC NĂNG:
    #     Xây dựng một tầng bao gồm nhiều khối Bottleneck nối tiếp nhau.
    # INPUT:
    #     - block (nn.Module): Lớp khối (Bottleneck).
    #     - planes (int): Số kênh cơ sở.
    #     - blocks (int): Số khối cần tạo.
    #     - stride (int): Bước trượt.
    # OUTPUT:
    #     - nn.Sequential: Tầng chứa chuỗi các block.
    # -------------------------------------------------------------------------
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
    # CHỨC NĂNG:
    #     Thực hiện nội suy giảm kích thước theo hệ số factor rồi phóng to lại kích thước gốc 
    #     bằng Nearest Neighbor để mô phỏng tái lập ảnh và phát hiện sai số điểm ảnh.
    # INPUT:
    #     - img (torch.Tensor): Tensor ảnh đầu vào (B, C, H, W).
    #     - factor (float): Hệ số co giãn nội suy (vd: 0.5).
    # OUTPUT:
    #     - (torch.Tensor): Tensor ảnh sau khi co giãn rồi phóng to lại.
    # -------------------------------------------------------------------------
    def interpolate(self, img, factor):
        return F.interpolate(
            F.interpolate(img,
                          scale_factor=factor,
                          mode='nearest',
                          recompute_scale_factor=True),
            scale_factor=1 / factor,
            mode='nearest',
            recompute_scale_factor=True)

    # -------------------------------------------------------------------------
    # CHỨC NĂNG:
    #     Trừ ảnh gốc với ảnh nội suy để lấy bản đồ dấu vết tần số cao (artifact = x - interpolate(x, 0.5)),
    #     sau đó truyền qua các tầng tích chập và pooling để xuất vector đặc trưng 512 chiều.
    # INPUT:
    #     - x (torch.Tensor): Tensor ảnh đầu vào (B, 3, H, W).
    # OUTPUT:
    #     - (torch.Tensor): Vector đặc trưng phẳng (B, 512).
    # -------------------------------------------------------------------------
    def forward(self, x):
        artifact = x - self.interpolate(x, 0.5)

        x = self.conv1(artifact * 2.0 / 3.0)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        return x
