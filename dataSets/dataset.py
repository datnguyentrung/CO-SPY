"""
=============================================================================
FILE: dataSets/dataset.py
CHỨC NĂNG & NHIỆM VỤ CỦA FILE:
    - Định nghĩa các lớp Dataset kế thừa từ torch.utils.data.Dataset phục vụ cho
      việc nạp và tiền xử lý dữ liệu huấn luyện (training) và kiểm thử (testing):
        1. TrainDataset: Nạp dữ liệu huấn luyện ảnh thật và ảnh giả AI, gán nhãn,
           hỗ trợ mô phỏng nén JPEG và xáo trộn ngẫu nhiên.
        2. CoSpyBenchTestDataset: Bộ nạp dữ liệu kiểm thử cân bằng cho chuẩn Co-Spy-Bench,
           ghép cặp ảnh thật từ nhiều nguồn (MS-COCO, Flickr, ...) và ảnh giả từ 22 mô hình AI.
        3. AIGCDetectTestDataset: Bộ nạp dữ liệu kiểm thử theo chuẩn benchmark AIGCDetectionBenchMark.
        4. FusionTrainDataset: Bộ nạp chuyên biệt cho chế độ huấn luyện Fusion, áp dụng song song
           hai pipeline biến đổi ảnh khác nhau (sem_transform và art_transform) lên cùng một ảnh gốc.

DANH SÁCH CÁC PHƯƠNG THỨC / HÀM TRONG FILE:
    1. TrainDataset.__init__(self, train_dataset, split, add_jpeg, transform): Khởi tạo danh sách ảnh và nhãn (0/1).
    2. TrainDataset.__len__(self): Trả về tổng số lượng mẫu ảnh trong tập dữ liệu.
    3. TrainDataset.__getitem__(self, idx): Đọc ảnh RGB theo chỉ số, áp dụng tiền xử lý và trả về (image, label).
    4. CoSpyBenchTestDataset.__init__(self, dataset, model, num_real, add_jpeg, transform): Khởi tạo tập kiểm thử Co-Spy-Bench.
    5. CoSpyBenchTestDataset.__len__(self): Trả về tổng số ảnh kiểm thử (gồm cả ảnh thật và giả).
    6. CoSpyBenchTestDataset.__getitem__(self, idx): Lấy ảnh thật hoặc ảnh giả theo chỉ số và trả về (image, label).
    7. AIGCDetectTestDataset.__init__(self, dataset, model, transform): Khởi tạo tập kiểm thử AIGCDetectionBenchMark.
    8. AIGCDetectTestDataset.__len__(self): Trả về kích thước tập kiểm thử AIGCDetectionBenchMark.
    9. AIGCDetectTestDataset.__getitem__(self, idx): Lấy ảnh và nhãn tương ứng.
    10. FusionTrainDataset.__init__(self, base, sem_transform, art_transform): Khởi tạo tập dữ liệu cho Fusion branch.
    11. FusionTrainDataset.__len__(self): Trả về số lượng mẫu ảnh từ tập base dataset.
    12. FusionTrainDataset.__getitem__(self, i): Đọc ảnh và trả về (sem_image, art_image, label).
=============================================================================
"""

import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from utils import get_list, png_to_jpeg

from .cospy_real import MSCOCO2017, Flickr30k, OtherReal


# =============================================================================
# CLASS: TrainDataset
# =============================================================================
class TrainDataset(Dataset):
    """Dataset nạp dữ liệu ảnh Thật (nhãn 0) và Giả (nhãn 1) cho quá trình huấn luyện và validation."""

    # -------------------------------------------------------------------------
    # CHỨC NĂNG:
    #     Khởi tạo danh sách đường dẫn ảnh Thật và Giả từ đĩa theo cấu trúc thư mục
    #     của progan hoặc sd-v1_4, gán nhãn nhị phân tương ứng và xáo trộn ngẫu nhiên.
    # INPUT:
    #     - train_dataset (str): Tên bộ dữ liệu huấn luyện ('progan' hoặc 'sd-v1_4').
    #     - split (str): Phân đoạn dữ liệu ('train' hoặc 'val'). Mặc định 'train'.
    #     - add_jpeg (bool): Có áp dụng nén JPEG (chất lượng 95) hay không. Mặc định False.
    #     - transform (callable): Pipeline biến đổi hình ảnh (torchvision transforms).
    # OUTPUT:
    #     - Đối tượng TrainDataset.
    # -------------------------------------------------------------------------
    def __init__(
        self,
        train_dataset,
        split="train",
        add_jpeg=False,
        transform=None,
        seed=None,
    ):
        assert split in ["train", "val"]
        # Thư mục gốc chứa dữ liệu huấn luyện
        root_dir = f"data/train/{train_dataset}"

        # Đọc danh sách file ảnh theo loại dataset
        if train_dataset == "progan":
            real_list = get_list(os.path.join(root_dir, split), must_contain="0_real")
            fake_list = get_list(os.path.join(root_dir, split), must_contain="1_fake")
        elif train_dataset in ["sd-v1_4", "sd_v1_4"]:
            real_list = get_list(os.path.join(root_dir, "mscoco2017", f"{split}2017"))
            fake_list = get_list(
                os.path.join(root_dir, "stable-diffusion-v1-4", f"{split}2017")
            )

        # Thiết lập nhãn: 0 là ảnh Thật (Real), 1 là ảnh Giả AI (Fake)
        self.labels_dict = {}
        for i in real_list:
            self.labels_dict[i] = 0
        for i in fake_list:
            self.labels_dict[i] = 1

        # Gộp toàn bộ danh sách và xáo trộn ngẫu nhiên. Khi training truyền
        # seed, thứ tự path được ổn định để sampler checkpoint có cùng ý nghĩa
        # giữa Kaggle, Colab và local.
        self.total_list = real_list + fake_list
        if seed is None:
            np.random.shuffle(self.total_list)
        else:
            self.total_list.sort()
            np.random.RandomState(int(seed)).shuffle(self.total_list)

        self.add_jpeg = add_jpeg
        self.transform = transform

    # -------------------------------------------------------------------------
    # CHỨC NĂNG:
    #     Trả về tổng số lượng mẫu ảnh trong toàn bộ dataset.
    # INPUT:
    #     - Không có tham số ngoài self.
    # OUTPUT:
    #     - (int): Tổng số ảnh trong tập dữ liệu.
    # -------------------------------------------------------------------------
    def __len__(self):
        return len(self.total_list)

    # -------------------------------------------------------------------------
    # CHỨC NĂNG:
    #     Đọc ảnh từ file theo chỉ số idx, chuyển hệ màu RGB, áp dụng nén JPEG giả lập
    #     (nếu cấu hình) và thực hiện các phép biến đổi transform.
    # INPUT:
    #     - idx (int): Chỉ số phần tử trong danh sách mẫu dữ liệu.
    # OUTPUT:
    #     - image (PIL.Image hoặc torch.Tensor): Dữ liệu ảnh sau tiền xử lý.
    #     - label (int): Nhãn nhị phân (0: Real, 1: Fake).
    # -------------------------------------------------------------------------
    def __getitem__(self, idx):
        img_path = self.total_list[idx]
        label = self.labels_dict[img_path]
        image = Image.open(img_path).convert("RGB")

        # Áp dụng nén JPEG nếu có
        if self.add_jpeg:
            image = png_to_jpeg(image, quality=95)

        # Áp dụng các phép biến đổi ảnh
        if self.transform is not None:
            image = self.transform(image)
        return image, label


# =============================================================================
# CLASS: CoSpyBenchTestDataset
# =============================================================================
class CoSpyBenchTestDataset(Dataset):
    """Dataset kiểm thử chuẩn Co-Spy-Bench cân bằng tỷ lệ ảnh thật và ảnh giả AI."""

    # -------------------------------------------------------------------------
    # CHỨC NĂNG:
    #     Khởi tạo tập kiểm thử cân bằng số lượng giữa ảnh Thật (MSCOCO/Flickr)
    #     và ảnh Giả (sinh từ mô hình AI cụ thể).
    # INPUT:
    #     - dataset (str): Tên tập dữ liệu gốc ('mscoco', 'flickr', ...).
    #     - model (str): Tên mô hình sinh ảnh AI cần kiểm thử (vd: 'sd-v1_4', 'midjourney', ...).
    #     - num_real (int): Số lượng ảnh thật tối đa lấy vào tập test. Mặc định 2000.
    #     - add_jpeg (bool): Áp dụng nén JPEG chuẩn 95 hay không. Mặc định True.
    #     - transform (callable): Pipeline biến đổi hình ảnh.
    # OUTPUT:
    #     - Đối tượng CoSpyBenchTestDataset.
    # -------------------------------------------------------------------------
    def __init__(self, dataset, model, num_real=2000, add_jpeg=True, transform=None):
        root_path = "data/test/Co-Spy-Bench/synthetic"

        # Đọc danh sách ảnh sinh từ model AI tương ứng
        fake_dir = os.path.join(root_path, dataset, model)
        fake_list = [i for i in os.listdir(fake_dir) if i.endswith(".png")]
        fake_list.sort()
        self.fake = [os.path.join(fake_dir, i) for i in fake_list]

        # Đọc ảnh thật tương ứng với tập dữ liệu
        if dataset == "mscoco":
            self.real = MSCOCO2017()
        elif dataset == "flickr":
            self.real = Flickr30k()
        else:
            self.real = OtherReal(dataset)

        # Đảm bảo số lượng ảnh thật và ảnh giả bằng nhau để tránh mất cân bằng dữ liệu kiểm thử
        self.num_real = min(num_real, len(self.real), len(self.fake))
        self.image_idx = list(range(self.num_real * 2))
        self.labels = [0] * self.num_real + [1] * self.num_real

        self.add_jpeg = add_jpeg
        self.transform = transform

    # -------------------------------------------------------------------------
    # CHỨC NĂNG:
    #     Trả về tổng số mẫu ảnh trong tập kiểm thử (bằng 2 * num_real).
    # INPUT:
    #     - Không có tham số ngoài self.
    # OUTPUT:
    #     - (int): Tổng số lượng ảnh test.
    # -------------------------------------------------------------------------
    def __len__(self):
        return len(self.image_idx)

    # -------------------------------------------------------------------------
    # CHỨC NĂNG:
    #     Lấy ảnh thật (nếu idx < num_real) hoặc ảnh giả AI (nếu idx >= num_real),
    #     áp dụng nén JPEG và pipeline biến đổi ảnh.
    # INPUT:
    #     - idx (int): Chỉ số ảnh cần lấy.
    # OUTPUT:
    #     - image (PIL.Image hoặc torch.Tensor): Ảnh kiểm thử sau biến đổi.
    #     - label (int): Nhãn ảnh (0: Real, 1: AI-Generated).
    # -------------------------------------------------------------------------
    def __getitem__(self, idx):
        if idx < self.num_real:
            image, _ = self.real[idx]
        else:
            image = Image.open(self.fake[idx - self.num_real]).convert("RGB")

        if self.add_jpeg:
            image = png_to_jpeg(image, quality=95)

        if self.transform is not None:
            image = self.transform(image)
        label = self.labels[idx]

        return image, label


# =============================================================================
# CLASS: AIGCDetectTestDataset
# =============================================================================
class AIGCDetectTestDataset(Dataset):
    """Dataset kiểm thử chuẩn AIGCDetectionBenchMark phục vụ mô hình ProGAN."""

    # -------------------------------------------------------------------------
    # CHỨC NĂNG:
    #     Khởi tạo dataset kiểm thử từ thư mục AIGCDetectionBenchMark với các ảnh '0_real' và '1_fake'.
    # INPUT:
    #     - dataset (str): Tên tập dữ liệu kiểm thử.
    #     - model (str): Tên mô hình sinh ảnh AI tương ứng.
    #     - transform (callable): Pipeline biến đổi hình ảnh.
    # OUTPUT:
    #     - Đối tượng AIGCDetectTestDataset.
    # -------------------------------------------------------------------------
    def __init__(self, dataset, model, transform=None):
        root_path = "data/test/AIGCDetectionBenchMark"

        image_dir = os.path.join(root_path, dataset, model)
        real_list = get_list(image_dir, must_contain="0_real")
        fake_list = get_list(image_dir, must_contain="1_fake")

        self.labels_dict = {}
        for i in real_list:
            self.labels_dict[i] = 0
        for i in fake_list:
            self.labels_dict[i] = 1

        self.total_list = real_list + fake_list
        np.random.shuffle(self.total_list)

        self.transform = transform

    # -------------------------------------------------------------------------
    # CHỨC NĂNG:
    #     Trả về kích thước tập kiểm thử.
    # INPUT:
    #     - Không có tham số ngoài self.
    # OUTPUT:
    #     - (int): Tổng số ảnh test trong benchmark.
    # -------------------------------------------------------------------------
    def __len__(self):
        return len(self.total_list)

    # -------------------------------------------------------------------------
    # CHỨC NĂNG:
    #     Đọc ảnh RGB theo đường dẫn và biến đổi qua pipeline transform.
    # INPUT:
    #     - idx (int): Chỉ số ảnh cần lấy.
    # OUTPUT:
    #     - image (PIL.Image hoặc torch.Tensor): Ảnh sau biến đổi.
    #     - label (int): Nhãn nhị phân (0 hoặc 1).
    # -------------------------------------------------------------------------
    def __getitem__(self, idx):
        img_path = self.total_list[idx]
        label = self.labels_dict[img_path]
        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label


# =============================================================================
# CLASS: FusionTrainDataset
# =============================================================================
class FusionTrainDataset(Dataset):
    """Dataset chuyên biệt cho giai đoạn Late Fusion, sinh ra 2 phiên bản biến đổi ảnh cho 2 nhánh."""

    # -------------------------------------------------------------------------
    # CHỨC NĂNG:
    #     Đóng gói một TrainDataset gốc cùng với hai pipeline tiền xử lý ảnh riêng biệt:
    #     một dành cho nhánh ngữ nghĩa (Semantic) và một dành cho nhánh dấu vết tái tạo (Artifact).
    # INPUT:
    #     - base (TrainDataset): Dataset gốc chứa danh sách đường dẫn ảnh và nhãn.
    #     - sem_transform (callable): Transform cho nhánh Semantic (thường là CLIP preprocess).
    #     - art_transform (callable): Transform cho nhánh Artifact (thường là VAE crop/resize).
    # OUTPUT:
    #     - Đối tượng FusionTrainDataset.
    # -------------------------------------------------------------------------
    def __init__(
        self,
        base: TrainDataset,
        sem_transform,
        art_transform,
        npr_transform=None,
    ):
        self.base = base
        self.sem_transform = sem_transform
        self.art_transform = art_transform
        self.npr_transform = npr_transform

    # -------------------------------------------------------------------------
    # CHỨC NĂNG:
    #     Trả về tổng số mẫu của tập dữ liệu huấn luyện fusion.
    # INPUT:
    #     - Không có tham số ngoài self.
    # OUTPUT:
    #     - (int): Tổng số lượng mẫu ảnh.
    # -------------------------------------------------------------------------
    def __len__(self):
        return len(self.base.total_list)

    # -------------------------------------------------------------------------
    # CHỨC NĂNG:
    #     Đọc 1 ảnh gốc và áp dụng đồng thời cả 2 bộ transform riêng biệt:
    #     sem_transform và art_transform.
    # INPUT:
    #     - i (int): Chỉ số ảnh trong dataset.
    # OUTPUT:
    #     - sem_image (torch.Tensor): Ảnh qua pipeline của nhánh Semantic.
    #     - art_image (torch.Tensor): Ảnh qua pipeline của nhánh Artifact.
    #     - label (int): Nhãn nhị phân (0: Real, 1: Fake).
    # -------------------------------------------------------------------------
    def __getitem__(self, i):
        img_path = self.base.total_list[i]
        label = self.base.labels_dict[img_path]
        img = Image.open(img_path).convert("RGB")
        sem_image = self.sem_transform(img)
        art_image = self.art_transform(img)
        if self.npr_transform is None:
            return sem_image, art_image, label
        return sem_image, art_image, self.npr_transform(img), label


class FusionEvalDataset(Dataset):
    """Apply semantic and multi-view artifact transforms above a raw test set."""

    def __init__(self, base, sem_transform, artifact_view_transform):
        self.base = base
        self.sem_transform = sem_transform
        self.artifact_view_transform = artifact_view_transform

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        image, label = self.base[index]
        if not isinstance(image, Image.Image):
            image = transforms.ToPILImage()(image)
        return (
            self.sem_transform(image),
            self.artifact_view_transform(image),
            int(label),
        )


class FusionFeatureDataset(Dataset):
    """Load cached branch features from numpy arrays."""

    def __init__(self, sem_features, art_features, labels, npr_features=None):
        self.sem_features = sem_features
        self.art_features = art_features
        self.labels = labels
        self.npr_features = npr_features

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        sem = torch.from_numpy(np.asarray(self.sem_features[index]))
        art = torch.from_numpy(np.asarray(self.art_features[index]))
        label = int(self.labels[index])
        if self.npr_features is None:
            return sem, art, label
        npr = torch.from_numpy(np.asarray(self.npr_features[index]))
        return sem, art, npr, label
