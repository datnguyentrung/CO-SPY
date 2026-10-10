"""
=============================================================================
FILE: dataSets/cospy_real.py
CHỨC NĂNG & NHIỆM VỤ CỦA FILE:
    - Quản lý và cung cấp dữ liệu ảnh thật (Real Images) đi kèm chú thích văn bản (Captions/Prompts)
      phục vụ cho việc kiểm thử trên bộ Benchmark Co-Spy-Bench.
    - Hỗ trợ nạp dữ liệu từ 3 nhóm nguồn chính:
        1. MSCOCO2017: Dữ liệu ảnh thật từ tập MS COCO 2017 (val hoặc train) cùng file json chú thích.
        2. Flickr30k: Nạp trực tiếp dataset Flickr30k thông qua thư viện Hugging Face `datasets`.
        3. OtherReal: Các tập ảnh thật chuẩn khác như CC3M, SBU, TextCaps lưu trong thư mục benchmark.

DANH SÁCH CÁC PHƯƠNG THỨC / HÀM TRONG FILE:
    1. MSCOCO2017.__init__(self, split, transform): Khởi tạo danh sách file ảnh và file chú thích json của MS COCO.
    2. MSCOCO2017.__len__(self): Trả về tổng số ảnh trong tập MS COCO.
    3. MSCOCO2017.__getitem__(self, idx): Trả về ảnh đã qua transform và chuỗi chú thích tương ứng.
    4. Flickr30k.__init__(self, split, transform): Tải và nạp dataset Flickr30k từ Hugging Face.
    5. Flickr30k.__len__(self): Trả về số lượng mẫu trong Flickr30k.
    6. Flickr30k.__getitem__(self, idx): Lấy ảnh và chọn ngẫu nhiên 1 chú thích trong danh sách captions.
    7. OtherReal.__init__(self, dataset_name, transform): Khởi tạo dataset cho các tập CC3M, SBU, TextCaps.
    8. OtherReal.__len__(self): Trả về số lượng ảnh trong tập OtherReal tương ứng.
    9. OtherReal.__getitem__(self, idx): Đọc file ảnh và file json chú thích đi kèm.
=============================================================================
"""

import json
import os

import numpy as np
import torch
from PIL import Image

import datasets as ds


# =============================================================================
# CLASS: MSCOCO2017
# =============================================================================
class MSCOCO2017(torch.utils.data.Dataset):
    """Dataset quản lý ảnh thật từ tập MSCOCO 2017 (train hoặc val) kèm theo prompt chú thích (caption)."""

    # -------------------------------------------------------------------------
    # CHỨC NĂNG:
    #     Quét danh sách file ảnh .jpg trong thư mục MS COCO và nạp từ điển chú thích từ file json.
    # INPUT:
    #     - split (str): 'val' hoặc 'train'. Mặc định 'val'.
    #     - transform (callable): Pipeline biến đổi hình ảnh. Mặc định None.
    # OUTPUT:
    #     - Đối tượng MSCOCO2017.
    # -------------------------------------------------------------------------
    def __init__(self, split="val", transform=None):
        image_dir = f"data/train/sd-v1_4/mscoco2017/{split}2017"
        self.image_list = [
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
            if f.endswith(".jpg")
        ]
        self.image_list.sort()
        caption_filepath = f"data/train/sd-v1_4/mscoco2017/{split}2017.json"
        with open(caption_filepath, "r") as f:
            self.captions = json.load(f)

        self.transform = transform

    # -------------------------------------------------------------------------
    # CHỨC NĂNG:
    #     Trả về số lượng ảnh có trong tập MS COCO.
    # INPUT:
    #     - Không có tham số ngoài self.
    # OUTPUT:
    #     - (int): Tổng số lượng ảnh.
    # -------------------------------------------------------------------------
    def __len__(self):
        return len(self.image_list)

    # -------------------------------------------------------------------------
    # CHỨC NĂNG:
    #     Đọc ảnh PIL RGB theo chỉ số idx, áp dụng transform (nếu có), và lấy chuỗi chú thích tương ứng.
    # INPUT:
    #     - idx (int): Chỉ số phần tử cần lấy.
    # OUTPUT:
    #     - image (PIL.Image hoặc torch.Tensor): Dữ liệu ảnh.
    #     - caption (str): Chuỗi chú thích mô tả bức ảnh.
    # -------------------------------------------------------------------------
    def __getitem__(self, idx):
        image_filepath = self.image_list[idx]
        image = Image.open(image_filepath).convert("RGB")
        if self.transform:
            image = self.transform(image)
        image_id = os.path.basename(image_filepath)
        caption = self.captions[image_id]
        return image, caption


# =============================================================================
# CLASS: Flickr30k
# =============================================================================
class Flickr30k(torch.utils.data.Dataset):
    """Dataset quản lý ảnh thật từ nguồn Flickr30k (tải qua HuggingFace datasets)."""

    # -------------------------------------------------------------------------
    # CHỨC NĂNG:
    #     Tải hoặc nạp cache dataset Flickr30k qua thư viện datasets của Hugging Face.
    # INPUT:
    #     - split (str): Phân đoạn tập dữ liệu ('test', 'train', 'val'). Mặc định 'test'.
    #     - transform (callable): Pipeline biến đổi hình ảnh. Mặc định None.
    # OUTPUT:
    #     - Đối tượng Flickr30k.
    # -------------------------------------------------------------------------
    def __init__(self, split="test", transform=None):
        self.dataset = ds.load_dataset("nlphuji/flickr30k")[split]
        self.transform = transform

    # -------------------------------------------------------------------------
    # CHỨC NĂNG:
    #     Trả về tổng số mẫu của tập Flickr30k.
    # INPUT:
    #     - Không có tham số ngoài self.
    # OUTPUT:
    #     - (int): Số lượng phần tử trong dataset.
    # -------------------------------------------------------------------------
    def __len__(self):
        return len(self.dataset)

    # -------------------------------------------------------------------------
    # CHỨC NĂNG:
    #     Lấy ảnh và chọn ngẫu nhiên một trong số các câu mô tả chú thích đi kèm.
    # INPUT:
    #     - idx (int): Chỉ số phần tử cần lấy.
    # OUTPUT:
    #     - image (PIL.Image hoặc torch.Tensor): Dữ liệu ảnh sau transform.
    #     - caption (str): Câu chú thích ngẫu nhiên được chọn.
    # -------------------------------------------------------------------------
    def __getitem__(self, idx):
        example = self.dataset[idx]
        image = example["image"]
        if self.transform:
            image = self.transform(image)
        caption_list = example["caption"]
        caption = np.random.choice(caption_list)
        return image, caption


# =============================================================================
# CLASS: OtherReal
# =============================================================================
class OtherReal(torch.utils.data.Dataset):
    """Dataset quản lý các mẫu ảnh thật dùng cho benchmark từ các nguồn khác: CC3M, SBU, TextCaps."""

    # -------------------------------------------------------------------------
    # CHỨC NĂNG:
    #     Khởi tạo danh sách ảnh cho một trong các tập CC3M, SBU hoặc TextCaps.
    # INPUT:
    #     - dataset_name (str): Tên tập dữ liệu ('cc3m', 'sbu', hoặc 'textcaps').
    #     - transform (callable): Pipeline biến đổi hình ảnh. Mặc định None.
    # OUTPUT:
    #     - Đối tượng OtherReal.
    # -------------------------------------------------------------------------
    def __init__(self, dataset_name, transform=None):
        root_dir = "data/test/Co-Spy-Bench/real_image_examples"
        if dataset_name == "cc3m":
            self.image_dir = os.path.join(root_dir, "CC3M")
        elif dataset_name == "sbu":
            self.image_dir = os.path.join(root_dir, "SBU")
        elif dataset_name == "textcaps":
            self.image_dir = os.path.join(root_dir, "TextCaps")
        else:
            raise ValueError(
                f"Dataset {dataset_name} not supported in OtherReal class."
            )

        self.image_list = [f for f in os.listdir(self.image_dir) if f.endswith(".jpg")]
        self.image_list.sort()

        self.transform = transform

    # -------------------------------------------------------------------------
    # CHỨC NĂNG:
    #     Trả về số lượng ảnh của tập dữ liệu này.
    # INPUT:
    #     - Không có tham số ngoài self.
    # OUTPUT:
    #     - (int): Tổng số lượng file ảnh.
    # -------------------------------------------------------------------------
    def __len__(self):
        return len(self.image_list)

    # -------------------------------------------------------------------------
    # CHỨC NĂNG:
    #     Đọc ảnh và nạp chuỗi chú thích tương ứng từ file json cùng tên.
    # INPUT:
    #     - idx (int): Chỉ số phần tử cần lấy.
    # OUTPUT:
    #     - image (PIL.Image hoặc torch.Tensor): Dữ liệu ảnh sau transform.
    #     - caption (str): Chuỗi chú thích được đọc từ file json.
    # -------------------------------------------------------------------------
    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        caption_filepath = img_path.replace(".jpg", ".json")
        with open(caption_filepath, "r") as f:
            caption_data = json.load(f)
        caption = caption_data["caption"]
        return image, caption
