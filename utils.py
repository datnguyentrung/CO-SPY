"""
=============================================================================
FILE: utils.py
CHỨC NĂNG & NHIỆM VỤ CỦA FILE:
    - Cung cấp toàn bộ các hàm và tiện ích phụ trợ (Utilities) cho hệ thống CO-SPY:
        1. Cố định seed ngẫu nhiên nhằm đảm bảo tính tái lập (reproducibility).
        2. Đọc và quét danh sách tệp ảnh đệ quy hoặc từ file pickle.
        3. Tăng cường dữ liệu (Data Augmentation) cho ảnh RGB và tensor (Gaussian blur, JPEG compression).
        4. Resize ảnh tùy biến với nhiều chế độ nội suy ngẫu nhiên (Bilinear, Bicubic, Lanczos, Nearest).
        5. Chuyển đổi trọng số mô hình về CPU để lưu trữ checkpoint an toàn.
        6. Định nghĩa hàm mất mát làm mượt nhãn (LabelSmoothingBCEWithLogits).
        7. Đánh giá độ đo hiệu năng phân loại: Average Precision (AP) và Accuracy.

DANH SÁCH CÁC PHƯƠNG THỨC / HÀM TRONG FILE:
    1. seed_torch(seed): Cố định random seed cho Python, NumPy, PyTorch và CUDA.
    2. recursively_read(rootdir, must_contain, exts): Quét đệ quy lấy danh sách đường dẫn ảnh hợp lệ.
    3. get_list(path, must_contain): Đọc danh sách ảnh từ file pickle hoặc thư mục.
    4. data_augment(img, aug_config): Áp dụng làm mờ Gaussian và nén JPEG lên ảnh PIL.
    5. tensor_data_augment(images, aug_config): Áp dụng tăng cường dữ liệu hàng loạt trên batch Tensor PyTorch.
    6. sample_continuous(s): Lấy mẫu ngẫu nhiên giá trị thực liên tục từ khoảng [s[0], s[1]].
    7. sample_discrete(s): Lấy mẫu ngẫu nhiên một phần tử từ danh sách các giá trị rời rạc.
    8. gaussian_blur(img, sigma): Làm mờ Gaussian trên mảng ảnh numpy 3 kênh.
    9. cv2_jpg(img, compress_val): Nén và giải nén ảnh JPEG sử dụng OpenCV.
    10. pil_jpg(img, compress_val): Nén và giải nén ảnh JPEG sử dụng PIL qua BytesIO trong bộ nhớ.
    11. png_to_jpeg(img, quality): Chuyển đổi định dạng ảnh PIL từ PNG sang JPEG.
    12. jpeg_from_key(img, compress_val, key): Điều phối gọi hàm nén JPEG qua cv2 hoặc pil.
    13. custom_resize(img, rz_interp, loadSize): Thay đổi kích thước ảnh với phương pháp nội suy ngẫu nhiên.
    14. weights2cpu(weights): Di chuyển toàn bộ state_dict tensor trọng số về CPU.
    15. LabelSmoothingBCEWithLogits.forward(self, pred, target): Hàm mất mát BCE có làm mượt nhãn.
    16. evaluate(y_pred, y_true): Tính toán 2 độ đo Average Precision (AP) và Accuracy.
=============================================================================
"""

import os
import cv2
import torch
import pickle
import random
import numpy as np
from io import BytesIO
from PIL import Image, ImageFile
from torchvision import transforms
import torchvision.transforms.functional as TF
from scipy.ndimage.filters import gaussian_filter
from sklearn.metrics import average_precision_score

ImageFile.LOAD_TRUNCATED_IMAGES = True


# -------------------------------------------------------------------------
# CHỨC NĂNG:
#     Thiết lập hạt giống ngẫu nhiên (seed) cố định cho toàn bộ các thư viện 
#     (random, os, numpy, torch, cuda, cudnn) để đảm bảo kết quả huấn luyện 
#     và đánh giá có thể tái lập lại một cách chính xác.
# INPUT:
#     - seed (int): Giá trị hạt giống ngẫu nhiên (vd: 1024).
# OUTPUT:
#     - Không trả về giá trị (None).
# -------------------------------------------------------------------------
def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# -------------------------------------------------------------------------
# CHỨC NĂNG:
#     Duyệt cây thư mục đệ quy bắt đầu từ rootdir để tìm tất cả các file ảnh 
#     có định dạng hợp lệ trong exts và đường dẫn chứa chuỗi must_contain.
# INPUT:
#     - rootdir (str): Thư mục gốc bắt đầu tìm kiếm.
#     - must_contain (str): Chuỗi con bắt buộc phải có trong đường dẫn file.
#     - exts (list[str]): Danh sách đuôi file ảnh hợp lệ (png, jpg, jpeg, ...).
# OUTPUT:
#     - out (list[str]): Danh sách các đường dẫn tuyệt đối/tương đối tới file ảnh tìm được.
# -------------------------------------------------------------------------
def recursively_read(rootdir, must_contain, exts=["png", "PNG", "jpg", "JPG", "jpeg", "JPEG"]):
    out = []
    for r, d, f in os.walk(rootdir):
        for file in f:
            if (file.split('.')[1] in exts) and (must_contain in os.path.join(r, file)):
                out.append(os.path.join(r, file))
    return out


# -------------------------------------------------------------------------
# CHỨC NĂNG:
#     Lấy danh sách các đường dẫn file ảnh. Nếu path là file .pickle thì đọc 
#     danh sách đã lưu sẵn; nếu là thư mục thì quét đệ quy qua recursively_read.
# INPUT:
#     - path (str): Đường dẫn đến file .pickle hoặc thư mục chứa ảnh.
#     - must_contain (str): Từ khóa chuỗi bắt buộc phải nằm trong đường dẫn ảnh.
# OUTPUT:
#     - image_list (list[str]): Danh sách đường dẫn ảnh đã lọc.
# -------------------------------------------------------------------------
def get_list(path, must_contain=''):
    if ".pickle" in path:
        with open(path, 'rb') as f:
            image_list = pickle.load(f)
        image_list = [item for item in image_list if must_contain in item]
    else:
        image_list = recursively_read(path, must_contain)
    return image_list


# -------------------------------------------------------------------------
# CHỨC NĂNG:
#     Thực hiện tăng cường dữ liệu (Data Augmentation) ngẫu nhiên trên ảnh PIL:
#     - Làm mờ Gaussian Blur theo xác suất blur_prob và độ lệch chuẩn blur_sig.
#     - Nén JPEG compression theo xác suất jpg_prob với chất lượng jpg_qual và phương pháp jpg_method.
# INPUT:
#     - img (PIL.Image.Image): Ảnh gốc đầu vào.
#     - aug_config (dict): Dictionary cấu hình tham số tăng cường (blur_prob, blur_sig, jpg_prob, jpg_qual, jpg_method).
# OUTPUT:
#     - (PIL.Image.Image): Ảnh PIL sau khi đã áp dụng tăng cường dữ liệu.
# -------------------------------------------------------------------------
def data_augment(img, aug_config):
    img = np.array(img)
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
        img = np.repeat(img, 3, axis=2)

    if random.random() < aug_config["blur_prob"]:
        sig = sample_continuous(aug_config["blur_sig"])
        gaussian_blur(img, sig)

    if random.random() < aug_config["jpg_prob"]:
        method = sample_discrete(aug_config["jpg_method"])
        qual = sample_discrete(aug_config["jpg_qual"])
        img = jpeg_from_key(img, qual, method)

    return Image.fromarray(img)


# -------------------------------------------------------------------------
# CHỨC NĂNG:
#     Áp dụng tăng cường dữ liệu (blur và JPEG nén) trực tiếp trên một mini-batch 
#     tensor ảnh của PyTorch trên GPU/CPU.
# INPUT:
#     - images (torch.Tensor): Tensor kích thước (B, C, H, W) trong khoảng [0, 1].
#     - aug_config (dict): Cấu hình các tham số tăng cường độ mờ và nén ảnh.
# OUTPUT:
#     - outputs (torch.Tensor): Tensor kích thước (B, C, H, W) sau biến đổi.
# -------------------------------------------------------------------------
def tensor_data_augment(images, aug_config):
    device = images.device
    images = images.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = np.uint8(images * 255.)
    outputs = []
    for img in images:
        if random.random() < aug_config["blur_prob"]:
            sig = sample_continuous(aug_config["blur_sig"])
            gaussian_blur(img, sig)

        if random.random() < aug_config["jpg_prob"]:
            method = sample_discrete(aug_config["jpg_method"])
            qual = sample_discrete(aug_config["jpg_qual"])
            img = jpeg_from_key(img, qual, method)
        outputs.append(img)
    outputs = np.stack(outputs)
    outputs = torch.from_numpy(outputs).to(device).permute(0, 3, 1, 2).float() / 255.
    return outputs


# -------------------------------------------------------------------------
# CHỨC NĂNG:
#     Lấy mẫu một giá trị liên tục trong đoạn [s[0], s[1]]. Nếu s chỉ có 1 phần tử thì trả về s[0].
# INPUT:
#     - s (list hoặc tuple): Danh sách chứa 1 hoặc 2 giá trị số [min, max].
# OUTPUT:
#     - (float): Giá trị số thực ngẫu nhiên được lấy mẫu.
# -------------------------------------------------------------------------
def sample_continuous(s):
    if len(s) == 1:
        return s[0]
    if len(s) == 2:
        rg = s[1] - s[0]
        return random.random() * rg + s[0]
    raise ValueError("Length of iterable s should be 1 or 2.")


# -------------------------------------------------------------------------
# CHỨC NĂNG:
#     Lấy mẫu ngẫu nhiên 1 giá trị rời rạc từ danh sách các phần tử s.
# INPUT:
#     - s (list hoặc tuple): Danh sách các giá trị rời rạc.
# OUTPUT:
#     - Một phần tử ngẫu nhiên được chọn từ s.
# -------------------------------------------------------------------------
def sample_discrete(s):
    if len(s) == 1:
        return s[0]
    return random.choice(s)


# -------------------------------------------------------------------------
# CHỨC NĂNG:
#     Áp dụng bộ lọc làm mờ Gauss (Gaussian filter) trực tiếp trên từng kênh màu của ảnh numpy.
# INPUT:
#     - img (np.ndarray): Mảng ảnh 3 chiều (H, W, 3).
#     - sigma (float): Độ lệch chuẩn của hàm Gauss điều khiển mức độ làm mờ.
# OUTPUT:
#     - Không trả về giá trị (thay đổi trực tiếp trên mảng img).
# -------------------------------------------------------------------------
def gaussian_blur(img, sigma):
    gaussian_filter(img[:,:,0], output=img[:,:,0], sigma=sigma)
    gaussian_filter(img[:,:,1], output=img[:,:,1], sigma=sigma)
    gaussian_filter(img[:,:,2], output=img[:,:,2], sigma=sigma)


# -------------------------------------------------------------------------
# CHỨC NĂNG:
#     Mô phỏng nén mất mát định dạng JPEG sử dụng thư viện OpenCV.
# INPUT:
#     - img (np.ndarray): Mảng ảnh RGB (H, W, 3).
#     - compress_val (int): Chỉ số chất lượng nén JPEG (từ 0 đến 100).
# OUTPUT:
#     - (np.ndarray): Mảng ảnh RGB sau khi mã hóa và giải mã JPEG.
# -------------------------------------------------------------------------
def cv2_jpg(img, compress_val):
    img_cv2 = img[:,:,::-1]
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), compress_val]
    result, encimg = cv2.imencode('.jpg', img_cv2, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return decimg[:,:,::-1]


# -------------------------------------------------------------------------
# CHỨC NĂNG:
#     Mô phỏng nén mất mát định dạng JPEG sử dụng thư viện PIL qua vùng đệm bộ nhớ BytesIO.
# INPUT:
#     - img (np.ndarray): Mảng ảnh numpy RGB.
#     - compress_val (int): Chỉ số chất lượng nén JPEG (từ 0 đến 100).
# OUTPUT:
#     - (np.ndarray): Mảng ảnh numpy sau khi nén và giải nén JPEG.
# -------------------------------------------------------------------------
def pil_jpg(img, compress_val):
    out = BytesIO()
    img = Image.fromarray(img)
    img.save(out, format='jpeg', quality=compress_val)
    img = Image.open(out)
    img = np.array(img)
    out.close()
    return img


# -------------------------------------------------------------------------
# CHỨC NĂNG:
#     Chuyển đổi một bức ảnh PIL từ PNG sang JPEG với chất lượng nén cho trước.
# INPUT:
#     - img (PIL.Image.Image): Đối tượng ảnh PIL gốc.
#     - quality (int): Mức chất lượng nén JPEG (mặc định 95).
# OUTPUT:
#     - (PIL.Image.Image): Đối tượng ảnh PIL ở chuẩn nén JPEG.
# -------------------------------------------------------------------------
def png_to_jpeg(img, quality=95):
    out = BytesIO()
    img.save(out, format='jpeg', quality=quality)
    img = np.array(Image.open(out))
    out.close()
    img = Image.fromarray(img)
    return img


# -------------------------------------------------------------------------
# CHỨC NĂNG:
#     Hàm điều phối nén JPEG theo từ khóa thuật toán ('cv2' hoặc 'pil').
# INPUT:
#     - img (np.ndarray): Mảng ảnh RGB đầu vào.
#     - compress_val (int): Mức chất lượng nén (0-100).
#     - key (str): Tên thư viện nén ('cv2' hoặc 'pil').
# OUTPUT:
#     - (np.ndarray): Mảng ảnh RGB sau nén JPEG.
# -------------------------------------------------------------------------
def jpeg_from_key(img, compress_val, key):
    jpeg_dict = {'cv2': cv2_jpg, 'pil': pil_jpg}
    method = jpeg_dict[key]
    return method(img, compress_val)


# -------------------------------------------------------------------------
# CHỨC NĂNG:
#     Thay đổi kích thước ảnh (Resize) bằng cách chọn ngẫu nhiên một phương pháp 
#     nội suy trong danh sách rz_interp (Bilinear, Bicubic, Lanczos, Nearest).
# INPUT:
#     - img (PIL.Image.Image): Ảnh PIL gốc.
#     - rz_interp (list[str]): Danh sách các thuật toán nội suy được phép chọn.
#     - loadSize (int hoặc tuple): Kích thước mục tiêu sau khi resize.
# OUTPUT:
#     - (PIL.Image.Image): Ảnh PIL sau khi đã thay đổi kích thước.
# -------------------------------------------------------------------------
def custom_resize(img, rz_interp, loadSize):
    rz_dict = {'bilinear': Image.BILINEAR,
                'bicubic': Image.BICUBIC,
                'lanczos': Image.LANCZOS,
                'nearest': Image.NEAREST}
    interp = sample_discrete(rz_interp)
    return TF.resize(img, loadSize, interpolation=rz_dict[interp])


# -------------------------------------------------------------------------
# CHỨC NĂNG:
#     Chuyển toàn bộ các tensor trọng số trong state_dict của mô hình về bộ nhớ CPU 
#     để lưu file an toàn, tránh phụ thuộc vào GPU lúc lưu.
# INPUT:
#     - weights (dict): Dictionary chứa các tensor tham số của mô hình PyTorch.
# OUTPUT:
#     - weights (dict): Dictionary đã chuyển toàn bộ tensor về thiết bị CPU.
# -------------------------------------------------------------------------
def weights2cpu(weights):
    for key in weights:
        weights[key] = weights[key].cpu()
    return weights


# =============================================================================
# CLASS: LabelSmoothingBCEWithLogits
# =============================================================================
class LabelSmoothingBCEWithLogits(torch.nn.Module):
    """Lớp hàm mất mát phân loại nhị phân áp dụng kỹ thuật làm mượt nhãn (Label Smoothing)."""
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingBCEWithLogits, self).__init__()
        self.smoothing = smoothing

    # -------------------------------------------------------------------------
    # CHỨC NĂNG:
    #     Tính toán mất mát nhị phân với nhãn mềm: y_soft = y * (1 - e) + 0.5 * e.
    # INPUT:
    #     - pred (torch.Tensor): Logits đầu ra của mô hình (chưa qua sigmoid).
    #     - target (torch.Tensor): Nhãn nhị phân thực tế (0 hoặc 1).
    # OUTPUT:
    #     - loss (torch.Tensor): Giá trị mất mát trung bình (scalar tensor).
    # -------------------------------------------------------------------------
    def forward(self, pred, target):
        target = target.float() * (1.0 - self.smoothing) + 0.5 * self.smoothing
        loss = torch.nn.functional.binary_cross_entropy_with_logits(pred, target, reduction='mean')
        return loss


# -------------------------------------------------------------------------
# CHỨC NĂNG:
#     Tính toán hai chỉ số đánh giá phân loại phổ biến:
#     1. Average Precision (AP): Đo diện tích dưới đường cong Precision-Recall.
#     2. Accuracy: Tỷ lệ dự đoán chính xác với ngưỡng xác suất 0.5.
# INPUT:
#     - y_pred (list hoặc np.ndarray): Danh sách xác suất dự đoán (trong khoảng [0, 1]).
#     - y_true (list hoặc np.ndarray): Danh sách nhãn thực tế (0: Real, 1: AI-Generated).
# OUTPUT:
#     - ap (float): Điểm Average Precision.
#     - accuracy (float): Tỷ lệ chính xác (từ 0.0 đến 1.0).
# -------------------------------------------------------------------------
def evaluate(y_pred, y_true):
    ap = average_precision_score(y_true, y_pred)
    accuracy = ((np.array(y_pred) > 0.5) == y_true).mean()
    return ap, accuracy


def bootstrap_ap_ci(
    y_pred,
    y_true,
    n_samples: int = 500,
    confidence: float = 0.95,
    seed: int = 1024,
):
    """Return a percentile bootstrap confidence interval for AP."""
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)
    if n_samples <= 0:
        return [None, None]

    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_samples):
        indices = rng.integers(0, len(y_true), size=len(y_true))
        sample_labels = y_true[indices]
        if np.unique(sample_labels).size < 2:
            continue
        values.append(
            average_precision_score(sample_labels, y_pred[indices])
        )
    if not values:
        return [None, None]
    alpha = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(values, [alpha, 1.0 - alpha])
    return [float(lower), float(upper)]
