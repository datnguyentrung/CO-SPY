"""
=============================================================================
FILE: evaluate.py
CHỨC NĂNG & NHIỆM VỤ CỦA FILE:
    - Cung cấp lớp Detector phục vụ cho hai tác vụ chính:
        1. Đánh giá kiểm thử tự động toàn diện trên các bộ benchmark tiêu chuẩn 
           (Co-Spy-Bench cho Stable Diffusion, AIGCDetectionBenchMark cho ProGAN).
        2. Chế độ quét (scan) tương tác trực tiếp với người dùng trên từng file ảnh 
           đơn lẻ để phân loại ảnh thật hay ảnh AI sinh ra.
    - Quản lý tải mô hình phân loại (ArtifactDetector, SemanticDetector, 
      CoSpyFusionDetector, End2EndDetector) cùng trọng số tương ứng (pretrained hoặc checkpoint).
    - Tính toán các chỉ số thống kê hiệu năng gồm Average Precision (AP) và Accuracy.
    - Tự động ghi log chi tiết và lưu kết quả đánh giá ra các file JSON.

DANH SÁCH CÁC PHƯƠNG THỨC / HÀM TRONG FILE:
    1. Detector.__init__(self, device, mode, branch, train_dataset, pretrain, ckpt, batch_size):
       Khởi tạo detector, nạp module kiến trúc tương ứng, load trọng số và đưa mô hình về chế độ eval.
    2. Detector.evaluate_benchmark(self):
       Chạy đánh giá hàng loạt trên các tập dữ liệu và mô hình sinh ảnh trong benchmark, ghi log và lưu kết quả JSON.
    3. Detector.scan(self):
       Nhận đường dẫn ảnh từ người dùng, tiền xử lý, chạy suy luận và in ra dự đoán (Real hay AI-Generated kèm xác suất).
=============================================================================
"""

import os
import json
import torch
import importlib
import numpy as np
from tqdm import tqdm
from PIL import Image
from loguru import logger

from dataSets import *
from utils import evaluate

import warnings
warnings.filterwarnings("ignore")


# =============================================================================
# CLASS: Detector
# =============================================================================
class Detector:
    # -------------------------------------------------------------------------
    # CHỨC NĂNG:
    #     Khởi tạo đối tượng Detector, nạp động kiến trúc mô hình phù hợp theo dataset 
    #     và chế độ (branch/fusion/end2end), nạp trọng số tương ứng từ pretrained hoặc checkpoint.
    # INPUT:
    #     - device (str): Thiết bị tính toán ('cuda:0', 'cpu', ...).
    #     - mode (str): Chế độ mô hình ('branch', 'fusion', 'end2end'). Mặc định: 'fusion'.
    #     - branch (str): Nhánh cụ thể ('artifact' hoặc 'semantic') nếu mode='branch'.
    #     - train_dataset (str): Bộ dữ liệu huấn luyện tương ứng kiến trúc ('sd-v1_4' hoặc 'progan').
    #     - pretrain (bool): True nếu dùng trọng số pretrained có sẵn, False nếu dùng checkpoint tự train.
    #     - ckpt (str): Đường dẫn thư mục checkpoint chứa weights.
    #     - batch_size (int): Kích thước batch khi đánh giá benchmark.
    # OUTPUT:
    #     - Đối tượng Detector sẵn sàng cho evaluate_benchmark() hoặc scan().
    # -------------------------------------------------------------------------
    def __init__(self,
                 device: str,
                 mode: str = "fusion",
                 branch: str = "artifact",
                 train_dataset: str = "sd-v1_4",
                 pretrain: bool = False,
                 ckpt: str = "ckpt",
                 batch_size: int = 32):

        # Cấu hình thiết bị và tham số
        self.device = device
        self.mode = mode
        self.branch = branch
        self.train_dataset = train_dataset
        self.pretrain = pretrain
        self.ckpt = ckpt
        self.batch_size = batch_size

        # Ánh xạ chuẩn xác tên package Python và tên thư mục dữ liệu/weights
        if train_dataset in ["sd_v1_4", "sd-v1_4"]:
            pkg_name = "sd_v1_4"   # Tên thư mục code trong detectors/
            dir_name = "sd-v1_4"   # Tên thư mục trong pretrained/, data/
        else:
            pkg_name = train_dataset
            dir_name = train_dataset

        self.pkg_name = pkg_name
        self.dir_name = dir_name

        # Nạp động module kiến trúc mô hình (progan hoặc sd_v1_4)
        detector_module = importlib.import_module(f"detectors.{pkg_name}")

        # Khởi tạo mô hình và nạp trọng số tùy theo chế độ (mode)
        if self.mode == "branch":
            if self.branch == "artifact":
                DetectorClass = getattr(detector_module, "ArtifactDetector")
            elif self.branch == "semantic":
                DetectorClass = getattr(detector_module, "SemanticDetector")
            else:
                raise ValueError(f"Unknown branch: {self.branch}")
            self.model = DetectorClass()
            # Chọn đường dẫn trọng số: pretrained của tác giả hoặc best_model trong ckpt
            if pretrain:
                weights_path = f"pretrained/{dir_name}/{self.branch}_weights.pth"
            else:
                weights_path = os.path.join(self.ckpt, dir_name, self.branch, "best_model.pth")
            if not os.path.exists(weights_path):
                raise ValueError(f"{self.branch} weights path does not exist: {weights_path}")
            self.model.load_weights(weights_path)
        elif self.mode == "fusion":
            if pretrain:
                p_dir = f"pretrained/{dir_name}"
                semantic_weights_path = os.path.join(p_dir, "semantic_weights.pth")
                artifact_weights_path = os.path.join(p_dir, "artifact_weights.pth")
                fusion_weights_path = os.path.join(p_dir, "fusion_weights.pth")
            else:
                c_dir = os.path.join(self.ckpt, dir_name)
                semantic_weights_path = os.path.join(c_dir, "semantic", "best_model.pth")
                artifact_weights_path = os.path.join(c_dir, "artifact", "best_model.pth")
                fusion_weights_path = os.path.join(c_dir, "fusion", "best_model.pth")
            if not os.path.exists(semantic_weights_path) or not os.path.exists(artifact_weights_path) or not os.path.exists(fusion_weights_path):
                raise ValueError(f"Semantic, Artifact or Fusion weights path does not exist for fusion mode. Looked in: {semantic_weights_path}, {artifact_weights_path}, {fusion_weights_path}")
            CoSpyFusionDetector = getattr(detector_module, "CoSpyFusionDetector")
            self.model = CoSpyFusionDetector(
                semantic_weights_path=semantic_weights_path,
                artifact_weights_path=artifact_weights_path)
            self.model.load_weights(fusion_weights_path)
        elif self.mode == "end2end":
            End2EndDetector = getattr(detector_module, "End2EndDetector")
            self.model = End2EndDetector()
            end2end_weights_path = os.path.join(self.ckpt, self.train_dataset, "end2end", "best_model.pth")
            if not os.path.exists(end2end_weights_path):
                raise ValueError("End2End weights path does not exist for end2end mode")
            self.model.load_weights(end2end_weights_path)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        
        # Chuyển mô hình lên thiết bị (GPU/CPU) và kích hoạt chế độ eval (đóng băng dropout/batchnorm)
        self.model.to(self.device)
        self.model.eval()
    
    # -------------------------------------------------------------------------
    # CHỨC NĂNG:
    #     Thực hiện đánh giá toàn diện mô hình trên bộ Benchmark tương ứng:
    #     - Duyệt qua từng dataset và từng generative model trong benchmark list.
    #     - Nạp DataLoader và chạy dự đoán theo batch.
    #     - Tính toán Average Precision (AP) và Accuracy cho từng mô hình và trung bình toàn bộ.
    #     - Lưu kết quả chi tiết ra 'result.json', 'output.json' và ghi file log 'evaluation.log'.
    # INPUT:
    #     - Không nhận tham số ngoài self (sử dụng các thuộc tính đã cấu hình lúc khởi tạo).
    # OUTPUT:
    #     - Không trả về giá trị (None). Toàn bộ kết quả được ghi vào thư mục ckpt/eval hoặc ckpt/pretrain.
    # -------------------------------------------------------------------------
    def evaluate_benchmark(self):
        # Chọn bộ benchmark tương ứng với dataset
        if self.dir_name == "progan":
            benchmark_name = "AIGCDetectionBenchMark"
            TestDataset = AIGCDetectTestDataset
            eval_dataset_list = AIGCDetectionBenchMark_DATASET_LIST
            eval_model_list = AIGCDetectionBenchMark_MODEL_LIST
        elif self.dir_name == "sd-v1_4":
            benchmark_name = "Co-Spy-Bench"
            TestDataset = CoSpyBenchTestDataset
            eval_dataset_list = CoSpyBench_DATASET_LIST
            eval_model_list = CoSpyBench_MODEL_LIST
        else:
            raise ValueError(f"Unknown train dataset: {self.train_dataset}")

        # Tạo thư mục lưu kết quả đánh giá
        subdir = self.branch if self.mode == "branch" else self.mode
        if self.pretrain:
            save_dir = os.path.join(self.ckpt, self.dir_name, subdir, f"pretrain_{benchmark_name}")
        else:
            save_dir = os.path.join(self.ckpt, self.dir_name, subdir, f"eval_{benchmark_name}")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Cấu hình log đánh giá
        log_path = f"{save_dir}/evaluation.log"
        if os.path.exists(log_path):
            os.remove(log_path)
        
        logger_id = logger.add(
            log_path,
            format="{time:MM-DD at HH:mm:ss} | {level} | {module}:{line} | {message}",
            level="DEBUG",
        )

        save_output_path = os.path.join(save_dir, "output.json")
        save_result_path = os.path.join(save_dir, "result.json")

        result_all = {}
        output_all = {}
        # Vòng lặp duyệt qua từng dataset và từng generative model
        for dataset_name in eval_dataset_list:
            result_all[dataset_name] = {}
            output_all[dataset_name] = {}
            for model_name in eval_model_list:
                test_dataset = TestDataset(dataset=dataset_name, model=model_name, transform=self.model.test_transform)
                test_loader = torch.utils.data.DataLoader(test_dataset,
                                                          batch_size=self.batch_size,
                                                          shuffle=False,
                                                          num_workers=4,
                                                          pin_memory=True)

                y_pred, y_true = [], []
                for (images, labels) in tqdm(test_loader, desc=f"Evaluating {benchmark_name} - {dataset_name} - {model_name}"):
                    y_pred.extend(self.model.predict(images))
                    y_true.extend(labels.tolist())

                ap, accuracy = evaluate(y_pred, y_true)
                logger.info(f"Evaluate on {benchmark_name} - {dataset_name} - {model_name} | Size {len(y_true)} | AP {ap*100:.2f}% | Accuracy {accuracy*100:.2f}%")

                result_all[dataset_name][model_name] = {"size": len(y_true), "AP": ap, "Accuracy": accuracy}
                output_all[dataset_name][model_name] = {"y_pred": y_pred, "y_true": y_true}

        # Lưu kết quả ra file JSON
        with open(save_result_path, "w") as f:
            json.dump(result_all, f, indent=4)

        with open(save_output_path, "w") as f:
            json.dump(output_all, f, indent=4)
        
        # Tính toán và in ra hiệu năng trung bình trên toàn bộ benchmark
        avg_ap = np.mean([result_all[dataset][model]["AP"] for dataset in eval_dataset_list for model in eval_model_list])
        avg_accuracy = np.mean([result_all[dataset][model]["Accuracy"] for dataset in eval_dataset_list for model in eval_model_list])
        logger.info(f"Average Performance on {benchmark_name} | AP {avg_ap*100:.2f}% | Accuracy {avg_accuracy*100:.2f}%")

    # -------------------------------------------------------------------------
    # CHỨC NĂNG:
    #     Chế độ quét ảnh tương tác liên tục trong vòng lặp (Interactive Scan Loop):
    #     - Nạp mô hình một lần duy nhất.
    #     - Cho phép người dùng nhập liên tục đường dẫn các bức ảnh cần test.
    #     - Tự động làm sạch đường dẫn (bỏ dấu nháy kép/đơn khi kéo thả file).
    #     - Xuất kết quả chi tiết từng nhánh, trọng số kết hợp và kết luận.
    #     - Nhập 'q' hoặc 'exit' (hoặc nhấn Enter rỗng) để kết thúc phiên quét.
    # INPUT:
    #     - Nhận đường dẫn file ảnh từ standard input (bàn phím).
    # OUTPUT:
    #     - Không trả về giá trị (in kết quả trực tiếp ra màn hình console).
    # -------------------------------------------------------------------------
    def scan(self):
        print("\n" + "=" * 76)
        print(" CHẾ ĐỘ QUÉT ẢNH TƯƠNG TÁC (INTERACTIVE SCAN MODE)")
        print(" - Nhập đường dẫn file ảnh hoặc kéo thả ảnh vào terminal.")
        print(" - Nhập 'q', 'exit' hoặc gõ phím Enter trống để kết thúc.")
        print("=" * 76 + "\n")

        while True:
            try:
                user_input = input("Nhập đường dẫn ảnh để quét (hoặc 'q' để thoát): ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nĐã kết thúc phiên quét ảnh.")
                break

            # Kiểm tra điều kiện thoát
            if not user_input or user_input.lower() in ["q", "quit", "exit"]:
                print("Đã thoát chế độ quét ảnh. Cảm ơn bạn!")
                break

            # Tẩy sạch ký tự nháy kép hoặc nháy đơn khi kéo thả đường dẫn trên Windows terminal
            image_filepath = user_input.strip('"\'')

            if not os.path.exists(image_filepath):
                print(f"[CẢNH BÁO] Không tìm thấy file: {image_filepath}. Vui lòng thử lại!\n")
                continue

            try:
                # Đọc ảnh và chuyển sang hệ màu RGB
                image = Image.open(image_filepath).convert("RGB")
                image = self.model.test_transform(image)
                image = image.unsqueeze(0)  # Thêm chiều batch size (1, C, H, W)
                image = image.to(self.device)

                # Nếu đang ở chế độ Fusion, phân tích sâu từng nhánh
                if self.mode == "fusion" and hasattr(self.model, "sem") and hasattr(self.model, "art"):
                    with torch.no_grad():
                        x_sem = self.model.sem_transform(image)
                        x_art = self.model.art_transform(image)

                        logit_sem = self.model.sem(x_sem)
                        logit_art = self.model.art(x_art)

                        prob_sem = logit_sem.sigmoid().item()
                        prob_art = logit_art.sigmoid().item()

                        w_sem = self.model.fc.weight[0][0].item()
                        w_art = self.model.fc.weight[0][1].item()
                        bias = self.model.fc.bias[0].item()

                        feat = torch.cat([logit_sem, logit_art], dim=1)
                        total_logit = self.model.fc(feat)
                        prediction = total_logit.sigmoid().item()

                    tag_sem = "AI-Generated (TRUE)" if prob_sem > 0.5 else "Real (FALSE)"
                    tag_art = "AI-Generated (TRUE)" if prob_art > 0.5 else "Real (FALSE)"
                    tag_final = "AI-Generated" if prediction > 0.5 else "Real"

                    print("\n" + "=" * 76)
                    print(f"         KẾT QUẢ PHÂN TÍCH: {os.path.basename(image_filepath)}")
                    print("=" * 76)
                    print(f"[Nhánh 1 - Semantic (CLIP)]: Xác suất = {prob_sem:.3f} ({prob_sem*100:.1f}%) -> {tag_sem}")
                    print(f"                             Trọng số đóng góp w1 = {w_sem:.4f}, Logit = {logit_sem.item():.4f}")
                    print("-" * 76)
                    print(f"[Nhánh 2 - Artifact (VAE)]:  Xác suất = {prob_art:.3f} ({prob_art*100:.1f}%) -> {tag_art}")
                    print(f"                             Trọng số đóng góp w2 = {w_art:.4f}, Logit = {logit_art.item():.4f}")
                    print("-" * 76)
                    print(f"[Fusion Layer (Lớp kết hợp)]:")
                    print(f"  - Trọng số nhánh Semantic (w1) : {w_sem:.4f}")
                    print(f"  - Trọng số nhánh Artifact (w2) : {w_art:.4f}")
                    print(f"  - Độ lệch hệ số (Bias b)       : {bias:+.4f}")
                    print(f"  - Công thức tổng hợp Logit     : ({w_sem:.4f} * {logit_sem.item():.4f}) + ({w_art:.4f} * {logit_art.item():.4f}) {bias:+.4f} = {total_logit.item():.4f}")
                    print("=" * 76)
                    print(f"==> KẾT QUẢ TỔNG HỢP CUỐI CÙNG   : {prediction:.3f} ({prediction*100:.1f}%)")
                    print(f"==> KẾT LUẬN                     : {tag_final}")
                    print("=" * 76 + "\n")
                else:
                    # Chế độ branch hoặc end2end đơn thuần
                    prediction = self.model.predict(image)[0]
                    tag = "AI-Generated" if prediction > 0.5 else "Real"
                    print("\n" + "-" * 50)
                    print(f"File: {os.path.basename(image_filepath)}")
                    print(f"Co-Spy Prediction: {prediction:.3f} - {tag}")
                    print("-" * 50 + "\n")

            except Exception as e:
                print(f"[LỖI XỬ LÝ ẢNH]: {e}\n")
