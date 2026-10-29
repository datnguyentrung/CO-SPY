"""
=============================================================================
FILE: train.py
CHỨC NĂNG & NHIỆM VỤ CỦA FILE:
    - Cung cấp lớp Trainer điều hành toàn bộ quy trình huấn luyện (Training Pipeline) của CO-SPY.
    - Hỗ trợ 3 cơ chế huấn luyện:
        1. 'branch': Huấn luyện độc lập từng nhánh (ArtifactDetector hoặc SemanticDetector).
        2. 'fusion': Nạp trọng số đã huấn luyện của 2 nhánh, đóng băng (freeze) và chỉ huấn luyện
           lớp Linear phân loại kết hợp (Late Fusion Classifier).
        3. 'end2end': Huấn luyện đồng thời cả 2 nhánh từ đầu đến cuối (End-to-End).
    - Quản lý Optimizer (AdamW), hàm mất mát (BCEWithLogitsLoss), và Learning Rate Scheduler
      (Cosine Annealing hoặc Step-decay).
    - Hỗ trợ kỹ thuật tăng cường đặc trưng (Feature-space Interpolation / Mixup) cho nhánh Semantic.
    - Theo dõi đánh giá mô hình sau mỗi epoch, tự động lưu lại trọng số tốt nhất (best_model.pth)
      dựa trên độ chính xác (Accuracy) trên tập kiểm thử (Validation).

DANH SÁCH CÁC PHƯƠNG THỨC / HÀM TRONG FILE:
    1. Trainer.__init__(self, mode, device, branch, train_dataset, ckpt, epochs, batch_size, feat_interp, feat_interp_alpha, feat_interp_ratio):
       Khởi tạo bộ huấn luyện, nạp kiến trúc mô hình, cài đặt Optimizer, Loss Function, và thiết lập siêu tham số.
    2. Trainer._mix_features(self, feats, labels):
       Thực hiện nội suy tuyến tính (Mixup) trên không gian vector đặc trưng và nhãn mềm cho một phần batch.
    3. Trainer.train_step(self, batch_data):
       Thực hiện một bước huấn luyện đơn lẻ trên một mini-batch (forward pass, tính loss, backward, cập nhật tham số).
    4. Trainer._build_scheduler(self, steps_per_epoch):
       Khởi tạo bộ điều phối tốc độ học (CosineAnnealingLR) dựa trên tổng số bước hoặc theo epoch.
    5. Trainer._step_lr_per_epoch(self, epoch):
       Quy tắc giảm tỷ lệ học theo chu kỳ epoch (giảm 10% mỗi 10 epoch cho tập dữ liệu ProGAN).
    6. Trainer.train(self):
       Vòng lặp huấn luyện chính qua toàn bộ các epochs, thực hiện train, validate, log chỉ số và lưu checkpoint.
=============================================================================
"""

import hashlib
import importlib
import os
import time
import warnings

import torch
from loguru import logger

from dataSets import FusionTrainDataset, TrainDataset
from utils import evaluate

warnings.filterwarnings("ignore")


# =============================================================================
# CLASS: Trainer
# =============================================================================
class Trainer:
    # -------------------------------------------------------------------------
    # CHỨC NĂNG:
    #     Khởi tạo bộ điều phối huấn luyện: nạp mô hình theo dataset và mode,
    #     thiết lập Optimizer AdamW, Loss BCEWithLogitsLoss, cấu hình learning rate.
    # INPUT:
    #     - mode (str): Chế độ huấn luyện ('branch', 'fusion', 'end2end').
    #     - device (str): Thiết bị tính toán ('cuda:0' hoặc 'cpu').
    #     - branch (str): Tên nhánh ('artifact' hoặc 'semantic') nếu mode='branch'.
    #     - train_dataset (str): Tên bộ dữ liệu huấn luyện ('sd-v1_4' hoặc 'progan').
    #     - ckpt (str): Đường dẫn thư mục lưu trữ checkpoint mô hình.
    #     - epochs (int): Tổng số epoch cần huấn luyện. Mặc định: 10.
    #     - batch_size (int): Số mẫu dữ liệu trong một mini-batch. Mặc định: 32.
    #     - feat_interp (bool): Bật/tắt kỹ thuật nội suy đặc trưng cho nhánh Semantic.
    #     - feat_interp_alpha (float): Tham số phân phối Beta cho mixup đặc trưng.
    #     - feat_interp_ratio (float): Tỷ lệ các mẫu trong batch được áp dụng mixup.
    # OUTPUT:
    #     - Đối tượng Trainer đã sẵn sàng để gọi phương thức train().
    # -------------------------------------------------------------------------
    def __init__(
        self,
        mode: str,
        device: str,
        branch: str,
        train_dataset: str,
        ckpt: str = "ckpt",
        epochs: int = 10,
        batch_size: int = 32,
        feat_interp: bool = False,
        feat_interp_alpha: float = 0.2,
        feat_interp_ratio: float = 0.5,
        fusion_head: str = "v1",
        npr: bool = False,
        artifact_views: int = 1,
        cache_features: bool = True,
        branch_dropout: float = 0.15,
        downscale_prob: float = 0.0,
    ):

        self.device = device
        self.mode = mode
        self.branch = branch
        self.train_dataset = train_dataset
        self.ckpt = ckpt
        self.epochs = epochs
        self.batch_size = batch_size
        self.feat_interp = feat_interp
        self.feat_interp_alpha = feat_interp_alpha
        self.fusion_head = fusion_head
        self.npr = npr
        self.artifact_views = artifact_views
        self.cache_features = cache_features
        self.branch_dropout = branch_dropout
        self.downscale_prob = downscale_prob
        if fusion_head in {"v3", "v3_npr"} or npr:
            self.fusion_head = "v3"
            self.npr = True
        elif fusion_head not in {"v1", "v2"}:
            raise ValueError(f"Unknown fusion head: {fusion_head}")
        # Ánh xạ chuẩn xác tên package Python và tên thư mục dữ liệu/ckpt
        if train_dataset in ["sd_v1_4", "sd-v1_4"]:
            pkg_name = "sd_v1_4"  # Tên thư mục code trong detectors/
            dir_name = "sd-v1_4"  # Tên thư mục trong data/train/ và ckpt/
        else:
            pkg_name = train_dataset
            dir_name = train_dataset

        self.pkg_name = pkg_name
        self.dir_name = dir_name
        if pkg_name != "sd_v1_4" and (
            self.fusion_head != "v1" or self.npr
        ):
            raise ValueError("Adaptive fusion v2/v3 is currently supported for sd-v1_4 only")

        # Nạp động module tương ứng với bộ dataset ('progan' hoặc 'sd_v1_4')
        detector_module = importlib.import_module(f"detectors.{pkg_name}")

        # Khởi tạo mô hình dựa theo chế độ train (mode)
        if self.mode == "branch":
            ArtifactDetector = detector_module.ArtifactDetector
            SemanticDetector = detector_module.SemanticDetector
            NPRDetector = getattr(detector_module, "NPRDetector", None)
            if self.branch == "artifact":
                self.model = ArtifactDetector()
            elif self.branch == "semantic":
                self.model = SemanticDetector(downscale_prob=self.downscale_prob)
            elif self.branch == "npr" and NPRDetector is not None:
                self.model = NPRDetector()
            else:
                raise ValueError(f"Unknown detector: {self.branch}")
        elif self.mode == "fusion":
            # Chế độ Fusion yêu cầu phải có sẵn trọng số đã train xong của 2 nhánh con
            c_dir = (
                os.path.join(self.ckpt, dir_name)
                if os.path.exists(os.path.join(self.ckpt, dir_name))
                else os.path.join(self.ckpt, train_dataset)
            )
            semantic_weights_path = os.path.join(c_dir, "semantic", "best_model.pth")
            artifact_weights_path = os.path.join(c_dir, "artifact", "best_model.pth")
            if not os.path.exists(semantic_weights_path) or not os.path.exists(
                artifact_weights_path
            ):
                raise ValueError(
                    f"Semantic or Artifact weights path does not exist for fusion mode. Looked in: {semantic_weights_path}, {artifact_weights_path}"
                )
            npr_weights_path = os.path.join(c_dir, "npr", "best_model.pth")
            if self.fusion_head in {"v3", "v3_npr"} and not os.path.exists(
                npr_weights_path
            ):
                raise ValueError(
                    f"NPR weights path does not exist for v3 fusion: {npr_weights_path}"
                )
            weight_paths = [semantic_weights_path, artifact_weights_path]
            if self.npr:
                weight_paths.append(npr_weights_path)
            signature = "|".join(
                f"{path}:{os.stat(path).st_mtime_ns}:{os.stat(path).st_size}"
                for path in weight_paths
            )
            signature += (
                f"|fusion={self.fusion_head}"
                f"|downscale={self.downscale_prob}"
                f"|views={self.artifact_views}"
            )
            self.cache_key = hashlib.sha256(
                signature.encode("utf-8")
            ).hexdigest()[:12]
            CoSpyFusionDetector = detector_module.CoSpyFusionDetector
            self.model = CoSpyFusionDetector(
                semantic_weights_path=semantic_weights_path,
                artifact_weights_path=artifact_weights_path,
                npr_weights_path=npr_weights_path if self.npr else None,
                fusion_version="v3" if self.npr else self.fusion_head,
                branch_dropout=self.branch_dropout,
                artifact_views=self.artifact_views,
                semantic_downscale_prob=self.downscale_prob,
            )
            self.weight_init = 0.5 if dir_name == "sd-v1_4" else 0.0
            self.bias_floor = 2.0 if dir_name == "sd-v1_4" else 0.0
        elif self.mode == "end2end":
            End2EndDetector = detector_module.End2EndDetector
            self.model = End2EndDetector()
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        self.model.to(self.device)

        # Khởi tạo trọng số ngẫu nhiên theo phân phối chuẩn cho lớp phân loại fc
        if self.mode != "fusion" or self.fusion_head == "v1":
            torch.nn.init.normal_(self.model.fc.weight.data, 0.0, 0.02)
        if self.mode == "end2end":
            torch.nn.init.normal_(self.model.sem.fc.weight.data, 0.0, 0.02)
            torch.nn.init.normal_(self.model.art.fc.weight.data, 0.0, 0.02)

        if self.mode == "fusion" and self.fusion_head == "v1":
            with torch.no_grad():
                self.model.fc.weight.data += self.weight_init
                self.model.fc.bias.data.fill_(self.bias_floor)

        # Cấu hình bộ tối ưu hóa Optimizer (AdamW)
        _beta1 = 0.9
        _weight_decay = 1e-4 if self.mode == "fusion" else 0.0
        # Chỉ truyền các tham số cần tính gradient (requires_grad=True) vào optimizer
        params = [p for p in self.model.parameters() if p.requires_grad]
        logger.info(f"Trainable parameters: {len(params)}")

        # Learning rate: 1e-4 cho train nhánh thông thường, 1e-1 cho tinh chỉnh tầng fusion
        if self.mode == "fusion":
            self._lr = 1e-3 if self.fusion_head != "v1" else 1e-1
        else:
            self._lr = 1e-4
        self._lr_min = self._lr / self.epochs
        self.optimizer = torch.optim.AdamW(
            params, lr=self._lr, betas=(_beta1, 0.999), weight_decay=_weight_decay
        )

        # Hàm mất mát phân loại nhị phân (Binary Cross Entropy with Logits)
        self.criterion = torch.nn.BCEWithLogitsLoss()

        self.lr_scheduler = None

        # Cấu hình cập nhật Learning Rate theo từng batch hoặc theo từng epoch
        if self.train_dataset in ["sd_v1_4", "sd-v1_4"]:
            self.lr_step_per_batch = True
        elif self.train_dataset == "progan":
            self.lr_step_per_batch = False
        else:
            raise ValueError(f"Unknown train dataset: {self.train_dataset}")

    # -------------------------------------------------------------------------
    # CHỨC NĂNG:
    #     Áp dụng kỹ thuật nội suy tuyến tính không gian đặc trưng (Feature-space Interpolation)
    #     tương tự Mixup trên một tỷ lệ mẫu ngẫu nhiên trong batch:
    #     feat_mixed = delta * feat_a + (1 - delta) * feat_b
    #     label_mixed = delta * label_a + (1 - delta) * label_b với delta ~ Beta(alpha, alpha).
    # INPUT:
    #     - feats (torch.Tensor): Tensor đặc trưng hình ảnh trích xuất từ CLIP (B, D).
    #     - labels (torch.Tensor): Nhãn nhị phân gốc (B,).
    # OUTPUT:
    #     - feats_new (torch.Tensor): Tensor đặc trưng sau khi đã trộn nội suy.
    #     - labels_new (torch.Tensor): Nhãn mềm (soft labels) dạng số thực tương ứng sau khi trộn.
    # -------------------------------------------------------------------------
    def _mix_features(self, feats, labels):
        B = feats.size(0)
        device = feats.device
        labels_f = labels.float().clone()
        mix_mask = torch.rand(B, device=device) < self.feat_interp_ratio
        n_mix = int(mix_mask.sum().item())
        if n_mix == 0:
            return feats, labels_f
        mix_idx = torch.where(mix_mask)[0]
        partner = torch.randint(0, B, (n_mix,), device=device)
        beta = torch.distributions.Beta(self.feat_interp_alpha, self.feat_interp_alpha)
        delta = beta.sample((n_mix,)).to(device)
        feats_new = feats.clone()
        d2 = delta.unsqueeze(1)
        feats_new[mix_idx] = d2 * feats[mix_idx] + (1.0 - d2) * feats[partner]
        labels_new = labels_f.clone()
        labels_new[mix_idx] = (
            delta * labels_f[mix_idx] + (1.0 - delta) * labels_f[partner]
        )
        return feats_new, labels_new

    # -------------------------------------------------------------------------
    # CHỨC NĂNG:
    #     Thực hiện 1 bước lan truyền xuôi và ngược trên mini-batch dữ liệu:
    #     - Tùy chế độ (fusion, semantic mixup, hoặc standard), tính logits đầu ra.
    #     - Tính hàm mất mát loss qua BCEWithLogitsLoss (kèm phạt bias floor nếu là fusion).
    #     - Thực hiện backward() và optimizer.step() để cập nhật trọng số mạng.
    # INPUT:
    #     - batch_data (tuple):
    #         + Với fusion: (sem_images, art_images, labels)
    #         + Với branch/end2end: (images, labels)
    # OUTPUT:
    #     - eval_loss (float): Giá trị loss của batch này.
    #     - y_pred (list[float]): Danh sách xác suất dự đoán (sau hàm Sigmoid) cho các ảnh trong batch.
    #     - y_true (list[int]): Danh sách nhãn thực tế của batch.
    # -------------------------------------------------------------------------
    def train_step(self, batch_data):
        self.optimizer.zero_grad()

        if self.mode == "fusion":
            if self.fusion_head == "v1":
                sem_t, art_t, labels = batch_data
                sem_t = sem_t.to(self.device, non_blocking=True)
                art_t = art_t.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                pred_sem = self.model.sem(sem_t)
                pred_art = self.model.art(art_t)
                feat = torch.cat([pred_sem, pred_art], dim=1)
                if self.model.training and self.branch_dropout > 0:
                    drop = torch.rand(feat.shape[0], device=feat.device) < (
                        self.branch_dropout
                    )
                    stream = torch.randint(
                        0, 2, (feat.shape[0],), device=feat.device
                    )
                    keep = torch.ones_like(feat)
                    for index in range(2):
                        keep[:, index] = torch.logical_not(
                            torch.logical_and(drop, stream == index)
                        ).to(feat.dtype)
                    feat = feat * keep
                outputs = self.model.fc(feat)
                loss = self.criterion(outputs, labels.unsqueeze(1).float())
                b = self.model.fc.bias[0]
                loss = loss + torch.nn.functional.relu(self.bias_floor - b)
            else:
                if len(batch_data) == 4:
                    sem_t, art_t, npr_t, labels = batch_data
                    npr_t = npr_t.to(self.device, non_blocking=True)
                else:
                    sem_t, art_t, labels = batch_data
                    npr_t = None
                sem_t = sem_t.to(self.device, non_blocking=True)
                art_t = art_t.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                if sem_t.ndim == 4 or art_t.ndim >= 4:
                    sem_t, art_t, extracted_npr = self.model.extract_features(
                        sem_t, art_t, include_npr=npr_t is None
                    )
                    if npr_t is not None:
                        with torch.no_grad():
                            npr_t = self.model.npr.artifact_encoder(npr_t)
                    else:
                        npr_t = extracted_npr
                outputs = self.model.forward_features(
                    sem_t, art_t, npr_t
                )
                loss = self.criterion(outputs, labels.unsqueeze(1).float())
        else:
            inputs, labels = batch_data
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            # Nội suy không gian đặc trưng (Feature-space interpolation / Mixup) cho nhánh Semantic
            use_feat_interp = (
                self.feat_interp and self.mode == "branch" and self.branch == "semantic"
            )
            if use_feat_interp:
                with torch.no_grad():
                    encode_fn = (
                        getattr(self.model.clip, "encode_image", None)
                        or self.model.clip.get_image_features
                    )
                    feats = encode_fn(inputs)
                feats_mixed, labels_soft = self._mix_features(feats, labels)
                outputs = self.model.fc(feats_mixed)
                loss = self.criterion(outputs, labels_soft.unsqueeze(1))
            else:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels.unsqueeze(1).float())

        # Lan truyền ngược
        loss.backward()
        # Cập nhật tham số mô hình
        self.optimizer.step()
        if self.lr_scheduler is not None and self.lr_step_per_batch:
            self.lr_scheduler.step()

        eval_loss = loss.item()
        y_pred = outputs.sigmoid().flatten().tolist()
        y_true = labels.tolist()
        return eval_loss, y_pred, y_true

    # -------------------------------------------------------------------------
    # CHỨC NĂNG:
    #     Khởi tạo bộ lập lịch giảm tốc độ học (Learning Rate Scheduler):
    #     - Với sd-v1_4: CosineAnnealingLR giảm dần theo từng batch step.
    #     - Với progan: Step decay giảm theo từng chu kỳ epoch.
    # INPUT:
    #     - steps_per_epoch (int): Số batch trong một epoch huấn luyện.
    # OUTPUT:
    #     - Không trả về giá trị (gán trực tiếp vào self.lr_scheduler).
    # -------------------------------------------------------------------------
    def _build_scheduler(self, steps_per_epoch: int):
        if self.lr_step_per_batch:
            total_steps = self.epochs * steps_per_epoch
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps,
                eta_min=self._lr_min,
            )
            logger.info(
                f"LR cosine {self._lr:.1e} -> {self._lr_min:.1e} over {total_steps} steps"
            )
        else:
            logger.info(f"LR step-decay start {self._lr:.1e}, x0.9 every 10 epochs")

    # -------------------------------------------------------------------------
    # CHỨC NĂNG:
    #     Giảm tốc độ học (LR) sau mỗi 10 epoch đối với mô hình huấn luyện trên ProGAN: lr = lr * 0.9.
    # INPUT:
    #     - epoch (int): Chỉ số epoch hiện tại (bắt đầu từ 0).
    # OUTPUT:
    #     - Không trả về giá trị (cập nhật trực tiếp vào optimizer.param_groups).
    # -------------------------------------------------------------------------
    def _step_lr_per_epoch(self, epoch: int):
        if epoch % 10 == 0 and epoch != 0:
            for pg in self.optimizer.param_groups:
                pg["lr"] *= 0.9

    # -------------------------------------------------------------------------
    # CHỨC NĂNG:
    #     Thực hiện toàn bộ quy trình huấn luyện và kiểm thử qua các epoch:
    #     - Chuẩn bị DataLoader (TrainDataset hoặc FusionTrainDataset).
    #     - Chạy qua từng epoch: duyệt các batch qua train_step(), ghi nhận Loss/AP/Accuracy.
    #     - Sau mỗi epoch: chạy kiểm thử trên tập Validation để đánh giá năng lực khái quát hóa.
    #     - Tự động lưu checkpoint tốt nhất ('best_model.pth') khi đạt Accuracy cao kỷ lục.
    # INPUT:
    #     - Không nhận tham số ngoài self (sử dụng các thuộc tính đã cấu hình).
    # OUTPUT:
    #     - Không trả về giá trị. Các file trọng số ('best_model.pth', 'final_model.pth')
    #       và file log ('training.log') được lưu tại thư mục ckpt.
    # -------------------------------------------------------------------------
    def train(self):
        # Xác định thư mục lưu checkpoint
        if self.mode == "branch":
            subdir = self.branch
        elif self.mode == "fusion" and self.fusion_head != "v1":
            subdir = f"fusion_{self.fusion_head}"
        else:
            subdir = self.mode
        model_dir = os.path.join(self.ckpt, self.dir_name, subdir)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        # Cấu hình ghi log (Logger)
        log_path = f"{model_dir}/training.log"
        if os.path.exists(log_path):
            os.remove(log_path)

        logger_id = logger.add(
            log_path,
            format="{time:MM-DD at HH:mm:ss} | {level} | {module}:{line} | {message}",
            level="DEBUG",
        )

        # Nạp dữ liệu huấn luyện và kiểm thử (DataLoader)
        if self.mode == "fusion":
            train_split = "val" if self.fusion_head == "v1" else "train"
            base_train = TrainDataset(
                train_dataset=self.train_dataset,
                split=train_split,
                transform=None,
            )
            base_val = TrainDataset(
                train_dataset=self.train_dataset, split="val", transform=None
            )
            overlap = set(base_train.total_list).intersection(base_val.total_list)
            if overlap:
                raise ValueError(
                    f"Train/validation leakage detected: {len(overlap)} shared paths"
                )
            raw_train_dataset = FusionTrainDataset(
                base_train,
                self.model.sem.train_transform,
                self.model.art.train_transform,
                npr_transform=(
                    self.model.npr.train_transform
                    if self.fusion_head == "v3"
                    else None
                ),
            )
            raw_val_dataset = FusionTrainDataset(
                base_val,
                self.model.sem.val_transform,
                self.model.art.val_transform,
                npr_transform=(
                    self.model.npr.val_transform
                    if self.fusion_head == "v3"
                    else None
                ),
            )
            if self.fusion_head == "v1" or not self.cache_features:
                train_dataset = raw_train_dataset
                val_dataset = raw_val_dataset
            else:
                from detectors.sd_v1_4.feature_cache import (
                    build_fusion_feature_cache,
                )

                cache_dir = os.path.join(model_dir, "features", self.cache_key)
                train_dataset = build_fusion_feature_cache(
                    self.model,
                    raw_train_dataset,
                    cache_dir,
                    "train",
                    self.device,
                    batch_size=self.batch_size,
                )
                val_dataset = build_fusion_feature_cache(
                    self.model,
                    raw_val_dataset,
                    cache_dir,
                    "val",
                    self.device,
                    batch_size=self.batch_size,
                )
        else:
            train_dataset = TrainDataset(
                train_dataset=self.train_dataset,
                split="train",
                transform=self.model.train_transform,
            )
            val_dataset = TrainDataset(
                train_dataset=self.train_dataset,
                split="val",
                transform=self.model.val_transform,
            )
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        logger.info(f"Train size {len(train_dataset)} | Val size {len(val_dataset)}")
        if self.feat_interp and self.mode == "branch" and self.branch == "semantic":
            logger.info(
                f"feat_interp ON | alpha={self.feat_interp_alpha} "
                f"ratio={self.feat_interp_ratio}"
            )

        self._build_scheduler(steps_per_epoch=len(train_loader))

        # Bắt đầu vòng lặp huấn luyện qua các Epochs
        best_acc = 0.0
        best_ap = -float("inf")
        epochs_without_improvement = 0
        for epoch in range(self.epochs):
            self.model.train()
            time_start = time.time()
            for step_id, batch_data in enumerate(train_loader):
                eval_loss, y_pred, y_true = self.train_step(batch_data)
                ap, accuracy = evaluate(y_pred, y_true)

                # In thông số sau mỗi 100 batch
                if (step_id + 1) % 100 == 0:
                    time_end = time.time()
                    logger.info(
                        f"Epoch {epoch} | Batch {step_id + 1}/{len(train_loader)} | Loss {eval_loss:.4f} | AP {ap * 100:.2f}% | Accuracy {accuracy * 100:.2f}% | Time {time_end - time_start:.2f}s"
                    )
                    time_start = time.time()

            # Đánh giá mô hình trên tập validation (Evaluation)
            self.model.eval()
            y_pred, y_true = [], []
            with torch.no_grad():
                for batch in val_loader:
                    if self.mode == "fusion":
                        if len(batch) == 4:
                            sem_t, art_t, npr_t, labels = batch
                            npr_t = npr_t.to(self.device, non_blocking=True)
                        else:
                            sem_t, art_t, labels = batch
                            npr_t = None
                        sem_t = sem_t.to(self.device, non_blocking=True)
                        art_t = art_t.to(self.device, non_blocking=True)
                        if self.fusion_head == "v1":
                            feat = torch.cat(
                                [self.model.sem(sem_t), self.model.art(art_t)],
                                dim=1,
                            )
                            logits = self.model.fc(feat)
                        else:
                            if sem_t.ndim == 4 or art_t.ndim >= 4:
                                sem_t, art_t, extracted_npr = (
                                    self.model.extract_features(
                                        sem_t,
                                        art_t,
                                        include_npr=npr_t is None,
                                    )
                                )
                                if npr_t is not None:
                                    npr_t = self.model.npr.artifact_encoder(npr_t)
                                else:
                                    npr_t = extracted_npr
                            logits = self.model.forward_features(
                                sem_t, art_t, npr_t
                            )
                        y_pred.extend(logits.sigmoid().flatten().cpu().tolist())
                        y_true.extend(labels.tolist())
                    else:
                        images, labels = batch
                        y_pred.extend(self.model.predict(images))
                        y_true.extend(labels.tolist())

            ap, accuracy = evaluate(y_pred, y_true)
            eval_type = "Test" if self.mode == "branch" else "Total"
            cur_lr = self.optimizer.param_groups[0]["lr"]
            logger.info(
                f"Epoch {epoch} | {eval_type} AP {ap * 100:.2f}% | {eval_type} Accuracy {accuracy * 100:.2f}% | LR {cur_lr:.2e}"
            )

            if not self.lr_step_per_batch:
                self._step_lr_per_epoch(epoch)

            # Lưu lại model nếu đạt độ chính xác (Accuracy) cao nhất từ trước đến nay
            use_ap_checkpoint = self.mode == "fusion" and self.fusion_head != "v1"
            improved = ap > best_ap if use_ap_checkpoint else accuracy >= best_acc
            if use_ap_checkpoint:
                best_ap = max(best_ap, ap)
            else:
                best_acc = max(best_acc, accuracy)
            if improved:
                epochs_without_improvement = 0
                self.model.save_weights(f"{model_dir}/best_model.pth")
                metric_name = "AP" if use_ap_checkpoint else "accuracy"
                metric_value = ap if use_ap_checkpoint else accuracy
                logger.info(
                    f"Best model saved with {metric_name} {metric_value * 100:.2f}%"
                )
            else:
                epochs_without_improvement += 1

            # Lưu checkpoint định kỳ mỗi 5 epoch
            if epoch % 5 == 0:
                self.model.save_weights(f"{model_dir}/epoch_{epoch}.pth")
                logger.info(f"Model saved at epoch {epoch}")
            if use_ap_checkpoint and epochs_without_improvement >= 5:
                logger.info("Early stopping after 5 epochs without AP improvement")
                break

        # Lưu mô hình cuối cùng sau khi hoàn tất tất cả epoch
        self.model.save_weights(f"{model_dir}/final_model.pth")
        logger.info("Final model saved")

        logger.remove(logger_id)
