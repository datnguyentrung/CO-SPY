"""
=============================================================================
FILE: main.py
CHỨC NĂNG & NHIỆM VỤ CỦA FILE:
    - Điểm khởi chạy (entry point) chính cho toàn bộ hệ thống CO-SPY.
    - Xử lý các tham số dòng lệnh (CLI arguments) như phase, mode, gpu, dataset, batch size, v.v.
    - Thiết lập hạt giống ngẫu nhiên (seed) để đảm bảo tính tái lập (reproducibility).
    - Cấu hình thiết bị tính toán (CUDA GPU hoặc CPU).
    - Điều phối luồng thực thi tương ứng với 3 chế độ:
        1. Huấn luyện (train)
        2. Đánh giá benchmark (eval)
        3. Kiểm thử / quét phân loại ảnh đơn lẻ (test)

DANH SÁCH CÁC PHƯƠNG THỨC / HÀM TRONG FILE:
    1. main(args): Hàm điều phối chính, nhận arguments và kích hoạt Trainer hoặc Detector tương ứng với phase đã chọn.
=============================================================================
"""

import torch

from evaluate import Detector
from train import Trainer
from utils import seed_torch


# -------------------------------------------------------------------------
# CHỨC NĂNG:
#     Hàm thực thi chính điều phối luồng làm việc theo 'phase' được chỉ định:
#     - 'train': Khởi tạo đối tượng Trainer và gọi trainer.train() để huấn luyện.
#     - 'eval': Khởi tạo Detector và chạy kiểm thử toàn bộ benchmark với evaluate_benchmark().
#     - 'test': Khởi tạo Detector và mở chế độ quét dự đoán ảnh người dùng nhập (scan()).
# INPUT:
#     - args (argparse.Namespace): Đối tượng chứa toàn bộ tham số dòng lệnh:
#         + phase (str): 'train', 'eval', hoặc 'test'.
#         + mode (str): 'branch', 'fusion', hoặc 'end2end'.
#         + device (str): 'cuda:X' hoặc 'cpu'.
#         + train_dataset (str): Tên bộ dữ liệu huấn luyện (vd: 'sd-v1_4', 'progan').
#         + branch (str): 'artifact' hoặc 'semantic' (nếu chạy branch mode).
#         + pretrain (bool): Có dùng pretrained weights có sẵn hay không.
#         + ckpt (str): Thư mục lưu/chứa checkpoint weights.
#         + epochs (int): Số epoch huấn luyện.
#         + batch_size (int): Kích thước batch.
#         + seed (int): Seed ngẫu nhiên.
#         + feat_interp (bool): Có áp dụng feature interpolation hay không.
#         + feat_interp_alpha, feat_interp_ratio (float): Các siêu tham số cho feat_interp.
# OUTPUT:
#     - Không trả về giá trị (None). Kết quả in ra terminal hoặc lưu file logs/checkpoint.
# -------------------------------------------------------------------------
def main(args):
    #########################################
    # Giai đoạn 1: Huấn luyện (Training)
    #########################################
    if args.phase == "train":
        # Khởi tạo đối tượng Trainer với các siêu tham số truyền vào
        trainer = Trainer(
            mode=args.mode,
            device=args.device,
            branch=args.branch,
            train_dataset=args.train_dataset,
            ckpt=args.ckpt,
            epochs=args.epochs,
            batch_size=args.batch_size,
            feat_interp=args.feat_interp,
            feat_interp_alpha=args.feat_interp_alpha,
            feat_interp_ratio=args.feat_interp_ratio,
            fusion_head=args.fusion_head,
            npr=args.npr,
            artifact_views=args.artifact_views,
            cache_features=args.cache_features,
            branch_dropout=args.branch_dropout,
            downscale_prob=args.downscale_prob,
            seed=args.seed,
            resume_mode=args.resume_mode,
            resume_path=args.resume_path,
            experiment_id=args.experiment_id,
            checkpoint_dir=args.checkpoint_dir,
            checkpoint_input_roots=args.checkpoint_input_root,
            checkpoint_interval_minutes=args.checkpoint_interval_minutes,
            recovery_retention=args.recovery_retention,
            session_timeout_minutes=args.session_timeout_minutes,
            pre_timeout_minutes=args.pre_timeout_minutes,
            num_workers=args.num_workers,
            log_interval_batches=args.log_interval_batches,
            early_stopping_patience=args.early_stopping_patience,
        )
        # Bắt đầu quá trình huấn luyện
        trainer.train()
    #########################################
    # Giai đoạn 2: Đánh giá Benchmark (Evaluation)
    #########################################
    elif args.phase == "eval":
        # Khởi tạo Detector để đánh giá hàng loạt
        detector = Detector(
            device=args.device,
            mode=args.mode,
            train_dataset=args.train_dataset,
            pretrain=args.pretrain,
            ckpt=args.ckpt,
            batch_size=args.batch_size,
            branch=args.branch,
            fusion_head=args.fusion_head,
            npr=args.npr,
            artifact_views=args.artifact_views,
            branch_dropout=args.branch_dropout,
            benchmark=args.benchmark,
            bootstrap_samples=args.bootstrap_samples,
        )
        # Bắt đầu đánh giá toàn diện trên benchmark
        detector.evaluate_benchmark()
    ##########################################
    # Giai đoạn 3: Kiểm thử trên 1 ảnh (Single image scan)
    ##########################################
    elif args.phase == "test":
        # Khởi tạo Detector ở chế độ inference
        detector = Detector(
            device=args.device,
            mode=args.mode,
            train_dataset=args.train_dataset,
            pretrain=args.pretrain,
            ckpt=args.ckpt,
            batch_size=args.batch_size,
            fusion_head=args.fusion_head,
            npr=args.npr,
            artifact_views=args.artifact_views,
            branch_dropout=args.branch_dropout,
        )
        # Nhận đường dẫn ảnh và trả về kết quả
        detector.scan()
    else:
        raise ValueError(f"Unknown phase: {args.phase}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        "Co-Spy: Combining Semantic and Pixel Features to Detect Synthetic Images by AI"
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU id to use")
    parser.add_argument(
        "--phase",
        type=str,
        default="test",
        choices=["train", "eval", "test"],
        help="Select the phase to run Co-Spy: train / eval / test",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="fusion",
        choices=["branch", "fusion", "end2end"],
        help="Select the mode of Co-Spy training",
    )
    parser.add_argument(
        "--train_dataset", type=str, default="sd-v1_4", help="Training dataset"
    )
    parser.add_argument(
        "--branch",
        type=str,
        default="artifact",
        choices=["artifact", "semantic", "npr"],
        help="Branch detector (for branch mode)",
    )
    parser.add_argument(
        "--pretrain",
        action="store_true",
        help="Whether to use pre-trained weights for evaluation",
    )
    parser.add_argument("--ckpt", type=str, default="ckpt", help="Checkpoint directory")
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs"
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--seed", type=int, default=1024, help="Random seed")
    parser.add_argument(
        "--feat_interp",
        action="store_true",
        default=False,
        help="Feature-space interpolation for semantic branch.",
    )
    parser.add_argument(
        "--feat_interp_alpha",
        type=float,
        default=0.2,
        help="Beta(alpha, alpha) shape for feat_interp.",
    )
    parser.add_argument(
        "--feat_interp_ratio",
        type=float,
        default=0.5,
        help="Fraction of batch to apply feat_interp.",
    )
    parser.add_argument(
        "--fusion_head",
        type=str,
        default="v1",
        choices=["v1", "v2", "v3", "v3_npr"],
        help="Fusion head version. v1 preserves released checkpoints.",
    )
    parser.add_argument(
        "--npr",
        action="store_true",
        help="Enable the optional NPR stream (equivalent to fusion_head=v3).",
    )
    parser.add_argument(
        "--artifact_views",
        type=int,
        default=1,
        choices=[1, 2],
        help="Number of native-resolution artifact views used during evaluation.",
    )
    parser.add_argument(
        "--branch_dropout",
        type=float,
        default=0.15,
        help="Probability of dropping one complete adaptive-fusion stream.",
    )
    parser.add_argument(
        "--downscale_prob",
        type=float,
        default=0.0,
        help="Optional semantic-only downscale/upscale ablation probability.",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default="auto",
        choices=["auto", "cospy", "aigcdetect"],
        help="Development/final benchmark. auto=cospy for SD and aigcdetect for ProGAN.",
    )
    parser.add_argument(
        "--bootstrap_samples",
        type=int,
        default=500,
        help="Bootstrap samples for the per-cell AP confidence interval; 0 disables it.",
    )
    parser.add_argument(
        "--resume-mode",
        choices=["auto", "full", "weights", "new"],
        default="auto",
        help=(
            "auto resumes a compatible full checkpoint; weights starts a new run "
            "from checkpoint weights; new ignores checkpoints."
        ),
    )
    parser.add_argument(
        "--no-resume",
        action="store_const",
        const="new",
        dest="resume_mode",
        help="Start a new run and ignore existing checkpoints.",
    )
    parser.add_argument(
        "--resume-path",
        type=str,
        default=None,
        help="Explicit checkpoint file to restore instead of automatic discovery.",
    )
    parser.add_argument(
        "--experiment-id",
        type=str,
        default=None,
        help="Explicit experiment identifier used for checkpoint compatibility.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help=(
            "Exact checkpoint output directory. Defaults to the existing "
            "ckpt/<dataset>/<mode> directory."
        ),
    )
    parser.add_argument(
        "--checkpoint-input-root",
        action="append",
        default=[],
        help=(
            "Additional directory to scan for checkpoints. Repeat for multiple "
            "roots; Kaggle /kaggle/input and Colab Drive are detected automatically."
        ),
    )
    parser.add_argument(
        "--checkpoint-interval-minutes",
        type=float,
        default=15.0,
        help="Minutes between recovery checkpoints; 0 disables time-based saves.",
    )
    parser.add_argument(
        "--recovery-retention",
        type=int,
        default=1,
        help=(
            "Number of recovery states kept, including checkpoint_recovery.pth. "
            "The default of 1 keeps no extra recovery backups."
        ),
    )
    parser.add_argument(
        "--session-timeout-minutes",
        type=float,
        default=12 * 60,
        help="Stop at a safe batch boundary before the runtime limit; 0 disables.",
    )
    parser.add_argument(
        "--pre-timeout-minutes",
        type=float,
        default=15.0,
        help="Grace period reserved before the session timeout.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader worker count.",
    )
    parser.add_argument(
        "--log-interval-batches",
        type=int,
        default=100,
        help="Write batch progress to the log every N batches.",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=5,
        help=(
            "Stop after N validation epochs without improvement in the "
            "selected metric. Branch runs monitor AP; set 0 to disable."
        ),
    )
    parser.add_argument(
        "--no-cache-features",
        action="store_false",
        dest="cache_features",
        help=(
            "Disable one-pass feature caching for the SD v1.4 semantic branch and "
            "adaptive fusion."
        ),
    )
    parser.set_defaults(cache_features=True)

    args = parser.parse_args()

    # Set random seed
    seed_torch(args.seed)

    # Set GPU device
    args.device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"

    # Run the experiment
    main(args)
