"""
=============================================================================
FILE: dataSets/__init__.py
CHỨC NĂNG & NHIỆM VỤ CỦA FILE:
    - Đóng vai trò là package initializer cho module quản lý dữ liệu (dataSets).
    - Xuất khẩu (export) và tập hợp toàn bộ các lớp Dataset phục vụ huấn luyện (Train) 
      và kiểm thử (Test Benchmark):
        1. TrainDataset: Dataset nạp ảnh thật/giả phục vụ huấn luyện thông thường.
        2. FusionTrainDataset: Dataset nạp ảnh áp dụng song song 2 pipeline tiền xử lý cho nhánh Fusion.
        3. CoSpyBenchTestDataset: Dataset kiểm thử cân bằng trên bộ Benchmark Co-Spy-Bench.
        4. AIGCDetectTestDataset: Dataset kiểm thử trên bộ Benchmark AIGCDetectionBenchMark.
    - Cung cấp danh sách các tập dữ liệu ảnh thật chuẩn và 22 mô hình sinh ảnh AI (generative models) 
      được đánh giá trong các bộ benchmark tiêu chuẩn.

DANH SÁCH CÁC THÀNH PHẦN XUẤT KHẨU TRONG FILE:
    - Các lớp Dataset:
        1. TrainDataset: Nạp ảnh huấn luyện (Real nhãn 0, Fake nhãn 1), xáo trộn và nén JPEG tùy chọn.
        2. FusionTrainDataset: Tạo đồng thời hai tensor ảnh (Semantic transform và Artifact transform).
        3. CoSpyBenchTestDataset: Ghép cặp kiểm thử giữa ảnh thật (MSCOCO/Flickr/...) và ảnh từ 22 mô hình sinh ảnh.
        4. AIGCDetectTestDataset: Nạp dữ liệu kiểm thử theo chuẩn AIGCDetectionBenchMark cho ProGAN.
    - Danh sách hằng số Benchmark:
        5. CoSpyBench_DATASET_LIST: Danh sách các tập dữ liệu gốc ảnh thật (mscoco, flickr, cc3m, textcaps, sbu).
        6. CoSpyBench_MODEL_LIST: Danh sách 22 mô hình sinh ảnh AI được kiểm thử trong Co-Spy-Bench 
           (gồm SD-v1.4, SD-v1.5, SD-v2.1, SDXL, FLUX, LCM, PixArt, Playground v2/v2.5, SSD-1B, v.v.).
        7. AIGCDetectionBenchMark_DATASET_LIST: Danh sách tập kiểm thử cho AIGCDetectionBenchMark.
        8. AIGCDetectionBenchMark_MODEL_LIST: Danh sách 11 mô hình sinh ảnh AI trong AIGCDetectionBenchMark 
           (ADM, BigGAN, CycleGAN, DALLE2, GauGAN, Glide, Midjourney, ProGAN, SD-v1.4, SD-v1.5, StarGAN).
=============================================================================
"""

from .dataset import (
    AIGCDetectTestDataset,
    CoSpyBenchTestDataset,
    FusionTrainDataset,
    TrainDataset,
)

# Co-Spy-Bench: List of evaluated real datasets
CoSpyBench_DATASET_LIST = ["mscoco", "flickr", "cc3m", "textcaps", "sbu"]

# Co-Spy-Bench: List of evaluated generative models
CoSpyBench_MODEL_LIST = [
    # CompVis
    "CompVis@ldm-text2im-large-256",
    "CompVis@stable-diffusion-v1-4",
    # runwayml
    "runwayml@stable-diffusion-v1-5",
    # segmind
    "segmind@SSD-1B",
    "segmind@tiny-sd",
    "segmind@SegMoE-SD-4x2-v0",
    "segmind@small-sd",
    # stabilityai
    "stabilityai@stable-diffusion-2-1",
    "stabilityai@stable-diffusion-3-medium-diffusers",
    "stabilityai@sdxl-turbo",
    "stabilityai@stable-diffusion-2",
    "stabilityai@stable-diffusion-xl-base-1.0",
    # playgroundai
    "playgroundai@playground-v2.5-1024px-aesthetic",
    "playgroundai@playground-v2-1024px-aesthetic",
    "playgroundai@playground-v2-512px-base",
    "playgroundai@playground-v2-256px-base",
    # PixArt-alpha
    "PixArt-alpha@PixArt-XL-2-1024-MS",
    "PixArt-alpha@PixArt-XL-2-512x512",
    # latent-consistency
    "latent-consistency@lcm-lora-sdxl",
    "latent-consistency@lcm-lora-sdv1-5",
    # black-forest-labs
    "black-forest-labs@FLUX.1-schnell",
    "black-forest-labs@FLUX.1-dev",
]

# AIGCDetectionBenchMark: List of evaluated real datasets
AIGCDetectionBenchMark_DATASET_LIST = ["test"]

# AIGCDetectionBenchMark: List of evaluated generative models
AIGCDetectionBenchMark_MODEL_LIST = [
    "ADM",
    "biggan",
    "cyclegan",
    "DALLE2",
    "gaugan",
    "Glide",
    "Midjourney",
    "progan",
    "stable_diffusion_v_1_4",
    "stable_diffusion_v_1_5",
    "stargan",
    "stylegan",
    "stylegan2",
    "VQDM",
    "whichfaceisreal",
    "wukong",
]


__all__ = [
    "AIGCDetectTestDataset",
    "AIGCDetectionBenchMark_DATASET_LIST",
    "AIGCDetectionBenchMark_MODEL_LIST",
    "CoSpyBenchTestDataset",
    "CoSpyBench_DATASET_LIST",
    "CoSpyBench_MODEL_LIST",
    "FusionTrainDataset",
    "TrainDataset",
]
