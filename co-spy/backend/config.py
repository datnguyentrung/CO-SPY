import os
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parent
APP_ROOT = BACKEND_ROOT.parent
PROJECT_ROOT = APP_ROOT.parent

# Keep large Hub downloads in a stable, project-owned location. This survives
# every dev-server restart and avoids depending on an interpreter-specific cache.
# These variables are set before OpenCLIP/Hugging Face are imported.
MODEL_CACHE_DIR = Path(os.environ.get("CO_SPY_MODEL_CACHE", PROJECT_ROOT / ".model-cache"))
OPENCLIP_CACHE_DIR = Path(
    os.environ.get("CO_SPY_OPENCLIP_CACHE", MODEL_CACHE_DIR / "openclip")
)
OPENCLIP_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _find_cached_openclip_weights() -> str | None:
    """Reuse a completed OpenCLIP download from this project or the user cache."""
    filename = "open_clip_model.safetensors"
    repo_cache = "models--timm--ViT-SO400M-14-SigLIP-384"
    cache_roots = (
        OPENCLIP_CACHE_DIR,
        Path.home() / ".cache" / "huggingface" / "hub",
    )
    for root in cache_roots:
        for candidate in (root / repo_cache / "snapshots").glob(f"*/{filename}"):
            if candidate.is_file() and candidate.stat().st_size > 3_000_000_000:
                return str(candidate)
    return None

ORIGINAL_MODEL_DIR = PROJECT_ROOT / "pretrained" / "sd-v1_4"
FUSION_V3_MODEL_DIR = PROJECT_ROOT / "pretrained" / "sd-v1_4_fushion_v3"

ORIGINAL_CHECKPOINTS = {
    "semantic": ORIGINAL_MODEL_DIR / "semantic_weights.pth",
    "artifact": ORIGINAL_MODEL_DIR / "artifact_weights.pth",
    "fusion": ORIGINAL_MODEL_DIR / "fusion_weights.pth",
}
FUSION_V3_CHECKPOINTS = {
    "semantic": FUSION_V3_MODEL_DIR / "semantic" / "checkpoint_best.pth",
    "artifact": FUSION_V3_MODEL_DIR / "artifact" / "checkpoint_best.pth",
    "npr": FUSION_V3_MODEL_DIR / "npr" / "checkpoint_best.pth",
    "fusion": FUSION_V3_MODEL_DIR / "fusion_v3" / "checkpoint_best.pth",
}

ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_IMAGE_BYTES = 15 * 1024 * 1024

# An explicit path wins; otherwise reuse a complete cache before requesting a download.
LOCAL_OPENCLIP_WEIGHTS = os.environ.get("CO_SPY_OPENCLIP_WEIGHTS") or _find_cached_openclip_weights()
