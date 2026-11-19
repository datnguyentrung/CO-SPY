from __future__ import annotations

import io
import logging
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

from backend.config import ALLOWED_IMAGE_TYPES, MAX_IMAGE_BYTES
from backend.inference import CoSpyRuntime, ModelUnavailableError

logging.basicConfig(level=logging.INFO, format="%(message)s")
LOGGER = logging.getLogger(__name__)
runtime = CoSpyRuntime()


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Model initialization can take minutes and download/cache upstream assets.
    # Start it in the background so health is available while the UI waits.
    threading.Thread(target=runtime.load, name="cospy-model-loader", daemon=True).start()
    yield


app = FastAPI(title="CO-SPY Demo", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict[str, object]:
    return {
        "status": "ok" if runtime.ready else "loading" if runtime.loading else "degraded",
        "device": runtime.device.type,
        "original_loaded": runtime.original_loaded,
        "fusion_v3_loaded": runtime.fusion_v3_loaded,
        "errors": runtime.errors,
    }


@app.post("/api/predict")
async def predict(image: UploadFile = File(...)) -> dict[str, object]:
    if image.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(status_code=415, detail="Please upload a JPG, PNG, or WEBP image.")
    content = await image.read()
    if not content or len(content) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image must be between 1 byte and 15 MB.")
    try:
        pil_image = Image.open(io.BytesIO(content)).convert("RGB")
        pil_image.load()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(status_code=422, detail="Unable to read this image.") from exc
    try:
        result = runtime.predict_both(pil_image)
    except ModelUnavailableError as exc:
        raise HTTPException(status_code=503, detail="Models are not ready: " + str(exc)) from exc
    except RuntimeError as exc:
        LOGGER.exception("[PREDICT] inference runtime error for image=%s", image.filename)
        raise HTTPException(status_code=503, detail="Unable to analyze this image. Please try again.") from exc
    LOGGER.info("[PREDICT] image=%s original=%s fusion_v3=%s", image.filename, result["original"]["prediction"], result["fusion_v3"]["prediction"])
    return result
