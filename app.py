"""
VisionAI — Multi-Model Smart Image Classifier
FastAPI backend: MobileNetV2 + ShuffleNetV2 + SqueezeNet (PyTorch/torchvision)
FIXED VERSION:
  - pretrained=True replaced with weights=...Weights.DEFAULT (new torchvision API)
  - content_type check removed — PIL auto-detects format (browser MIME types are unreliable)
  - models loaded at module level (no lifespan/app_state race condition)
  - separated try/except blocks for precise error messages
"""

import io
import time
import json
import logging
import urllib.request
from collections import Counter

import torch
import torchvision.models as models
from torchvision.models import (
    MobileNet_V2_Weights,
    ShuffleNet_V2_X1_0_Weights,
    SqueezeNet1_1_Weights,
)
from torchvision import transforms
from PIL import Image

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ─── ImageNet Labels ──────────────────────────────────────────────────────────
LABELS_URL = (
    "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels"
    "/master/imagenet-simple-labels.json"
)
LABELS_CACHE = "imagenet_labels.json"

def load_labels() -> list:
    try:
        with open(LABELS_CACHE) as f:
            labels = json.load(f)
            logger.info("Loaded %d labels from cache.", len(labels))
            return labels
    except FileNotFoundError:
        pass
    try:
        logger.info("Downloading ImageNet labels …")
        with urllib.request.urlopen(LABELS_URL, timeout=15) as r:
            labels = json.loads(r.read().decode())
        with open(LABELS_CACHE, "w") as f:
            json.dump(labels, f)
        logger.info("Downloaded %d labels.", len(labels))
        return labels
    except Exception as exc:
        logger.warning("Label download failed (%s) — using numeric class IDs.", exc)
        return [f"class_{i}" for i in range(1000)]

# ─── Load labels + models at module level ─────────────────────────────────────
logger.info("Loading ImageNet labels …")
LABELS = load_labels()

logger.info("Loading MobileNetV2 …")
mobilenet = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
mobilenet.eval()

logger.info("Loading ShuffleNetV2 …")
shufflenet = models.shufflenet_v2_x1_0(weights=ShuffleNet_V2_X1_0_Weights.DEFAULT)
shufflenet.eval()

logger.info("Loading SqueezeNet1_1 …")
squeezenet = models.squeezenet1_1(weights=SqueezeNet1_1_Weights.DEFAULT)
squeezenet.eval()

logger.info("All 3 models loaded and ready.")

# ─── Preprocessing (shared across all models) ─────────────────────────────────
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

# ─── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="VisionAI Multi-Model Classifier", version="2.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Helpers ───────────────────────────────────────────────────────────────────
def fmt_label(raw: str) -> str:
    return " ".join(w.capitalize() for w in raw.replace("_", " ").split())

def run_inference(model: torch.nn.Module, tensor: torch.Tensor) -> dict:
    t0 = time.perf_counter()
    with torch.no_grad():
        output = model(tensor)
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)

    probs = torch.nn.functional.softmax(output[0], dim=0)
    top5_probs, top5_idx = torch.topk(probs, 5)

    top5 = [
        {
            "label": fmt_label(LABELS[top5_idx[i].item()]),
            "score": round(float(top5_probs[i].item()) * 100, 2),
        }
        for i in range(5)
    ]
    return {
        "top1_label":        top5[0]["label"],
        "top1_score":        top5[0]["score"],
        "top5":              top5,
        "inference_time_ms": elapsed_ms,
    }

def get_consensus(mob: str, shuf: str, squ: str) -> dict:
    count = Counter([mob, shuf, squ])
    winner, votes = count.most_common(1)[0]
    if votes >= 2:
        return {
            "consensus_label": winner,
            "consensus_type":  "majority_agreement",
            "agreement":       f"{votes}/3 models agree",
            "flagged":         False,
        }
    return {
        "consensus_label": mob,
        "consensus_type":  "tie_broken_by_mobilenet",
        "agreement":       "All 3 models disagree — defaulting to MobileNetV2",
        "flagged":         True,
    }

# ─── Routes ───────────────────────────────────────────────────────────────────
@app.get("/")
async def serve_index():
    return FileResponse("static/index.html")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 1. Read bytes
    try:
        contents = await file.read()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"File read error: {exc}")

    if not contents:
        raise HTTPException(status_code=400, detail="Empty file.")

    # 2. Decode image (PIL handles JPEG/PNG/WebP automatically)
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as exc:
        raise HTTPException(
            status_code=415,
            detail=f"Cannot open image. Please use JPEG, PNG, or WebP. ({exc})"
        )

    # 3. Preprocess
    try:
        tensor = TRANSFORM(image).unsqueeze(0)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Preprocessing error: {exc}")

    # 4. Run all 3 models
    try:
        mob_res  = run_inference(mobilenet,  tensor)
        shuf_res = run_inference(shufflenet, tensor)
        squ_res  = run_inference(squeezenet, tensor)
    except Exception as exc:
        logger.error("Inference error", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}")

    # 5. Consensus
    consensus = get_consensus(
        mob_res["top1_label"],
        shuf_res["top1_label"],
        squ_res["top1_label"],
    )

    logger.info(
        "mob=%s(%.1f%%) | shuf=%s(%.1f%%) | squ=%s(%.1f%%) → [%s] %s",
        mob_res["top1_label"],  mob_res["top1_score"],
        shuf_res["top1_label"], shuf_res["top1_score"],
        squ_res["top1_label"],  squ_res["top1_score"],
        consensus["consensus_type"], consensus["consensus_label"],
    )

    return {
        "mobilenet":  mob_res,
        "shufflenet": shuf_res,
        "squeezenet": squ_res,
        "consensus":  consensus,
    }

# Mount static AFTER routes
app.mount("/static", StaticFiles(directory="static"), name="static")