import io
import os
from contextlib import asynccontextmanager
from pathlib import Path

import torch
import yaml
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from torchvision import transforms

from model import CIFAR10_CLASSES, get_model

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

_model: torch.nn.Module | None = None
_device: torch.device | None = None

_MEAN = (0.4914, 0.4822, 0.4465)
_STD = (0.2470, 0.2435, 0.2616)

_INFER_TRANSFORM = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=_MEAN, std=_STD),
])


def _resolve_checkpoint_path() -> Path:
    """
    Find the checkpoint to load.
    Priority: CONFIG_PATH env var > /app/configs/training_config.yaml > local configs/
    """
    for cfg_path in [
        os.getenv("CONFIG_PATH", ""),
        "/app/configs/training_config.yaml",
        "configs/training_config.yaml",
    ]:
        if cfg_path and Path(cfg_path).exists():
            with open(cfg_path) as f:
                cfg = yaml.safe_load(f)
            ckpt = Path(cfg["output"]["checkpoint_dir"]) / cfg["output"]["model_name"]
            if ckpt.exists():
                return ckpt

    # Fallback: CHECKPOINT_PATH env var
    env_path = os.getenv("CHECKPOINT_PATH", "/app/checkpoints/classifier_v1.pt")
    return Path(env_path)


# ---------------------------------------------------------------------------
# Lifespan: load model once at startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _device

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = _resolve_checkpoint_path()

    if not ckpt_path.exists():
        raise RuntimeError(f"Checkpoint not found at {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=_device, weights_only=True)
    cfg = checkpoint.get("config", {})

    _model = get_model(
        architecture=cfg.get("model", {}).get("architecture", "resnet18"),
        num_classes=cfg.get("model", {}).get("num_classes", 10),
    ).to(_device)
    _model.load_state_dict(checkpoint["model_state_dict"])
    _model.eval()

    print(f"Model loaded from {ckpt_path} (epoch {checkpoint.get('epoch', '?')})", flush=True)
    yield

    # Cleanup
    _model = None


app = FastAPI(title="CIFAR-10 Classifier", version="1.0.0", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> JSONResponse:
    """Liveness and readiness probe target."""
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return JSONResponse({"status": "ok"})


@app.post("/predict")
async def predict(image: UploadFile = File()) -> JSONResponse:  # noqa: B008
    """
    Accept a PNG/JPEG image, return per-class probabilities.

    Example:
        curl -X POST http://localhost:8080/predict -F "image=@test.png"
    """
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    contents = await image.read()
    try:
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Cannot decode image: {exc}") from exc

    tensor = _INFER_TRANSFORM(pil_image).unsqueeze(0).to(_device)

    with torch.no_grad():
        logits = _model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).tolist()

    return JSONResponse({
        "predictions": [
            {"class": cls, "probability": round(p, 6)}
            for cls, p in zip(CIFAR10_CLASSES, probs)
        ],
        "top1": CIFAR10_CLASSES[int(torch.tensor(probs).argmax())],
    })


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
