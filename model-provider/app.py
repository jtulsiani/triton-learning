import os
from typing import List

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from torchvision.models import ResNet50_Weights, resnet50

MODEL_REPO_ROOT = os.getenv("MODEL_REPO_ROOT", "/models")
MODEL_NAME = "resnet50"
MODEL_VERSION = "1"
MODEL_DIR = os.path.join(MODEL_REPO_ROOT, MODEL_NAME, MODEL_VERSION)
MODEL_PATH = os.path.join(MODEL_DIR, "model.onnx")

app = FastAPI(title="Model Provider", version="1.0.0")


class InferenceRequest(BaseModel):
    tensor: List[List[List[List[float]]]]


@torch.no_grad()
def predict_top1(model: torch.nn.Module, input_tensor: torch.Tensor) -> int:
    logits = model(input_tensor)
    top1 = int(torch.argmax(logits, dim=1).item())
    return top1


def ensure_triton_repo_and_export(model: torch.nn.Module) -> None:
    os.makedirs(MODEL_DIR, exist_ok=True)
    if os.path.exists(MODEL_PATH):
        return

    dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy_input,
        MODEL_PATH,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )


@app.on_event("startup")
def startup_event() -> None:
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    model.eval()
    app.state.model = model
    ensure_triton_repo_and_export(model)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/infer")
@torch.no_grad()
def infer(payload: InferenceRequest) -> dict:
    model: torch.nn.Module = app.state.model
    input_tensor = torch.tensor(payload.tensor, dtype=torch.float32)
    top1 = predict_top1(model, input_tensor)
    return {"top1_class": top1}
