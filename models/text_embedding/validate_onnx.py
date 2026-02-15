import argparse
import numpy as np
import yaml
import torch
from transformers import AutoTokenizer
import onnxruntime as ort
from models.text_embedding.model import TextEmbeddingModel


def l2(a): 
    return np.linalg.norm(a, axis=-1, keepdims=True)

def cosine(a, b):
    return (a * b).sum(-1) / (l2(a).squeeze(-1) * l2(b).squeeze(-1) + 1e-12)

def main(cfg_path: str, onnx_path: str, device: str):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    model_name = cfg["hf_model_name"]
    max_seq_len = int(cfg.get("max_seq_len", 256))
    normalize = bool(cfg.get("normalize", True))

    texts = ["hello world", "this is a longer test sentence", "DGX Spark is fast"]

    tok = AutoTokenizer.from_pretrained(model_name)
    batch = tok(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_seq_len,
        return_tensors="pt",
    )
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)

    # PyTorch
    pt_model = TextEmbeddingModel(model_name, normalize=normalize).eval().to(device)
    with torch.no_grad():
        pt_out = pt_model(input_ids, attention_mask).float().cpu().numpy()

    # ONNX
    sess = ort.InferenceSession(onnx_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    onnx_out = sess.run(
        ["embeddings"],
        {
            "input_ids": input_ids.cpu().numpy().astype(np.int64),
            "attention_mask": attention_mask.cpu().numpy().astype(np.int64),
        },
    )[0]

    # Compare
    cos = cosine(pt_out, onnx_out)
    max_abs = np.max(np.abs(pt_out - onnx_out))
    print("cosine similarity per row:", cos)
    print("min cosine:", float(np.min(cos)))
    print("max abs diff:", float(max_abs))

    # Heuristic thresholds (tune if needed)
    assert np.min(cos) > 0.995, "Cosine similarity too low — check opset/dtypes/export."
    print("✅ ONNX parity looks good.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="models/text_embedding/config.yaml")
    ap.add_argument("--onnx", default="onnx_models/text_embedding/1/model.onnx")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    main(args.config, args.onnx, args.device)