import argparse
import os
import yaml
import torch
import sys
from pathlib import Path
from transformers import AutoTokenizer

# Add parent directory to path to import models
sys.path.append(str(Path(__file__).parent.parent.parent))

from models.text_embedding.model import TextEmbeddingModel


def export_onnx(cfg_path: str, out_path: str, device: str):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    model_name = cfg["hf_model_name"]
    max_seq_len = int(cfg.get("max_seq_len", 256))
    opset = int(cfg.get("opset", 11))  # Default to opset 11
    normalize = bool(cfg.get("normalize", True))

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = TextEmbeddingModel(model_name, normalize=normalize)
    model.eval().to(device)

    # Dummy inputs for tracing
    dummy = tokenizer(
        ["hello world", "another example"],
        padding="max_length",
        truncation=True,
        max_length=max_seq_len,
        return_tensors="pt",
    )
    input_ids = dummy["input_ids"].to(device)
    attention_mask = dummy["attention_mask"].to(device)

    # Export
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    dynamic_axes = {
        "input_ids": {0: "batch", 1: "seq"},
        "attention_mask": {0: "batch", 1: "seq"},
        "embeddings": {0: "batch"},
    }

    with torch.no_grad():
        torch.onnx.export(
            model,
            (input_ids, attention_mask),
            out_path,
            input_names=["input_ids", "attention_mask"],
            output_names=["embeddings"],
            dynamic_axes=dynamic_axes,
            opset_version=11,  # Use more stable opset version
            do_constant_folding=True,
            export_params=True,
            verbose=False,
            training=torch.onnx.TrainingMode.EVAL,
            operator_export_type=torch.onnx.OperatorExportTypes.ONNX
        )

    print(f"✅ Exported ONNX to: {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="models/text_embedding/config.yaml")
    ap.add_argument("--out", default="onnx_models/text_embedding/1/model.onnx")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--model_name", default="sentence-transformers/all-MiniLM-L6-v2", 
                   help="HuggingFace model name (overrides config)")
    ap.add_argument("--max_seq_len", type=int, default=256, help="Max sequence length")
    ap.add_argument("--no_normalize", action="store_true", help="Disable normalization")
    args = ap.parse_args()

    # If config file exists, use it; otherwise use command line args
    if os.path.exists(args.config):
        export_onnx(args.config, args.out, args.device)
    else:
        print(f"Config file not found: {args.config}, using command line arguments")
        
        # Create a temporary config dict
        cfg = {
            "hf_model_name": args.model_name,
            "max_seq_len": args.max_seq_len,
            "normalize": not args.no_normalize,
            "opset": 11  # Use more stable opset version
        }
        
        # Export directly
        export_onnx_from_dict(cfg, args.out, args.device)


def export_onnx_from_dict(cfg: dict, out_path: str, device: str):
    """Export ONNX using config dictionary instead of file."""
    model_name = cfg["hf_model_name"]
    max_seq_len = int(cfg.get("max_seq_len", 256))
    opset = int(cfg.get("opset", 11))  # Default to opset 11
    normalize = bool(cfg.get("normalize", True))

    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = TextEmbeddingModel(model_name, normalize=normalize)
    model.eval().to(device)

    # Dummy inputs for tracing
    dummy = tokenizer(
        ["hello world", "another example"],
        padding="max_length",
        truncation=True,
        max_length=max_seq_len,
        return_tensors="pt",
    )
    input_ids = dummy["input_ids"].to(device)
    attention_mask = dummy["attention_mask"].to(device)

    # Export
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    dynamic_axes = {
        "input_ids": {0: "batch", 1: "seq"},
        "attention_mask": {0: "batch", 1: "seq"},
        "embeddings": {0: "batch"},
    }

    print(f"Exporting to ONNX: {out_path}")
    with torch.no_grad():
        torch.onnx.export(
            model,
            (input_ids, attention_mask),
            out_path,
            input_names=["input_ids", "attention_mask"],
            output_names=["embeddings"],
            dynamic_axes=dynamic_axes,
            opset_version=11,  # Use more stable opset version
            do_constant_folding=True,
            export_params=True,
            verbose=False,
            training=torch.onnx.TrainingMode.EVAL,
            operator_export_type=torch.onnx.OperatorExportTypes.ONNX
        )

    print(f"✅ Exported ONNX to: {out_path}")