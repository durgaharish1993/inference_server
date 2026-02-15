"""Export reranker model to ONNX format."""
import torch
import onnx
import sys
import os
from pathlib import Path

# Add parent directory to path to import models
sys.path.append(str(Path(__file__).parent.parent))

from models.reranker.model import RerankerModel


def export_reranker_model(
    model_name: str = "microsoft/DialoGPT-medium",
    output_path: str = "../../onnx_models/reranker/1/model.onnx",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    batch_size: int = 1,
    max_length: int = 512
):
    """Export reranker model to ONNX format.
    
    Args:
        model_name: HuggingFace model name
        output_path: Path to save ONNX model
        device: Device to run export on
        batch_size: Batch size for export
        max_length: Maximum sequence length
    """
    print(f"Exporting reranker model: {model_name}")
    
    # Load model
    model = RerankerModel(model_name, device)
    model.eval()
    
    # Create dummy inputs
    dummy_input_ids = torch.randint(0, 1000, (batch_size, max_length)).to(device)
    dummy_attention_mask = torch.ones(batch_size, max_length).to(device)
    dummy_token_type_ids = torch.zeros(batch_size, max_length, dtype=torch.long).to(device)
    
    # Get model config
    config = model.get_config()
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Export to ONNX
    print(f"Exporting to: {output_path}")
    torch.onnx.export(
        model,
        (dummy_input_ids, dummy_attention_mask, dummy_token_type_ids),
        output_path,
        input_names=config["input_names"],
        output_names=config["output_names"],
        dynamic_axes=config["dynamic_axes"],
        opset_version=14,
        do_constant_folding=True,
        export_params=True
    )
    
    # Verify ONNX model
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    
    print(f"✅ Successfully exported reranker model to {output_path}")
    print(f"Model config: {config}")
    
    return output_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Export reranker model to ONNX")
    parser.add_argument("--model_name", type=str, default="microsoft/DialoGPT-medium",
                       help="HuggingFace model name")
    parser.add_argument("--output_path", type=str, 
                       default="../../onnx_models/reranker/1/model.onnx",
                       help="Output ONNX model path")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Batch size for export")
    parser.add_argument("--max_length", type=int, default=512,
                       help="Maximum sequence length")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device for export")
    
    args = parser.parse_args()
    
    export_reranker_model(
        model_name=args.model_name,
        output_path=args.output_path,
        device=args.device,
        batch_size=args.batch_size,
        max_length=args.max_length
    )