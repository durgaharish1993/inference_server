"""Export CLIP text encoder to ONNX format."""
import torch
import onnx
import sys
import os
from pathlib import Path

# Add parent directory to path to import models
sys.path.append(str(Path(__file__).parent.parent))

from models.clip.text_encoder import CLIPTextEncoder


def export_clip_text_encoder(
    model_name: str = "openai/clip-vit-base-patch32",
    output_path: str = "../../onnx_models/clip_text/1/model.onnx",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    batch_size: int = 1,
    max_length: int = 77
):
    """Export CLIP text encoder to ONNX format.
    
    Args:
        model_name: HuggingFace model name
        output_path: Path to save ONNX model
        device: Device to run export on
        batch_size: Batch size for export
        max_length: Maximum sequence length
    """
    print(f"Exporting CLIP text encoder: {model_name}")
    
    # Load model
    model = CLIPTextEncoder(model_name, device)
    model.eval()
    
    # Create dummy inputs
    dummy_input_ids = torch.randint(0, 1000, (batch_size, max_length)).to(device)
    dummy_attention_mask = torch.ones(batch_size, max_length).to(device)
    
    # Get model config
    config = model.get_config()
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Export to ONNX
    print(f"Exporting to: {output_path}")
    torch.onnx.export(
        model,
        (dummy_input_ids, dummy_attention_mask),
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
    
    print(f"✅ Successfully exported CLIP text encoder to {output_path}")
    print(f"Model config: {config}")
    
    return output_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Export CLIP text encoder to ONNX")
    parser.add_argument("--model_name", type=str, default="openai/clip-vit-base-patch32",
                       help="HuggingFace model name")
    parser.add_argument("--output_path", type=str, 
                       default="../../onnx_models/clip_text/1/model.onnx",
                       help="Output ONNX model path")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Batch size for export")
    parser.add_argument("--max_length", type=int, default=77,
                       help="Maximum sequence length")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device for export")
    
    args = parser.parse_args()
    
    export_clip_text_encoder(
        model_name=args.model_name,
        output_path=args.output_path,
        device=args.device,
        batch_size=args.batch_size,
        max_length=args.max_length
    )