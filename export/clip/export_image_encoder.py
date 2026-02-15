"""Export CLIP image encoder to ONNX format."""
import torch
import onnx
import sys
import os
from pathlib import Path

# Add parent directory to path to import models
sys.path.append(str(Path(__file__).parent.parent))

from models.clip.image_encoder import CLIPImageEncoder
from models.clip.preprocess import CLIPPreprocessor


def export_clip_image_encoder(
    model_name: str = "openai/clip-vit-base-patch32",
    output_path: str = "../../onnx_models/clip_image/1/model.onnx",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    batch_size: int = 1,
    image_size: int = 224
):
    """Export CLIP image encoder to ONNX format.
    
    Args:
        model_name: HuggingFace model name
        output_path: Path to save ONNX model
        device: Device to run export on
        batch_size: Batch size for export
        image_size: Input image size
    """
    print(f"Exporting CLIP image encoder: {model_name}")
    
    # Load model
    model = CLIPImageEncoder(model_name, device)
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(batch_size, 3, image_size, image_size).to(device)
    
    # Get model config
    config = model.get_config()
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Export to ONNX
    print(f"Exporting to: {output_path}")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["pixel_values"],
        output_names=["image_embeds"],
        dynamic_axes=config["dynamic_axes"],
        opset_version=14,
        do_constant_folding=True,
        export_params=True
    )
    
    # Verify ONNX model
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    
    print(f"✅ Successfully exported CLIP image encoder to {output_path}")
    print(f"Model input shape: {config['input_shape']}")
    print(f"Model output shape: {config['output_shape']}")
    
    return output_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Export CLIP image encoder to ONNX")
    parser.add_argument("--model_name", type=str, default="openai/clip-vit-base-patch32",
                       help="HuggingFace model name")
    parser.add_argument("--output_path", type=str, 
                       default="../../onnx_models/clip_image/1/model.onnx",
                       help="Output ONNX model path")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Batch size for export")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device for export")
    
    args = parser.parse_args()
    
    export_clip_image_encoder(
        model_name=args.model_name,
        output_path=args.output_path,
        device=args.device,
        batch_size=args.batch_size
    )