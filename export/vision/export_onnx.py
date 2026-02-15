"""Export vision model to ONNX format."""
import torch
import onnx
import sys
import os
from pathlib import Path

# Add parent directory to path to import models
sys.path.append(str(Path(__file__).parent.parent))

from models.vision.model import VisionModel
from models.vision.preprocess import VisionPreprocessor


def export_vision_model(
    model_name: str = "resnet50",
    output_path: str = "../../onnx_models/vision/1/model.onnx",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    batch_size: int = 1,
    input_size: int = 224,
    num_classes: int = None,
    feature_extraction: bool = True
):
    """Export vision model to ONNX format.
    
    Args:
        model_name: Model architecture name
        output_path: Path to save ONNX model
        device: Device to run export on
        batch_size: Batch size for export
        input_size: Input image size
        num_classes: Number of classes (for classification)
        feature_extraction: Whether to export as feature extractor
    """
    print(f"Exporting vision model: {model_name}")
    
    # Load model
    model = VisionModel(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=True,
        feature_extraction=feature_extraction
    )
    model.to(device)
    model.eval()
    
    # Create dummy input
    if model_name.startswith("vit"):
        # ViT expects specific input
        dummy_input = torch.randn(batch_size, 3, input_size, input_size).to(device)
    else:
        dummy_input = torch.randn(batch_size, 3, input_size, input_size).to(device)
    
    # Get model config
    config = model.get_config()
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Export to ONNX
    print(f"Exporting to: {output_path}")
    
    input_names = ["pixel_values"] if model_name.startswith("vit") else ["images"]
    output_names = ["features"] if feature_extraction else ["logits"]
    
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=config["dynamic_axes"],
        opset_version=14,
        do_constant_folding=True,
        export_params=True
    )
    
    # Verify ONNX model
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    
    print(f"✅ Successfully exported vision model to {output_path}")
    print(f"Model config: {config}")
    
    return output_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Export vision model to ONNX")
    parser.add_argument("--model_name", type=str, default="resnet50",
                       choices=["resnet18", "resnet34", "resnet50", "resnet101", 
                               "vit-base-patch16-224", "vit-large-patch16-224"],
                       help="Model architecture name")
    parser.add_argument("--output_path", type=str, 
                       default="../../onnx_models/vision/1/model.onnx",
                       help="Output ONNX model path")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Batch size for export")
    parser.add_argument("--input_size", type=int, default=224,
                       help="Input image size")
    parser.add_argument("--num_classes", type=int, default=None,
                       help="Number of classes for classification")
    parser.add_argument("--feature_extraction", action="store_true",
                       help="Export as feature extractor")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device for export")
    
    args = parser.parse_args()
    
    export_vision_model(
        model_name=args.model_name,
        output_path=args.output_path,
        device=args.device,
        batch_size=args.batch_size,
        input_size=args.input_size,
        num_classes=args.num_classes,
        feature_extraction=args.feature_extraction
    )