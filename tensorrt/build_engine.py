"""TensorRT engine builder for optimized inference."""
import tensorrt as trt
import numpy as np
import onnx
import yaml
import argparse
import os
from pathlib import Path
from typing import Dict, Any, Optional


class TensorRTBuilder:
    def __init__(self, logger_level: trt.Logger.Severity = trt.Logger.WARNING):
        self.logger = trt.Logger(logger_level)
        self.builder = trt.Builder(self.logger)
        self.config = self.builder.create_builder_config()
        
    def build_engine_from_onnx(self,
                              onnx_path: str,
                              engine_path: str,
                              config: Dict[str, Any]) -> str:
        """Build TensorRT engine from ONNX model.
        
        Args:
            onnx_path: Path to ONNX model
            engine_path: Path to save TensorRT engine
            config: TensorRT build configuration
            
        Returns:
            Path to saved engine file
        """
        print(f"Building TensorRT engine from {onnx_path}")
        
        # Parse ONNX model
        network = self.builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        parser = trt.OnnxParser(network, self.logger)
        
        with open(onnx_path, 'rb') as model:
            if not parser.parse(model.read()):
                for error in range(parser.num_errors):
                    print(f"ONNX Parser Error: {parser.get_error(error)}")
                raise ValueError("Failed to parse ONNX model")
        
        # Configure builder
        self._configure_builder(config)
        
        # Set optimization profiles for dynamic shapes
        if config.get("dynamic_shapes"):
            self._set_optimization_profiles(network, config["dynamic_shapes"])
        
        # Build engine
        print("Building TensorRT engine...")
        serialized_engine = self.builder.build_serialized_network(network, self.config)
        
        if serialized_engine is None:
            raise RuntimeError("Failed to build TensorRT engine")
        
        # Save engine
        os.makedirs(os.path.dirname(engine_path), exist_ok=True)
        with open(engine_path, 'wb') as f:
            f.write(serialized_engine)
        
        print(f"✅ TensorRT engine saved to {engine_path}")
        return engine_path
    
    def _configure_builder(self, config: Dict[str, Any]):
        """Configure TensorRT builder settings."""
        # Memory pool
        if config.get("max_workspace_size"):
            self.config.set_memory_pool_limit(
                trt.MemoryPoolType.WORKSPACE, 
                config["max_workspace_size"] * (1 << 20)  # MB to bytes
            )
        
        # Precision
        if config.get("fp16", False):
            self.config.set_flag(trt.BuilderFlag.FP16)
            print("Enabled FP16 precision")
        
        if config.get("int8", False):
            self.config.set_flag(trt.BuilderFlag.INT8)
            print("Enabled INT8 precision")
            # Note: INT8 calibration would need additional setup
        
        # Optimization level
        if config.get("optimization_level"):
            self.config.builder_optimization_level = config["optimization_level"]
    
    def _set_optimization_profiles(self, network, dynamic_shapes: Dict[str, Dict]):
        """Set optimization profiles for dynamic input shapes."""
        profile = self.builder.create_optimization_profile()
        
        for input_name, shapes in dynamic_shapes.items():
            min_shape = shapes["min"]
            opt_shape = shapes["opt"] 
            max_shape = shapes["max"]
            
            profile.set_shape(input_name, min_shape, opt_shape, max_shape)
            print(f"Set dynamic shape for {input_name}: min={min_shape}, opt={opt_shape}, max={max_shape}")
        
        self.config.add_optimization_profile(profile)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load TensorRT build configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def build_engine(config_path: str):
    """Build TensorRT engine from configuration file."""
    config = load_config(config_path)
    
    # Extract paths and settings
    onnx_path = config["onnx_path"]
    engine_path = config["engine_path"]
    build_config = config.get("build_config", {})
    
    # Build engine
    builder = TensorRTBuilder()
    engine_path = builder.build_engine_from_onnx(onnx_path, engine_path, build_config)
    
    print(f"Engine build complete: {engine_path}")
    return engine_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build TensorRT engines")
    parser.add_argument("--config", type=str, required=True,
                       help="Path to TensorRT build configuration file")
    
    args = parser.parse_args()
    
    build_engine(args.config)