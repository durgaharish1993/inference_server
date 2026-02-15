"""Inference router for handling requests to Triton models."""
import numpy as np
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient
from tritonclient.utils import np_to_triton_dtype
import logging
from typing import List, Dict, Any, Optional, Union
import asyncio
import io
import base64
from PIL import Image

logger = logging.getLogger(__name__)


class InferenceRouter:
    def __init__(self, triton_url: str, use_grpc: bool = True):
        """Initialize inference router with Triton client.
        
        Args:
            triton_url: URL of Triton inference server
            use_grpc: Whether to use gRPC or HTTP client
        """
        self.triton_url = triton_url
        self.use_grpc = use_grpc
        
        if use_grpc:
            self.client = grpcclient.InferenceServerClient(url=triton_url)
        else:
            self.client = httpclient.InferenceServerClient(url=triton_url)
        
        # Verify connection
        if not self.client.is_server_ready():
            raise ConnectionError(f"Triton server not ready at {triton_url}")
        
        logger.info(f"Connected to Triton server: {triton_url}")
    
    async def get_available_models(self) -> List[str]:
        """Get list of available models."""
        try:
            if self.use_grpc:
                models = self.client.get_model_repository_index()
                return [model.name for model in models]
            else:
                models = self.client.get_model_repository_index()
                return [model["name"] for model in models]
        except Exception as e:
            logger.error(f"Error getting available models: {e}")
            raise
    
    def _prepare_inputs(self, input_data: Dict[str, np.ndarray], model_name: str) -> List:
        """Prepare inputs for Triton inference."""
        inputs = []
        
        for input_name, data in input_data.items():
            if self.use_grpc:
                input_obj = grpcclient.InferInput(
                    input_name, 
                    data.shape, 
                    np_to_triton_dtype(data.dtype)
                )
                input_obj.set_data_from_numpy(data)
            else:
                input_obj = httpclient.InferInput(
                    input_name,
                    data.shape,
                    np_to_triton_dtype(data.dtype)
                )
                input_obj.set_data_from_numpy(data)
            
            inputs.append(input_obj)
        
        return inputs
    
    def _prepare_outputs(self, output_names: List[str]) -> List:
        """Prepare outputs for Triton inference."""
        outputs = []
        
        for output_name in output_names:
            if self.use_grpc:
                output_obj = grpcclient.InferRequestedOutput(output_name)
            else:
                output_obj = httpclient.InferRequestedOutput(output_name)
            
            outputs.append(output_obj)
        
        return outputs
    
    async def _infer(self, model_name: str, inputs: List, outputs: List) -> Dict[str, np.ndarray]:
        """Run inference on Triton server."""
        try:
            # Run inference
            response = self.client.infer(
                model_name=model_name,
                inputs=inputs,
                outputs=outputs
            )
            
            # Extract results
            results = {}
            for output in outputs:
                output_name = output.name()
                results[output_name] = response.as_numpy(output_name)
            
            return results
            
        except Exception as e:
            logger.error(f"Inference error for model {model_name}: {e}")
            raise
    
    async def embed_text(self, texts: List[str]) -> List[List[float]]:
        """Generate text embeddings."""
        # This would need tokenization logic
        # For now, return dummy embeddings
        batch_size = len(texts)
        embeddings = np.random.randn(batch_size, 768).astype(np.float32)
        return embeddings.tolist()
    
    async def embed_images(self, images: List[str]) -> List[List[float]]:
        """Generate image embeddings using CLIP image encoder."""
        try:
            # Decode base64 images and preprocess
            processed_images = []
            for img_b64 in images:
                # Decode base64 image
                img_data = base64.b64decode(img_b64)
                img = Image.open(io.BytesIO(img_data)).convert('RGB')
                
                # Preprocess (resize, normalize, etc.)
                # This would use the CLIP preprocessing
                img_array = np.array(img.resize((224, 224))).astype(np.float32)
                img_array = (img_array / 255.0 - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
                img_array = img_array.transpose(2, 0, 1)  # HWC -> CHW
                processed_images.append(img_array)
            
            # Stack into batch
            batch_images = np.stack(processed_images)
            
            # Prepare inputs/outputs
            inputs = self._prepare_inputs({"pixel_values": batch_images}, "clip_image")
            outputs = self._prepare_outputs(["image_embeds"])
            
            # Run inference
            results = await self._infer("clip_image", inputs, outputs)
            
            return results["image_embeds"].tolist()
            
        except Exception as e:
            logger.error(f"Error in image embedding: {e}")
            raise
    
    async def embed_clip_text(self, texts: List[str]) -> List[List[float]]:
        """Generate text embeddings using CLIP text encoder."""
        # This would need CLIP tokenization
        # For now, return dummy embeddings
        batch_size = len(texts)
        embeddings = np.random.randn(batch_size, 512).astype(np.float32)
        return embeddings.tolist()
    
    async def extract_vision_features(self, images: List[str]) -> List[List[float]]:
        """Extract vision features from images."""
        # Similar to embed_images but using vision model
        batch_size = len(images)
        features = np.random.randn(batch_size, 2048).astype(np.float32)
        return features.tolist()
    
    async def rerank_documents(self, 
                             query: str, 
                             documents: List[str], 
                             top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """Rerank documents for a query."""
        # This would need tokenization and reranker model inference
        # For now, return dummy rankings
        scores = np.random.rand(len(documents)).tolist()
        
        # Sort by score (descending)
        ranked_results = []
        for i, (doc, score) in enumerate(zip(documents, scores)):
            ranked_results.append({
                "index": i,
                "document": doc,
                "score": float(score)
            })
        
        ranked_results.sort(key=lambda x: x["score"], reverse=True)
        
        if top_k:
            ranked_results = ranked_results[:top_k]
        
        return ranked_results