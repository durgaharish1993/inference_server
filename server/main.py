"""Main FastAPI server for inference serving."""
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
import os
from typing import Optional

from router import InferenceRouter
from schemas import *

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting inference server...")
    
    # Initialize router with Triton client
    triton_url = os.getenv("TRITON_URL", "localhost:8001")
    app.state.router = InferenceRouter(triton_url)
    
    logger.info(f"Connected to Triton server: {triton_url}")
    yield
    
    # Shutdown
    logger.info("Shutting down inference server...")


# Create FastAPI app
app = FastAPI(
    title="DGX Inference Server",
    description="Multi-modal inference API with text embedding, CLIP, vision, and reranking capabilities",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,  
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "message": "Inference server is running"}


@app.get("/models")
async def list_models():
    """List available models."""
    try:
        models = await app.state.router.get_available_models()
        return {"models": models}
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Text embedding endpoints
@app.post("/embed/text", response_model=TextEmbeddingResponse)
async def embed_text(request: TextEmbeddingRequest):
    """Generate text embeddings."""
    try:
        embeddings = await app.state.router.embed_text(request.texts)
        return TextEmbeddingResponse(embeddings=embeddings)
    except Exception as e:
        logger.error(f"Error in text embedding: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# CLIP endpoints  
@app.post("/embed/image", response_model=ImageEmbeddingResponse)
async def embed_image(request: ImageEmbeddingRequest):
    """Generate image embeddings using CLIP."""
    try:
        embeddings = await app.state.router.embed_images(request.images)
        return ImageEmbeddingResponse(embeddings=embeddings)
    except Exception as e:
        logger.error(f"Error in image embedding: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/embed/clip_text", response_model=CLIPTextEmbeddingResponse)
async def embed_clip_text(request: CLIPTextEmbeddingRequest):
    """Generate text embeddings using CLIP text encoder."""
    try:
        embeddings = await app.state.router.embed_clip_text(request.texts)
        return CLIPTextEmbeddingResponse(embeddings=embeddings)
    except Exception as e:
        logger.error(f"Error in CLIP text embedding: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Vision endpoints
@app.post("/vision/features", response_model=VisionFeaturesResponse) 
async def extract_vision_features(request: VisionFeaturesRequest):
    """Extract vision features from images."""
    try:
        features = await app.state.router.extract_vision_features(request.images)
        return VisionFeaturesResponse(features=features)
    except Exception as e:
        logger.error(f"Error in vision feature extraction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Reranking endpoints
@app.post("/rerank", response_model=RerankResponse)
async def rerank_documents(request: RerankRequest):
    """Rerank documents for a query."""
    try:
        ranked_results = await app.state.router.rerank_documents(
            request.query, 
            request.documents,
            request.top_k
        )
        return RerankResponse(ranked_documents=ranked_results)
    except Exception as e:
        logger.error(f"Error in document reranking: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=os.getenv("DEBUG", "false").lower() == "true"
    )