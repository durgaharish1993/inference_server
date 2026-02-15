"""Pydantic schemas for API request/response models."""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union


# Text embedding schemas
class TextEmbeddingRequest(BaseModel):
    texts: List[str] = Field(..., description="List of texts to embed", min_items=1)


class TextEmbeddingResponse(BaseModel):
    embeddings: List[List[float]] = Field(..., description="Text embeddings")


# Image embedding schemas
class ImageEmbeddingRequest(BaseModel):
    images: List[str] = Field(..., description="List of base64-encoded images", min_items=1)


class ImageEmbeddingResponse(BaseModel):
    embeddings: List[List[float]] = Field(..., description="Image embeddings")


# CLIP text embedding schemas
class CLIPTextEmbeddingRequest(BaseModel):
    texts: List[str] = Field(..., description="List of texts to embed with CLIP", min_items=1)


class CLIPTextEmbeddingResponse(BaseModel):
    embeddings: List[List[float]] = Field(..., description="CLIP text embeddings")


# Vision feature extraction schemas
class VisionFeaturesRequest(BaseModel):
    images: List[str] = Field(..., description="List of base64-encoded images", min_items=1)


class VisionFeaturesResponse(BaseModel):
    features: List[List[float]] = Field(..., description="Vision features")


# Reranking schemas
class RerankRequest(BaseModel):
    query: str = Field(..., description="Query text")
    documents: List[str] = Field(..., description="List of documents to rank", min_items=1)
    top_k: Optional[int] = Field(None, description="Number of top results to return", ge=1)


class RankedDocument(BaseModel):
    index: int = Field(..., description="Original index of the document")
    document: str = Field(..., description="Document text")
    score: float = Field(..., description="Relevance score")


class RerankResponse(BaseModel):
    ranked_documents: List[RankedDocument] = Field(..., description="Ranked documents with scores")


# Batch processing schemas
class BatchTextEmbeddingRequest(BaseModel):
    batches: List[List[str]] = Field(..., description="Batches of texts to embed")


class BatchImageEmbeddingRequest(BaseModel):
    batches: List[List[str]] = Field(..., description="Batches of base64-encoded images")


# Multi-modal schemas
class MultiModalEmbeddingRequest(BaseModel):
    texts: Optional[List[str]] = Field(None, description="Texts to embed")
    images: Optional[List[str]] = Field(None, description="Base64-encoded images to embed")


class MultiModalEmbeddingResponse(BaseModel):
    text_embeddings: Optional[List[List[float]]] = Field(None, description="Text embeddings")
    image_embeddings: Optional[List[List[float]]] = Field(None, description="Image embeddings")


# Search and similarity schemas
class SimilaritySearchRequest(BaseModel):
    query_embedding: List[float] = Field(..., description="Query embedding vector")
    candidate_embeddings: List[List[float]] = Field(..., description="Candidate embeddings")
    top_k: Optional[int] = Field(10, description="Number of top results", ge=1)


class SimilarityResult(BaseModel):
    index: int = Field(..., description="Index of the candidate")
    score: float = Field(..., description="Similarity score")


class SimilaritySearchResponse(BaseModel):
    results: List[SimilarityResult] = Field(..., description="Similarity search results")


# Health and status schemas
class HealthResponse(BaseModel):
    status: str = Field(..., description="Health status")
    message: str = Field(..., description="Health message")


class ModelInfo(BaseModel):
    name: str = Field(..., description="Model name")
    version: str = Field(..., description="Model version")
    platform: str = Field(..., description="Model platform")
    state: str = Field(..., description="Model state")


class ModelsResponse(BaseModel):
    models: List[str] = Field(..., description="Available models")


# Error schemas
class ErrorResponse(BaseModel):
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")