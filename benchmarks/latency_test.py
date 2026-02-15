"""Latency benchmark for inference models."""
import asyncio
import time
import statistics
import requests
import json
import base64
import numpy as np
from typing import List, Dict, Any
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LatencyBenchmark:
    def __init__(self, server_url: str = "http://localhost:8000"):
        self.server_url = server_url
        self.session = requests.Session()
    
    def generate_dummy_image(self, size: tuple = (224, 224)) -> str:
        """Generate dummy image as base64 string."""
        # Create random image
        img_array = np.random.randint(0, 255, (size[0], size[1], 3), dtype=np.uint8)
        
        # Convert to base64
        from PIL import Image
        import io
        img = Image.fromarray(img_array)
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG')
        img_b64 = base64.b64encode(buffer.getvalue()).decode()
        
        return img_b64
    
    def measure_latency(self, endpoint: str, payload: dict, num_requests: int = 100) -> Dict[str, float]:
        """Measure latency for endpoint."""
        latencies = []
        
        logger.info(f"Running {num_requests} requests to {endpoint}")
        
        for i in range(num_requests):
            start_time = time.time()
            
            try:
                response = self.session.post(f"{self.server_url}{endpoint}", json=payload)
                response.raise_for_status()
                
                end_time = time.time()
                latency = (end_time - start_time) * 1000  # Convert to ms
                latencies.append(latency)
                
                if (i + 1) % 10 == 0:
                    logger.info(f"Completed {i + 1}/{num_requests} requests")
                    
            except Exception as e:
                logger.error(f"Request {i+1} failed: {e}")
                continue
        
        if not latencies:
            raise ValueError("No successful requests")
        
        return {
            "mean": statistics.mean(latencies),
            "median": statistics.median(latencies),
            "p95": np.percentile(latencies, 95),
            "p99": np.percentile(latencies, 99),
            "min": min(latencies),
            "max": max(latencies),
            "std": statistics.stdev(latencies) if len(latencies) > 1 else 0,
            "total_requests": len(latencies),
            "failed_requests": num_requests - len(latencies)
        }
    
    def benchmark_text_embedding(self, num_requests: int = 100, batch_size: int = 1) -> Dict[str, float]:
        """Benchmark text embedding endpoint."""
        payload = {
            "texts": ["This is a sample text for embedding."] * batch_size
        }
        return self.measure_latency("/embed/text", payload, num_requests)
    
    def benchmark_image_embedding(self, num_requests: int = 100, batch_size: int = 1) -> Dict[str, float]:
        """Benchmark image embedding endpoint."""
        dummy_image = self.generate_dummy_image()
        payload = {
            "images": [dummy_image] * batch_size
        }
        return self.measure_latency("/embed/image", payload, num_requests)
    
    def benchmark_clip_text(self, num_requests: int = 100, batch_size: int = 1) -> Dict[str, float]:
        """Benchmark CLIP text embedding endpoint."""
        payload = {
            "texts": ["A sample text for CLIP embedding."] * batch_size
        }
        return self.measure_latency("/embed/clip_text", payload, num_requests)
    
    def benchmark_vision_features(self, num_requests: int = 100, batch_size: int = 1) -> Dict[str, float]:
        """Benchmark vision feature extraction endpoint."""
        dummy_image = self.generate_dummy_image()
        payload = {
            "images": [dummy_image] * batch_size
        }
        return self.measure_latency("/vision/features", payload, num_requests)
    
    def benchmark_reranking(self, num_requests: int = 100, num_docs: int = 10) -> Dict[str, float]:
        """Benchmark document reranking endpoint."""
        payload = {
            "query": "What is machine learning?",
            "documents": [f"Document {i} about various topics including AI and ML." for i in range(num_docs)],
            "top_k": 5
        }
        return self.measure_latency("/rerank", payload, num_requests)
    
    def run_full_benchmark(self, num_requests: int = 100) -> Dict[str, Dict[str, float]]:
        """Run full benchmark suite."""
        results = {}
        
        benchmarks = [
            ("text_embedding", lambda: self.benchmark_text_embedding(num_requests)),
            ("image_embedding", lambda: self.benchmark_image_embedding(num_requests)),
            ("clip_text", lambda: self.benchmark_clip_text(num_requests)),
            ("vision_features", lambda: self.benchmark_vision_features(num_requests)),
            ("reranking", lambda: self.benchmark_reranking(num_requests))
        ]
        
        for name, benchmark_func in benchmarks:
            logger.info(f"Running benchmark: {name}")
            try:
                results[name] = benchmark_func()
                logger.info(f"Completed benchmark: {name}")
            except Exception as e:
                logger.error(f"Benchmark {name} failed: {e}")
                results[name] = {"error": str(e)}
        
        return results


def print_results(results: Dict[str, Dict[str, float]]):
    """Print benchmark results in a formatted table."""
    print("\n" + "="*80)
    print("LATENCY BENCHMARK RESULTS")
    print("="*80)
    
    header = f"{'Endpoint':<20} {'Mean (ms)':<12} {'P95 (ms)':<12} {'P99 (ms)':<12} {'Requests':<10}"
    print(header)
    print("-" * len(header))
    
    for endpoint, metrics in results.items():
        if "error" in metrics:
            print(f"{endpoint:<20} ERROR: {metrics['error']}")
        else:
            print(f"{endpoint:<20} {metrics['mean']:<12.2f} {metrics['p95']:<12.2f} "
                  f"{metrics['p99']:<12.2f} {metrics['total_requests']:<10}")
    
    print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run latency benchmarks")
    parser.add_argument("--server-url", type=str, default="http://localhost:8000",
                       help="Server URL")
    parser.add_argument("--num-requests", type=int, default=100,
                       help="Number of requests per benchmark")
    parser.add_argument("--endpoint", type=str, default="all",
                       choices=["all", "text", "image", "clip_text", "vision", "rerank"],
                       help="Specific endpoint to benchmark")
    
    args = parser.parse_args()
    
    benchmark = LatencyBenchmark(args.server_url)
    
    if args.endpoint == "all":
        results = benchmark.run_full_benchmark(args.num_requests)
    elif args.endpoint == "text":
        results = {"text_embedding": benchmark.benchmark_text_embedding(args.num_requests)}
    elif args.endpoint == "image":
        results = {"image_embedding": benchmark.benchmark_image_embedding(args.num_requests)}
    elif args.endpoint == "clip_text":
        results = {"clip_text": benchmark.benchmark_clip_text(args.num_requests)}
    elif args.endpoint == "vision":
        results = {"vision_features": benchmark.benchmark_vision_features(args.num_requests)}
    elif args.endpoint == "rerank":
        results = {"reranking": benchmark.benchmark_reranking(args.num_requests)}
    
    print_results(results)
    
    # Save results to file
    with open("latency_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info("Results saved to latency_results.json")