"""Throughput benchmark for inference models."""
import asyncio
import aiohttp
import time
import json
import base64
import numpy as np
from typing import List, Dict, Any, Tuple
import argparse
import logging
from concurrent.futures import ThreadPoolExecutor
import threading

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ThroughputBenchmark:
    def __init__(self, server_url: str = "http://localhost:8000", max_concurrent: int = 50):
        self.server_url = server_url
        self.max_concurrent = max_concurrent
        self.session = None
    
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
    
    async def make_request(self, endpoint: str, payload: dict) -> Tuple[bool, float]:
        """Make single async request and return (success, latency)."""
        start_time = time.time()
        
        try:
            async with self.session.post(f"{self.server_url}{endpoint}", json=payload) as response:
                await response.json()
                end_time = time.time()
                return response.status == 200, (end_time - start_time) * 1000
        except Exception as e:
            end_time = time.time()
            logger.debug(f"Request failed: {e}")
            return False, (end_time - start_time) * 1000
    
    async def run_concurrent_requests(self, 
                                    endpoint: str, 
                                    payload: dict, 
                                    total_requests: int,
                                    duration_seconds: int = None) -> Dict[str, Any]:
        """Run concurrent requests for throughput testing."""
        
        connector = aiohttp.TCPConnector(limit=self.max_concurrent)
        timeout = aiohttp.ClientTimeout(total=60)
        self.session = aiohttp.ClientSession(connector=connector, timeout=timeout)
        
        try:
            results = []
            start_time = time.time()
            
            if duration_seconds:
                # Time-based benchmark
                end_time = start_time + duration_seconds
                request_count = 0
                
                while time.time() < end_time:
                    # Create batch of concurrent requests
                    batch_size = min(self.max_concurrent, 
                                   int((end_time - time.time()) * 10))  # Estimate requests needed
                    if batch_size <= 0:
                        break
                    
                    tasks = [self.make_request(endpoint, payload) for _ in range(batch_size)]
                    batch_results = await asyncio.gather(*tasks)
                    results.extend(batch_results)
                    request_count += batch_size
                    
                    logger.info(f"Completed {request_count} requests")
            
            else:
                # Count-based benchmark
                semaphore = asyncio.Semaphore(self.max_concurrent)
                
                async def bounded_request():
                    async with semaphore:
                        return await self.make_request(endpoint, payload)
                
                # Create all tasks
                tasks = [bounded_request() for _ in range(total_requests)]
                
                # Execute in batches to avoid memory issues
                batch_size = self.max_concurrent
                for i in range(0, len(tasks), batch_size):
                    batch = tasks[i:i + batch_size]
                    batch_results = await asyncio.gather(*batch)
                    results.extend(batch_results)
                    
                    logger.info(f"Completed {min(i + batch_size, total_requests)}/{total_requests} requests")
            
            end_time = time.time()
            total_duration = end_time - start_time
            
            # Process results
            successful_requests = sum(1 for success, _ in results if success)
            failed_requests = len(results) - successful_requests
            latencies = [latency for success, latency in results if success]
            
            throughput = successful_requests / total_duration if total_duration > 0 else 0
            
            return {
                "total_requests": len(results),
                "successful_requests": successful_requests,
                "failed_requests": failed_requests,
                "duration_seconds": total_duration,
                "throughput_rps": throughput,
                "avg_latency_ms": np.mean(latencies) if latencies else 0,
                "p95_latency_ms": np.percentile(latencies, 95) if latencies else 0,
                "p99_latency_ms": np.percentile(latencies, 99) if latencies else 0
            }
            
        finally:
            await self.session.close()
    
    async def benchmark_text_embedding(self, 
                                     total_requests: int = 1000,
                                     batch_size: int = 1,
                                     duration_seconds: int = None) -> Dict[str, Any]:
        """Benchmark text embedding throughput."""
        payload = {
            "texts": ["This is a sample text for embedding."] * batch_size
        }
        return await self.run_concurrent_requests("/embed/text", payload, total_requests, duration_seconds)
    
    async def benchmark_image_embedding(self,
                                      total_requests: int = 1000,
                                      batch_size: int = 1, 
                                      duration_seconds: int = None) -> Dict[str, Any]:
        """Benchmark image embedding throughput."""
        dummy_image = self.generate_dummy_image()
        payload = {
            "images": [dummy_image] * batch_size
        }
        return await self.run_concurrent_requests("/embed/image", payload, total_requests, duration_seconds)
    
    async def benchmark_reranking(self,
                                total_requests: int = 1000,
                                num_docs: int = 10,
                                duration_seconds: int = None) -> Dict[str, Any]:
        """Benchmark reranking throughput."""
        payload = {
            "query": "What is machine learning?",
            "documents": [f"Document {i} about various topics." for i in range(num_docs)],
            "top_k": 5
        }
        return await self.run_concurrent_requests("/rerank", payload, total_requests, duration_seconds)
    
    async def run_full_benchmark(self, 
                               total_requests: int = 1000,
                               duration_seconds: int = None) -> Dict[str, Dict[str, Any]]:
        """Run full throughput benchmark suite."""
        results = {}
        
        benchmarks = [
            ("text_embedding", lambda: self.benchmark_text_embedding(total_requests, 1, duration_seconds)),
            ("image_embedding", lambda: self.benchmark_image_embedding(total_requests, 1, duration_seconds)),
            ("reranking", lambda: self.benchmark_reranking(total_requests, 10, duration_seconds))
        ]
        
        for name, benchmark_func in benchmarks:
            logger.info(f"Running throughput benchmark: {name}")
            try:
                results[name] = await benchmark_func()
                logger.info(f"Completed throughput benchmark: {name}")
            except Exception as e:
                logger.error(f"Benchmark {name} failed: {e}")
                results[name] = {"error": str(e)}
        
        return results


def print_results(results: Dict[str, Dict[str, Any]]):
    """Print throughput benchmark results."""
    print("\n" + "="*90)
    print("THROUGHPUT BENCHMARK RESULTS")
    print("="*90)
    
    header = f"{'Endpoint':<20} {'RPS':<10} {'Avg Lat':<10} {'P95 Lat':<10} {'Success':<10} {'Failed':<10}"
    print(header)
    print("-" * len(header))
    
    for endpoint, metrics in results.items():
        if "error" in metrics:
            print(f"{endpoint:<20} ERROR: {metrics['error']}")
        else:
            print(f"{endpoint:<20} {metrics['throughput_rps']:<10.2f} "
                  f"{metrics['avg_latency_ms']:<10.2f} {metrics['p95_latency_ms']:<10.2f} "
                  f"{metrics['successful_requests']:<10} {metrics['failed_requests']:<10}")
    
    print("="*90)


async def main():
    parser = argparse.ArgumentParser(description="Run throughput benchmarks")
    parser.add_argument("--server-url", type=str, default="http://localhost:8000",
                       help="Server URL")
    parser.add_argument("--total-requests", type=int, default=1000,
                       help="Total number of requests")
    parser.add_argument("--duration", type=int, default=None,
                       help="Duration in seconds (overrides total-requests)")
    parser.add_argument("--max-concurrent", type=int, default=50,
                       help="Maximum concurrent requests")
    parser.add_argument("--endpoint", type=str, default="all",
                       choices=["all", "text", "image", "rerank"],
                       help="Specific endpoint to benchmark")
    
    args = parser.parse_args()
    
    benchmark = ThroughputBenchmark(args.server_url, args.max_concurrent)
    
    if args.endpoint == "all":
        results = await benchmark.run_full_benchmark(args.total_requests, args.duration)
    elif args.endpoint == "text":
        results = {"text_embedding": await benchmark.benchmark_text_embedding(args.total_requests, 1, args.duration)}
    elif args.endpoint == "image":
        results = {"image_embedding": await benchmark.benchmark_image_embedding(args.total_requests, 1, args.duration)}
    elif args.endpoint == "rerank":
        results = {"reranking": await benchmark.benchmark_reranking(args.total_requests, 10, args.duration)}
    
    print_results(results)
    
    # Save results
    with open("throughput_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info("Results saved to throughput_results.json")


if __name__ == "__main__":
    asyncio.run(main())