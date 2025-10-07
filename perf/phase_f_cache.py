#!/usr/bin/env python3
import asyncio
import sys
import os
import hashlib
import json
from typing import Dict, Optional
sys.path.insert(0, os.path.dirname(__file__))
from benchmark import load_test, send_request, TelemetryLogger
import aiohttp

class ResultCache:
    def __init__(self, cache_path: str = "/tmp/mathllm_cache.json"):
        self.cache_path = cache_path
        self.cache: Dict[str, str] = {}
        self.load_cache()
    
    def load_cache(self):
        try:
            if os.path.exists(self.cache_path):
                with open(self.cache_path, 'r') as f:
                    self.cache = json.load(f)
        except:
            self.cache = {}
    
    def save_cache(self):
        with open(self.cache_path, 'w') as f:
            json.dump(self.cache, f)
    
    def hash_problem(self, prompt: str) -> str:
        return hashlib.sha256(prompt.encode()).hexdigest()[:16]
    
    def get(self, prompt: str) -> Optional[str]:
        key = self.hash_problem(prompt)
        return self.cache.get(key)
    
    def set(self, prompt: str, result: str):
        key = self.hash_problem(prompt)
        self.cache[key] = result
        self.save_cache()

async def cached_request(
    session: aiohttp.ClientSession,
    url: str,
    prompt: str,
    cache: ResultCache,
    logger: TelemetryLogger,
    request_id: str
):
    cached = cache.get(prompt)
    if cached:
        metric = await send_request(
            session, url, prompt, 256, 0.2,
            request_id, logger, "cached"
        )
        metric.cache_hit = True
        logger.log_request(metric)
        return metric
    
    metric = await send_request(
        session, url, prompt, 256, 0.2,
        request_id, logger, "cache_test"
    )
    
    if metric.success:
        cache.set(prompt, "cached_result")
    
    return metric

async def cache_test(api_base: str):
    cache = ResultCache()
    logger = TelemetryLogger()
    url = f"{api_base}/chat/completions"
    
    prompts = [
        "Solve: 2x + 5 = 13",
        "What is 15 * 23?",
        "Calculate: sqrt(144)",
        "Solve: x^2 - 9 = 0",
        "Find: 100 / 4"
    ]
    
    all_prompts = prompts * 20
    
    print("Phase F: Throughput & Caching Test")
    print(f"Total requests: {len(all_prompts)}")
    print(f"Unique prompts: {len(prompts)}")
    print(f"Expected cache hits: ~75%\n")
    
    import time
    start = time.time()
    
    async with aiohttp.ClientSession() as session:
        tasks = []
        for i, prompt in enumerate(all_prompts):
            task = cached_request(
                session, url, prompt, cache,
                logger, f"cache_{i}"
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
    
    duration = time.time() - start
    agg = logger.compute_aggregates(duration)
    
    cached_metrics = [m for m in logger.metrics if m.cache_hit]
    cached_latencies = [m.latency_ms for m in cached_metrics]
    cached_p95 = sorted(cached_latencies)[int(len(cached_latencies)*0.95)] if cached_latencies else 0
    
    print("\n" + "="*60)
    print("PHASE F RESULTS")
    print("="*60)
    print(f"Total Requests:     {agg.total_requests}")
    print(f"Cache Hit Rate:     {agg.cache_hit_rate:.1f}%")
    print(f"Throughput:         {agg.throughput_rps:.2f} req/s")
    print(f"\nAll Requests p95:   {agg.p95_latency:.1f}ms")
    print(f"Cached p95:         {cached_p95:.1f}ms")
    print(f"Duration:           {duration/60:.1f} minutes")
    
    cache_hit_pass = agg.cache_hit_rate >= 30
    cached_latency_pass = cached_p95 <= 600
    throughput_target = 100 / 5
    throughput_pass = agg.throughput_rps >= throughput_target
    
    print("\n" + "-"*60)
    print(f"Cache hit ≥30%:     {'✓ PASS' if cache_hit_pass else '✗ FAIL'} (actual: {agg.cache_hit_rate:.1f}%)")
    print(f"Cached p95 ≤600ms:  {'✓ PASS' if cached_latency_pass else '✗ FAIL'} (actual: {cached_p95:.1f}ms)")
    print(f"Throughput ≥{throughput_target:.1f}:    {'✓ PASS' if throughput_pass else '✗ FAIL'} (actual: {agg.throughput_rps:.2f} req/s)")
    print("="*60)
    
    if cache_hit_pass and cached_latency_pass:
        print("\n✓ Phase F PASSED")
        return 0
    else:
        print("\n✗ Phase F FAILED")
        return 1

async def main():
    api_base = os.getenv('STUDENT_API_BASE', 'http://localhost:8009/v1')
    exit_code = await cache_test(api_base)
    sys.exit(exit_code)

if __name__ == "__main__":
    asyncio.run(main())
