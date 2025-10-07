import asyncio
import json
import time
from datetime import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
import aiohttp
import statistics
import os

@dataclass
class RequestMetrics:
    timestamp: str
    request_id: str
    phase: str
    latency_ms: float
    input_length: int
    output_length: int
    num_seqs: int
    success: bool
    error: Optional[str] = None
    cache_hit: bool = False

@dataclass
class AggregateMetrics:
    total_requests: int
    successful_requests: int
    failed_requests: int
    oom_count: int
    p50_latency: float
    p90_latency: float
    p95_latency: float
    p99_latency: float
    mean_latency: float
    throughput_rps: float
    cache_hit_rate: float
    mean_input_length: float
    mean_output_length: float

class TelemetryLogger:
    def __init__(self, log_path: str = "/tmp/mathllm_telemetry.jsonl"):
        self.log_path = log_path
        self.metrics: List[RequestMetrics] = []
        
    def log_request(self, metric: RequestMetrics):
        self.metrics.append(metric)
        with open(self.log_path, 'a') as f:
            f.write(json.dumps(asdict(metric)) + '\n')
    
    def compute_aggregates(self, duration_seconds: float) -> AggregateMetrics:
        if not self.metrics:
            return AggregateMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        
        successful = [m for m in self.metrics if m.success]
        failed = [m for m in self.metrics if not m.success]
        oom = [m for m in failed if m.error and 'oom' in m.error.lower()]
        cache_hits = [m for m in successful if m.cache_hit]
        
        latencies = [m.latency_ms for m in successful]
        latencies.sort()
        
        def percentile(data, p):
            if not data:
                return 0
            k = (len(data) - 1) * p / 100
            f = int(k)
            c = f + 1 if f + 1 < len(data) else f
            return data[f] + (k - f) * (data[c] - data[f])
        
        return AggregateMetrics(
            total_requests=len(self.metrics),
            successful_requests=len(successful),
            failed_requests=len(failed),
            oom_count=len(oom),
            p50_latency=percentile(latencies, 50),
            p90_latency=percentile(latencies, 90),
            p95_latency=percentile(latencies, 95),
            p99_latency=percentile(latencies, 99),
            mean_latency=statistics.mean(latencies) if latencies else 0,
            throughput_rps=len(successful) / duration_seconds if duration_seconds > 0 else 0,
            cache_hit_rate=len(cache_hits) / len(successful) * 100 if successful else 0,
            mean_input_length=statistics.mean([m.input_length for m in successful]) if successful else 0,
            mean_output_length=statistics.mean([m.output_length for m in successful]) if successful else 0
        )
    
    def print_summary(self, duration_seconds: float):
        agg = self.compute_aggregates(duration_seconds)
        print("\n" + "="*60)
        print("TELEMETRY SUMMARY")
        print("="*60)
        print(f"Total Requests:     {agg.total_requests}")
        print(f"Successful:         {agg.successful_requests}")
        print(f"Failed:             {agg.failed_requests}")
        print(f"OOM Count:          {agg.oom_count}")
        print(f"\nLatency (ms):")
        print(f"  p50:              {agg.p50_latency:.1f}")
        print(f"  p90:              {agg.p90_latency:.1f}")
        print(f"  p95:              {agg.p95_latency:.1f}")
        print(f"  p99:              {agg.p99_latency:.1f}")
        print(f"  mean:             {agg.mean_latency:.1f}")
        print(f"\nThroughput:         {agg.throughput_rps:.2f} req/s")
        print(f"Cache Hit Rate:     {agg.cache_hit_rate:.1f}%")
        print(f"\nAvg Input Length:   {agg.mean_input_length:.0f} tokens")
        print(f"Avg Output Length:  {agg.mean_output_length:.0f} tokens")
        print("="*60)
        
        return agg

async def send_request(
    session: aiohttp.ClientSession,
    url: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    request_id: str,
    logger: TelemetryLogger,
    phase: str = "student"
) -> RequestMetrics:
    start_time = time.time()
    
    payload = {
        "model": "default",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    
    try:
        async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=30)) as response:
            result = await response.json()
            latency_ms = (time.time() - start_time) * 1000
            
            if response.status == 200:
                choice = result.get('choices', [{}])[0]
                content = choice.get('message', {}).get('content', '')
                usage = result.get('usage', {})
                
                metric = RequestMetrics(
                    timestamp=datetime.now().isoformat(),
                    request_id=request_id,
                    phase=phase,
                    latency_ms=latency_ms,
                    input_length=usage.get('prompt_tokens', len(prompt.split())),
                    output_length=usage.get('completion_tokens', len(content.split())),
                    num_seqs=1,
                    success=True,
                    cache_hit=False
                )
            else:
                error_msg = result.get('error', {}).get('message', f'HTTP {response.status}')
                metric = RequestMetrics(
                    timestamp=datetime.now().isoformat(),
                    request_id=request_id,
                    phase=phase,
                    latency_ms=latency_ms,
                    input_length=0,
                    output_length=0,
                    num_seqs=1,
                    success=False,
                    error=error_msg
                )
                
    except asyncio.TimeoutError:
        latency_ms = (time.time() - start_time) * 1000
        metric = RequestMetrics(
            timestamp=datetime.now().isoformat(),
            request_id=request_id,
            phase=phase,
            latency_ms=latency_ms,
            input_length=0,
            output_length=0,
            num_seqs=1,
            success=False,
            error="timeout"
        )
    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        metric = RequestMetrics(
            timestamp=datetime.now().isoformat(),
            request_id=request_id,
            phase=phase,
            latency_ms=latency_ms,
            input_length=0,
            output_length=0,
            num_seqs=1,
            success=False,
            error=str(e)
        )
    
    logger.log_request(metric)
    return metric

async def smoke_test(api_base: str, num_requests: int = 5):
    logger = TelemetryLogger()
    url = f"{api_base}/chat/completions"
    
    test_prompts = [
        "Solve: 2x + 5 = 13",
        "Calculate the derivative of x^2 + 3x",
        "What is the integral of sin(x)?",
        "Solve the equation: x^2 - 4 = 0",
        "Simplify: (x+1)(x-1)"
    ]
    
    print(f"\n{'='*60}")
    print(f"SMOKE TEST: {num_requests} requests to {api_base}")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    
    async with aiohttp.ClientSession() as session:
        tasks = []
        for i in range(num_requests):
            prompt = test_prompts[i % len(test_prompts)]
            task = send_request(
                session, url, prompt, 
                max_tokens=256, 
                temperature=0.2,
                request_id=f"smoke_{i}",
                logger=logger,
                phase="smoke_test"
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
    
    duration = time.time() - start_time
    agg = logger.print_summary(duration)
    
    return agg

async def load_test(
    api_base: str, 
    num_requests: int = 100,
    concurrency: int = 4,
    max_tokens: int = 256
):
    logger = TelemetryLogger()
    url = f"{api_base}/chat/completions"
    
    test_prompts = [
        "Solve: 3x - 7 = 20",
        "Find derivative: x^3 + 2x^2 - x + 1",
        "Integrate: cos(x) dx",
        "Solve: x^2 + 5x + 6 = 0",
        "Expand: (x+2)(x+3)",
        "Factor: x^2 - 9",
        "Solve: 2x/3 + 4 = 10",
        "Differentiate: e^x * sin(x)",
        "Solve system: x+y=5, x-y=1",
        "Simplify: sqrt(18)"
    ]
    
    print(f"\n{'='*60}")
    print(f"LOAD TEST: {num_requests} requests, concurrency={concurrency}")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    
    async with aiohttp.ClientSession() as session:
        semaphore = asyncio.Semaphore(concurrency)
        
        async def bounded_request(i):
            async with semaphore:
                prompt = test_prompts[i % len(test_prompts)]
                return await send_request(
                    session, url, prompt,
                    max_tokens=max_tokens,
                    temperature=0.2,
                    request_id=f"load_{i}",
                    logger=logger,
                    phase="load_test"
                )
        
        tasks = [bounded_request(i) for i in range(num_requests)]
        results = await asyncio.gather(*tasks)
    
    duration = time.time() - start_time
    agg = logger.print_summary(duration)
    
    return agg

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python benchmark.py [smoke|load] [api_base]")
        print("Example: python benchmark.py smoke http://localhost:8009/v1")
        sys.exit(1)
    
    test_type = sys.argv[1]
    api_base = sys.argv[2] if len(sys.argv) > 2 else "http://localhost:8009/v1"
    
    if test_type == "smoke":
        asyncio.run(smoke_test(api_base, num_requests=5))
    elif test_type == "load":
        asyncio.run(load_test(api_base, num_requests=100, concurrency=4))
    else:
        print(f"Unknown test type: {test_type}")
        sys.exit(1)
