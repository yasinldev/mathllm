#!/usr/bin/env python3
import asyncio
import sys
import os
import time
sys.path.insert(0, os.path.dirname(__file__))
from benchmark import load_test

async def long_load_test(api_base: str, duration_minutes: int = 15):
    duration_seconds = duration_minutes * 60
    num_requests = int(duration_minutes * 20)
    
    print(f"Running {duration_minutes}min load test...")
    print(f"Total requests: {num_requests}")
    print(f"Concurrency: 4")
    
    agg = await load_test(
        api_base=api_base,
        num_requests=num_requests,
        concurrency=4,
        max_tokens=256
    )
    
    return agg

async def main():
    api_base = os.getenv('STUDENT_API_BASE', 'http://localhost:8009/v1')
    
    print("Phase E: Telemetry & Guardrails")
    print(f"Student API: {api_base}")
    print("\nEnsure:")
    print("1. Telemetry logging enabled")
    print("2. OOM guardrails active")
    print("3. Final Student config from Phase B/C")
    
    input("\nPress ENTER to start 15min load test...")
    
    start = time.time()
    agg = await long_load_test(api_base, duration_minutes=15)
    duration = time.time() - start
    
    print("\n" + "="*60)
    print("PHASE E VALIDATION")
    print("="*60)
    
    oom_pass = agg.oom_count == 0
    success_rate = (agg.successful_requests / agg.total_requests * 100) if agg.total_requests > 0 else 0
    stability_pass = success_rate >= 95
    
    print(f"Duration:           {duration/60:.1f} minutes")
    print(f"Total Requests:     {agg.total_requests}")
    print(f"Success Rate:       {success_rate:.1f}%")
    print(f"OOM Count:          {agg.oom_count}")
    print(f"\nLatency (ms):")
    print(f"  p50:              {agg.p50_latency:.1f}")
    print(f"  p95:              {agg.p95_latency:.1f}")
    print(f"  p99:              {agg.p99_latency:.1f}")
    print(f"\nThroughput:         {agg.throughput_rps:.2f} req/s")
    print(f"\nTelemetry Log:      {os.getenv('TELEMETRY_LOG_PATH', '/tmp/mathllm_telemetry.jsonl')}")
    
    print("\n" + "-"*60)
    print(f"OOM = 0:            {'✓ PASS' if oom_pass else '✗ FAIL'}")
    print(f"Success Rate ≥95%:  {'✓ PASS' if stability_pass else '✗ FAIL'}")
    print("="*60)
    
    if oom_pass and stability_pass:
        print("\n✓ Phase E PASSED: System stable for 15min")
        return 0
    else:
        print("\n✗ Phase E FAILED")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
