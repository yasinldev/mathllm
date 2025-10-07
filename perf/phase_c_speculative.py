#!/usr/bin/env python3
import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from benchmark import load_test

async def main():
    api_base = os.getenv('STUDENT_API_BASE', 'http://localhost:8009/v1')
    
    print("Phase C: Speculative Decoding Evaluation")
    print(f"Student API: {api_base}")
    print("\nTest 1: Baseline (no speculative decoding)")
    input("Ensure SPECULATIVE_ENABLED=false, restart Student, press ENTER...")
    
    baseline = await load_test(
        api_base=api_base,
        num_requests=30,
        concurrency=4,
        max_tokens=256
    )
    
    print("\n\nTest 2: Speculative Decoding")
    print("Set SPECULATIVE_ENABLED=true in .env")
    print("Set DRAFT_MODEL=TinyLlama/TinyLlama-1.1B-Chat-v1.0-AWQ")
    input("Restart Student with speculative decoding, press ENTER...")
    
    speculative = await load_test(
        api_base=api_base,
        num_requests=30,
        concurrency=4,
        max_tokens=256
    )
    
    print("\n" + "="*60)
    print("PHASE C COMPARISON")
    print("="*60)
    
    speedup_p95 = ((baseline.p95_latency - speculative.p95_latency) / baseline.p95_latency) * 100
    speedup_throughput = ((speculative.throughput_rps - baseline.throughput_rps) / baseline.throughput_rps) * 100
    
    print(f"{'Metric':<20} {'Baseline':<15} {'Speculative':<15} {'Change':<15}")
    print("-"*65)
    print(f"{'p50 (ms)':<20} {baseline.p50_latency:<15.1f} {speculative.p50_latency:<15.1f} {((speculative.p50_latency-baseline.p50_latency)/baseline.p50_latency*100):+.1f}%")
    print(f"{'p95 (ms)':<20} {baseline.p95_latency:<15.1f} {speculative.p95_latency:<15.1f} {speedup_p95:+.1f}%")
    print(f"{'Throughput (rps)':<20} {baseline.throughput_rps:<15.2f} {speculative.throughput_rps:<15.2f} {speedup_throughput:+.1f}%")
    print(f"{'Success Rate':<20} {baseline.successful_requests}/{baseline.total_requests:<14} {speculative.successful_requests}/{speculative.total_requests:<14}")
    print("="*65)
    
    target_speedup = 20
    passed = speedup_p95 <= -target_speedup or speedup_throughput >= target_speedup
    
    if passed:
        print(f"\n✓ Phase C PASSED: Speedup ≥{target_speedup}%")
        print("Recommendation: ENABLE speculative decoding")
        return 0
    else:
        print(f"\n✗ Phase C: Speedup <{target_speedup}%")
        print("Recommendation: DISABLE speculative decoding")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
