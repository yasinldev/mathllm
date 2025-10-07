#!/usr/bin/env python3
import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from benchmark import load_test

async def test_batch_config(max_num_seqs: int, api_base: str):
    print(f"\n{'='*60}")
    print(f"Testing max-num-seqs={max_num_seqs}")
    print(f"{'='*60}")
    agg = await load_test(
        api_base=api_base,
        num_requests=20,
        concurrency=4,
        max_tokens=256
    )
    return agg

async def main():
    api_base = os.getenv('STUDENT_API_BASE', 'http://localhost:8009/v1')
    target_p95 = 1600
    
    print("Phase B: KV-cache & Batching Tuning")
    print(f"Student API: {api_base}")
    print(f"Target p95: ≤{target_p95}ms (batch 2-4 concurrent)")
    
    configs = [8, 10, 12]
    results = {}
    
    for max_num_seqs in configs:
        print(f"\n\nRestart Student with max-num-seqs={max_num_seqs}")
        input("Press ENTER when ready...")
        agg = await test_batch_config(max_num_seqs, api_base)
        results[max_num_seqs] = agg
    
    print("\n" + "="*60)
    print("PHASE B RESULTS SUMMARY")
    print("="*60)
    print(f"{'Config':<15} {'p50':<10} {'p90':<10} {'p95':<10} {'OOM':<8} {'Pass':<8}")
    print("-"*60)
    
    best_config = None
    best_p95 = float('inf')
    
    for max_num_seqs, agg in results.items():
        passed = agg.p95_latency <= target_p95 and agg.oom_count == 0
        status = "✓" if passed else "✗"
        print(f"seqs={max_num_seqs:<9} {agg.p50_latency:<10.1f} {agg.p90_latency:<10.1f} {agg.p95_latency:<10.1f} {agg.oom_count:<8} {status:<8}")
        
        if passed and agg.p95_latency < best_p95:
            best_config = max_num_seqs
            best_p95 = agg.p95_latency
    
    print("="*60)
    
    if best_config:
        print(f"\n✓ Best config: max-num-seqs={best_config} (p95={best_p95:.1f}ms)")
        print(f"Update STUDENT_MAX_NUM_SEQS={best_config} in .env")
        return 0
    else:
        print("\n✗ No config passed Phase B criteria")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
