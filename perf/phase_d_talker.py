#!/usr/bin/env python3
import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from benchmark import load_test

async def main():
    student_base = os.getenv('STUDENT_API_BASE', 'http://localhost:8009/v1')
    talker_base = os.getenv('TALKER_API_BASE', 'http://localhost:8010/v1')
    
    print("Phase D: Talker Deployment Choice")
    print("\nOption A: vLLM Sequential (GPU)")
    print("Stop Student, start Talker with MODE=vllm")
    print("\nOption B: llama.cpp (CPU + GPU offload)")
    print("Keep Student running, start Talker with MODE=llamacpp")
    
    choice = input("\nTest which option? (A/B/both): ").strip().upper()
    
    results = {}
    
    if choice in ['A', 'BOTH']:
        print("\n" + "="*60)
        print("Testing Option A: vLLM Sequential")
        print("="*60)
        input("Stop Student, start Talker (MODE=vllm), press ENTER...")
        
        agg_a = await load_test(
            api_base=talker_base,
            num_requests=20,
            concurrency=2,
            max_tokens=128
        )
        results['vllm'] = agg_a
    
    if choice in ['B', 'BOTH']:
        print("\n" + "="*60)
        print("Testing Option B: llama.cpp")
        print("="*60)
        print("Start Talker with MODE=llamacpp")
        print("Adjust TALKER_NGL (GPU layers: 10-24)")
        input("Press ENTER when ready...")
        
        agg_b = await load_test(
            api_base=talker_base,
            num_requests=20,
            concurrency=2,
            max_tokens=128
        )
        results['llamacpp'] = agg_b
    
    print("\n" + "="*60)
    print("PHASE D RESULTS")
    print("="*60)
    print(f"{'Option':<15} {'p50':<10} {'p95':<10} {'Target':<15} {'Pass':<8}")
    print("-"*60)
    
    best_option = None
    best_p95 = float('inf')
    
    for option, agg in results.items():
        target = 1200 if option == 'vllm' else 900
        passed = agg.p95_latency <= target and agg.oom_count == 0
        status = "✓" if passed else "✗"
        print(f"{option:<15} {agg.p50_latency:<10.1f} {agg.p95_latency:<10.1f} ≤{target}ms{'':<7} {status:<8}")
        
        if passed and agg.p95_latency < best_p95:
            best_option = option
            best_p95 = agg.p95_latency
    
    print("="*60)
    
    if best_option:
        print(f"\n✓ Recommendation: Use {best_option} (p95={best_p95:.1f}ms)")
        print(f"Update TALKER_MODE={best_option} in .env")
        return 0
    else:
        print("\n✗ No option passed Phase D criteria")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
