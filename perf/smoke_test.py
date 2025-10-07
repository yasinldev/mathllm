#!/usr/bin/env python3
import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from benchmark import smoke_test

async def main():
    api_base = os.getenv('STUDENT_API_BASE', 'http://localhost:8009/v1')
    target_p95 = 1300  # ms
    
    print(f"Student API: {api_base}")
    print(f"Target p95: ≤{target_p95}ms\n")
    
    agg = await smoke_test(api_base, num_requests=5)
    
    # Validation
    print("\n" + "="*60)
    print("PHASE A VALIDATION")
    print("="*60)
    
    oom_pass = agg.oom_count == 0
    p95_pass = agg.p95_latency <= target_p95
    success_pass = agg.successful_requests >= 4  # At least 4/5 must succeed
    
    print(f"OOM Count = 0:          {'✓ PASS' if oom_pass else '✗ FAIL'} (actual: {agg.oom_count})")
    print(f"p95 ≤ {target_p95}ms:          {'✓ PASS' if p95_pass else '✗ FAIL'} (actual: {agg.p95_latency:.1f}ms)")
    print(f"Success Rate ≥ 80%:     {'✓ PASS' if success_pass else '✗ FAIL'} (actual: {agg.successful_requests}/{agg.total_requests})")
    
    all_pass = oom_pass and p95_pass and success_pass
    
    if all_pass:
        print("\n🎉 Phase A: BASELINE PASSED")
        print("="*60)
        return 0
    else:
        print("\n❌ Phase A: BASELINE FAILED")
        print("="*60)
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
