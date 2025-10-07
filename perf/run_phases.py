#!/usr/bin/env python3
import sys
import os

def main():
    print("="*60)
    print("MATHLLM PERFORMANCE SPRINT - ALL PHASES")
    print("="*60)
    
    phases = [
        ("A", "smoke_test.py", "Baseline smoke test (5 queries)"),
        ("B", "phase_b_tuning.py", "KV-cache & batching tuning"),
        ("C", "phase_c_speculative.py", "Speculative decoding evaluation"),
        ("D", "phase_d_talker.py", "Talker deployment choice"),
        ("E", "phase_e_telemetry.py", "15min telemetry & guardrails"),
        ("F", "phase_f_cache.py", "Caching & throughput test")
    ]
    
    print("\nAvailable phases:")
    for phase_id, script, desc in phases:
        print(f"  {phase_id}: {desc}")
    
    print("\nPhase G: Generate perf.md report (manual)")
    print("="*60)
    
    if len(sys.argv) > 1:
        phase = sys.argv[1].upper()
        for phase_id, script, desc in phases:
            if phase == phase_id:
                print(f"\nRunning Phase {phase}: {desc}")
                os.system(f"python3 {os.path.dirname(__file__)}/{script}")
                return
        
        print(f"Unknown phase: {phase}")
        sys.exit(1)
    else:
        print("\nUsage: python3 run_phases.py <phase>")
        print("Example: python3 run_phases.py A")
        sys.exit(1)

if __name__ == "__main__":
    main()
