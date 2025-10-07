# MathLLM Performance Sprint

Production performance optimization for 12GB VRAM (RTX 5070).

## Quick Start

```bash
cd perf
bash setup.sh
./start_student.sh
```

In another terminal:
```bash
python3 healthcheck.py http://localhost:8009/v1
python3 smoke_test.py
```

## Phase A: Baseline
Student: nvidia/OpenMath-Nemotron-7B
Port: 8009
Target: p95 ≤1300ms, OOM=0

## Phase B: KV-cache Tuning
Test max-num-seqs: 8→10→12
Target: batch p95 ≤1600ms

## Phase C: Speculative Decoding
Draft: TinyLlama-1.1B
Target: ≥20% speedup

## Phase D: Talker Deployment
Option A: vLLM sequential (p95 ≤1200ms)
Option B: llama.cpp -ngl (p95 ≤900ms)

## Phase E: Telemetry
JSONL logging
OOM guardrails
15min load test

## Phase F: Caching
Result cache by hash
Target: hit rate ≥30%, cached p95 ≤600ms

## Phase G: Documentation
perf.md with metrics
Final configuration

## Test Scripts

```bash
python3 smoke_test.py
python3 phase_b_tuning.py
python3 phase_c_speculative.py
python3 phase_d_talker.py
python3 phase_e_telemetry.py
python3 phase_f_cache.py
```
