# MathLLM Performance Engineering Report

## Executive Summary

12GB VRAM optimization targeting nvidia/OpenMath-Nemotron-7B Student + 8B Talker.

**Key Results:**
- Student p95: XXXms (single), XXXms (batch 2-4)
- Talker p95: XXXms
- OOM Count: 0
- Throughput: XX.X req/s
- Cache Hit Rate: XX%

## Hardware Configuration

- GPU: NVIDIA RTX 5070 (12GB VRAM)
- CUDA Version: X.X
- vLLM Version: X.X.X
- System: Linux

## Phase A: Baseline

### Configuration
```bash
Model: nvidia/OpenMath-Nemotron-7B
Port: 8009
gpu-memory-utilization: 0.85
max-model-len: 4096
max-num-seqs: 8
swap-space: 8GB
enforce-eager: true
```

### Results
| Metric | Value |
|--------|-------|
| p50 latency | XXX ms |
| p90 latency | XXX ms |
| p95 latency | XXX ms |
| p99 latency | XXX ms |
| OOM count | 0 |
| Success rate | XX% |

**Status:** ✓ PASS / ✗ FAIL

## Phase B: KV-Cache & Batching Tuning

### Tested Configurations

| max-num-seqs | p50 (ms) | p95 (ms) | OOM | Pass |
|--------------|----------|----------|-----|------|
| 8            | XXX      | XXX      | 0   | ✓/✗  |
| 10           | XXX      | XXX      | 0   | ✓/✗  |
| 12           | XXX      | XXX      | 0   | ✓/✗  |

**Optimal Config:** max-num-seqs=X
**Status:** ✓ PASS / ✗ FAIL

## Phase C: Speculative Decoding

### Configuration
```bash
Draft Model: TinyLlama/TinyLlama-1.1B-Chat-v1.0-AWQ
speculative-model: enabled/disabled
```

### Comparison

| Metric | Baseline | Speculative | Speedup |
|--------|----------|-------------|---------|
| p50 (ms) | XXX | XXX | +XX% |
| p95 (ms) | XXX | XXX | +XX% |
| Throughput (rps) | XX.X | XX.X | +XX% |

**Decision:** ENABLED / DISABLED
**Reason:** Speedup XX% (target: ≥20%)
**Status:** ✓ PASS / ✗ FAIL

## Phase D: Talker Deployment

### Option A: vLLM Sequential

| Metric | Value |
|--------|-------|
| p50 (ms) | XXX |
| p95 (ms) | XXX |
| Target | ≤1200ms |
| Pass | ✓/✗ |

### Option B: llama.cpp CPU Offload

```bash
Mode: llamacpp
GPU Layers (-ngl): XX
CPU Threads: 8
```

| Metric | Value |
|--------|-------|
| p50 (ms) | XXX |
| p95 (ms) | XXX |
| Target | ≤900ms |
| Pass | ✓/✗ |

**Selected Option:** A / B
**Reason:** Lower latency / Better resource usage
**Status:** ✓ PASS / ✗ FAIL

## Phase E: Telemetry & Guardrails

### 15-Minute Load Test

| Metric | Value |
|--------|-------|
| Total requests | XXXX |
| Success rate | XX% |
| OOM count | 0 |
| p50 latency | XXX ms |
| p95 latency | XXX ms |
| Throughput | XX.X req/s |

**Telemetry Log:** /tmp/mathllm_telemetry.jsonl
**Status:** ✓ PASS / ✗ FAIL

## Phase F: Caching & Throughput

### Cache Performance

| Metric | Value | Target |
|--------|-------|--------|
| Cache hit rate | XX% | ≥30% |
| Cached p95 | XXX ms | ≤600ms |
| Total requests | XXX | 100 |
| Duration | X.X min | 5 min |
| Throughput | XX req/s | ≥20 req/s |

**Status:** ✓ PASS / ✗ FAIL

## Final Configuration

### Student Server
```bash
MODEL=nvidia/OpenMath-Nemotron-7B
PORT=8009
GPU_MEMORY_UTILIZATION=0.85
MAX_MODEL_LEN=4096
MAX_NUM_SEQS=X
SWAP_SPACE=8
SPECULATIVE_ENABLED=true/false
DRAFT_MODEL=TinyLlama/TinyLlama-1.1B-Chat-v1.0-AWQ
```

### Talker Server
```bash
MODE=vllm/llamacpp
MODEL=meta-llama/Llama-3.1-8B-Instruct
PORT=8010
MAX_NUM_SEQS=6
NGL=XX (if llamacpp)
```

## Performance Summary

### Before Optimization
- Student p95: XXXms
- Throughput: XX req/s
- OOM events: X

### After Optimization
- Student p95: XXXms (-XX%)
- Throughput: XX req/s (+XX%)
- OOM events: 0 (-100%)

## Resource Utilization

### VRAM Usage
- Student: ~XX GB
- Talker (if vLLM): ~XX GB
- Peak: XX GB / 12 GB (XX%)

### GPU Utilization
- Average: XX%
- Peak: XX%

### CPU Usage
- Average: XX%
- Peak: XX%

## Known Limitations

1. **Long Context Impact**: Requests >2048 tokens increase p95 by XX%
2. **Batch Size Ceiling**: max-num-seqs >X causes OOM risk
3. **Speculative Decoding**: XX% quality/acceptance tradeoff
4. **Concurrent Load**: vLLM sequential = Student OR Talker (not both)

## Risk Mitigations

1. **OOM Prevention**: swap-space=8GB, max-num-seqs capped at X
2. **Latency SLA**: enforce-eager mode, optimized batch size
3. **Cache Invalidation**: TTL=3600s, hash-based keys
4. **Monitoring**: JSONL telemetry, real-time metrics

## Recommendations

1. **Production Settings**: Use final config from Phase B/C/D
2. **Monitoring**: Track OOM, p95, cache hit rate
3. **Scaling**: Consider GPU upgrade for concurrent Student+Talker
4. **Caching**: Enable for repeated problem patterns (XX% hit rate)

## Graphs

[TODO: Add graphs]
- Latency distribution (p50/p90/p95/p99)
- Throughput over time
- VRAM usage timeline
- Cache hit rate progression

## Appendix

### Start Commands

**Student:**
```bash
cd perf
./start_student.sh
```

**Talker (vLLM):**
```bash
TALKER_MODE=vllm ./start_talker.sh
```

**Talker (llama.cpp):**
```bash
TALKER_MODE=llamacpp ./start_talker.sh
```

### Test Commands

```bash
python3 healthcheck.py http://localhost:8009/v1
python3 smoke_test.py
python3 phase_b_tuning.py
python3 phase_c_speculative.py
python3 phase_d_talker.py
python3 phase_e_telemetry.py
python3 phase_f_cache.py
```

### Environment Variables

See `.env.example` for complete configuration.

## Conclusion

Performance optimization achieved:
- ✓/✗ Student p95 ≤1300ms (single)
- ✓/✗ Student p95 ≤1600ms (batch)
- ✓/✗ Talker p95 ≤900-1200ms
- ✓/✗ OOM = 0
- ✓/✗ Cache hit ≥30%

**Overall Grade:** X/7 phases passed

---

*Generated: [DATE]*
*Engineer: Performance Sprint Team*
*GPU: RTX 5070 12GB*
