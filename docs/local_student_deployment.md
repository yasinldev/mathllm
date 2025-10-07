# Local Student LLM Deployment

## Overview

MathLLM supports running a **local student LLM** via an OpenAI-compatible endpoint (e.g., vLLM, Text-Generation-Inference, or llama.cpp server). The student model generates solution plans and is supervised by the verifier-first policy.

---

## Quick Start

### 1. Install vLLM

```bash
pip install vllm
```

### 2. Download Model

We recommend **nvidia/OpenMath-Nemotron-7B-AWQ** for production use:

```bash
huggingface-cli download nvidia/OpenMath-Nemotron-7B-AWQ --local-dir ./models/openmath-nemotron-7b-awq
```

### 3. Launch vLLM Server

```bash
vllm serve nvidia/OpenMath-Nemotron-7B-AWQ \
  --host 0.0.0.0 \
  --port 8001 \
  --quantization awq \
  --dtype float16 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.85 \
  --api-key "local-dev-key"
```

**Key parameters:**

- `--quantization awq`: Reduces memory footprint (~4GB VRAM for 7B AWQ model)
- `--max-model-len 4096`: Sufficient for most math problems
- `--gpu-memory-utilization 0.85`: Adjust based on GPU capacity
- `--api-key`: Optional bearer token for endpoint security

### 4. Configure Environment

Copy the example env file:

```bash
cp env/student.example.env env/student.env
```

Edit `env/student.env`:

```bash
STUDENT_MODE=api
STUDENT_API_BASE=http://localhost:8001/v1
STUDENT_API_KEY=local-dev-key
STUDENT_MODEL_NAME=nvidia/OpenMath-Nemotron-7B-AWQ
STUDENT_TIMEOUT_SECONDS=30
```

### 5. Verify Connection

```bash
curl http://localhost:8001/v1/models \
  -H "Authorization: Bearer local-dev-key"
```

Expected response:

```json
{
  "object": "list",
  "data": [
    {
      "id": "nvidia/OpenMath-Nemotron-7B-AWQ",
      "object": "model",
      "owned_by": "vllm"
    }
  ]
}
```

---

## Environment Variables

### Required

- `STUDENT_MODE=api`: Enable API-based student endpoint
- `STUDENT_API_BASE`: OpenAI-compatible base URL (e.g., `http://localhost:8001/v1`)
- `STUDENT_MODEL_NAME`: Model identifier (must match vLLM server)

### Optional

- `STUDENT_API_KEY`: Bearer token for authentication (default: none)
- `STUDENT_TIMEOUT_SECONDS`: Request timeout (default: `30`)
- `STUDENT_MAX_TOKENS`: Max output tokens per request (default: `2048`)
- `STUDENT_TEMPERATURE`: Sampling temperature (default: `0.0` for deterministic outputs)

---

## Supported Frameworks

### vLLM (Recommended)

- **Pros**: Fast, efficient batching, OpenAI-compatible API
- **Cons**: Requires GPU with adequate VRAM
- **Installation**: `pip install vllm`

### Text-Generation-Inference (TGI)

```bash
docker run --gpus all --shm-size 1g -p 8001:80 \
  ghcr.io/huggingface/text-generation-inference:latest \
  --model-id nvidia/OpenMath-Nemotron-7B-AWQ \
  --quantize awq \
  --max-input-length 3072 \
  --max-total-tokens 4096
```

### llama.cpp Server

```bash
./server -m ./models/openmath-nemotron-7b-awq.gguf \
  --host 0.0.0.0 \
  --port 8001 \
  --ctx-size 4096 \
  --threads 8
```

---

## Model Selection

### Recommended Models

| Model | Size | Quantization | VRAM | Latency (P50) | Accuracy |
|-------|------|--------------|------|---------------|----------|
| **nvidia/OpenMath-Nemotron-7B-AWQ** | 7B | AWQ (4-bit) | ~4GB | ~150ms | ★★★★☆ |
| DeepSeek-Math-7B-RL | 7B | AWQ | ~4GB | ~180ms | ★★★★☆ |
| Qwen2.5-Math-7B-Instruct | 7B | AWQ | ~4GB | ~160ms | ★★★★☆ |
| Meta-Llama-3.1-8B-Instruct | 8B | AWQ | ~5GB | ~200ms | ★★★☆☆ |

### Evaluation Notes

- **Accuracy**: Based on internal tests (GSM8K, MATH, GPQA-Diamond subsets)
- **Latency**: P50 inference time on RTX 3090 (batch size 1, max 2048 tokens)
- **VRAM**: Approximate GPU memory usage (includes model + KV cache overhead)

---

## Prompt Engineering

The student LLM receives a structured prompt:

```
<problem>
Objective: integrate
Expression: \\int x^2 \\sin x\\,dx
Variables: [x]
</problem>

Generate a step-by-step solution plan in JSON format:
{
  "steps": [
    {"type": "tool", "tool": "integrate", "args": {"expr": "x**2 * sin(x)", "var": "x"}},
    {"type": "final", "payload": {"result": "R1"}}
  ]
}
```

**Key constraints:**

- No assumptions about solution correctness (verifier-first policy handles validation)
- Plans must be valid JSON with `steps` array
- Each step includes `type` (tool/final) and relevant payload

---

## Performance Tuning

### Batch Processing

For multiple problems, enable batch inference in vLLM:

```bash
vllm serve ... --max-num-seqs 8 --max-num-batched-tokens 8192
```

### KV Cache Optimization

Adjust cache settings based on problem complexity:

```bash
vllm serve ... --max-model-len 4096 --block-size 16 --swap-space 8
```

### Temperature Scheduling

For multi-attempt scenarios, use temperature scaling:

```python
StudentConfig(
    temperature=0.0,  # First attempt (deterministic)
    temperature=0.3,  # Fallback attempts (creative)
)
```

---

## Monitoring

Track student LLM health:

```bash
# vLLM metrics endpoint
curl http://localhost:8001/metrics
```

Key metrics:

- `vllm:request_success`: Total successful requests
- `vllm:time_to_first_token_seconds`: TTFT latency
- `vllm:time_per_output_token_seconds`: Generation speed
- `vllm:gpu_cache_usage_perc`: KV cache utilization

---

## Fallback Strategy

If local endpoint fails, configure teacher fallback:

```bash
# In router.py or policy config
TEACHER_MODE=openai
TEACHER_MODEL=gpt-4o-mini
```

The policy will attempt student first, then escalate to teacher on repeated failures.

---

## Security

### API Key Rotation

```bash
# Generate secure key
openssl rand -base64 32

# Update env
STUDENT_API_KEY=<new_key>

# Restart vLLM
vllm serve ... --api-key <new_key>
```

### Network Isolation

For production deployments:

1. Bind vLLM to internal network only (`--host 127.0.0.1`)
2. Use reverse proxy (nginx/Caddy) with rate limiting
3. Enable HTTPS with valid TLS certificates

---

## Troubleshooting

### Connection Refused

```bash
# Check vLLM server status
curl http://localhost:8001/health

# Verify port binding
netstat -tuln | grep 8001
```

### Out of Memory

Reduce batch size or model context:

```bash
vllm serve ... --max-model-len 2048 --max-num-seqs 4
```

### Timeout Errors

Increase timeout in `env/student.env`:

```bash
STUDENT_TIMEOUT_SECONDS=60
```

---

## Cost & Latency Comparison

| Mode | Latency (P50) | Latency (P95) | Cost per 1K requests | Notes |
|------|---------------|---------------|----------------------|-------|
| **Local (7B AWQ)** | ~150ms | ~300ms | $0 (compute only) | Requires GPU |
| OpenAI GPT-4o-mini | ~800ms | ~1500ms | ~$0.60 | Rate limits apply |
| Anthropic Claude Haiku | ~600ms | ~1200ms | ~$1.25 | Best accuracy |

**Recommendation**: Use local student for latency-sensitive applications; fallback to teacher for complex problems.

---

## References

- [vLLM Documentation](https://docs.vllm.ai/)
- [OpenMath-Nemotron Model Card](https://huggingface.co/nvidia/OpenMath-Nemotron-7B-AWQ)
- `python/src/mathllm/llm_student.py`: Student LLM implementation
- `env/student.example.env`: Configuration template
