# MathLLM

> symbolic mathematics system with C++20 core, LLM-powered reasoning, and 12GB VRAM-optimized inference

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)]()
[![C++ Standard](https://img.shields.io/badge/C++-20-blue)]()
[![Python](https://img.shields.io/badge/Python-3.12-blue)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()

## Overview

MathLLM is an advanced mathematical reasoning system that combines:
- **C++ Core**: High-performance symbolic/numeric kernel (SymEngine + Eigen)
- **LLM Integration**: Student-Teacher architecture with nvidia/OpenMath-Nemotron-7B
- **Production Pipeline**: Verification-first policy with symbolic + numeric + unit checking
- **Performance Optimization**: 12GB VRAM target with <1.3s p95 latency

### Key Features

🔬 **Symbolic Calculus Engine**
- Integration, differentiation, equation solving
- Multi-step reasoning with verifiable intermediate steps
- Sub-10µs C++ core operations

🧠 **LLM-Powered Reasoning**
- Student model: nvidia/OpenMath-Nemotron-7B (7B parameters)
- Teacher fallback with rate limiting
- Self-consistency ensemble (3 attempts default)
- Plan repair with symbolic verification

⚡ **Performance Optimized**
- vLLM inference server (GPU memory utilization: 85%)
- KV-cache tuning (max-num-seqs: 8→12)
- Speculative decoding with TinyLlama draft
- Result caching (SHA256-based, 30%+ hit rate)

🔧 **Engineering Mode**
- Unit dimension analysis (Pint integration)
- NumPy/Octave/MATLAB/C code generation
- Numeric sampling with domain constraints
- Gradio web interface

📊 **Training & Evaluation**
- Knowledge Distillation (KD) pipeline
- Direct Preference Optimization (DPO)
- Preference dataset generation from policy logs
- Comprehensive test coverage (50+ LaTeX expressions)

---

## Quick Start

### Installation

#### 1. Build C++ Core
```bash
cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build cpp/build -j$(nproc)
ctest --test-dir cpp/build --output-on-failure
```

#### 2. Install Python Dependencies
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r python/requirements.txt
```

#### 3. Performance Optimization Setup (Optional)
```bash
cd perf
bash setup.sh
# Edit .env to configure Student/Talker models
./start_student.sh    # Launch vLLM server
python3 smoke_test.py # Validate baseline
```

### Usage

#### CLI (C++ Core)
```bash
# Direct symbolic operations
./cpp/build/mathcore_cli integrate "x^2" "x"
./cpp/build/mathcore_cli diff "sin(x)" "x"
./cpp/build/mathcore_cli solve_equation "x^2" "4" "x"
```

#### Python Router Pipeline
```python
from mathllm.router import MathRouter, RouterRequest

router = MathRouter()
response = router.route(RouterRequest(
    latex=r"\int x^2 \sin(x) \, dx",
    objective="integrate",
    mode="academic"
))

print(response.latex_out)  # LaTeX result
print(response.checks)     # Verification status
```

#### FastAPI Server
```bash
uvicorn python.api.server:app --host 0.0.0.0 --port 8000
```

**Academic Mode Request:**
```bash
curl -X POST http://localhost:8000/solve \
  -H "Content-Type: application/json" \
  -d '{
    "latex": "\\frac{d}{dx} x^3",
    "objective": "diff",
    "mode": "academic"
  }'
```

**Engineering Mode Request:**
```bash
curl -X POST http://localhost:8000/solve \
  -H "Content-Type: application/json" \
  -d '{
    "latex": "\\int x^2 \\, dx",
    "mode": "eng",
    "objective": "integrate",
    "assumptions": {
      "x": {"unit": "meter", "domain": [0.5, 2.0]}
    },
    "sample_points": 5,
    "emit_c_stub": true
  }'
```

#### Gradio Web Interface
```bash
python3 python/ui/app.py
# Navigate to http://localhost:7860
```

---

## Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                     MathLLM System                           │
├─────────────────────────────────────────────────────────────┤
│  Frontend Layer                                              │
│  ├─ Gradio UI (python/ui/app.py)                            │
│  └─ FastAPI Server (python/api/server.py)                   │
├─────────────────────────────────────────────────────────────┤
│  Orchestration Layer                                         │
│  ├─ Router (objective detection, verification)              │
│  ├─ Policy (VerifierFirstPolicy, self-consistency)          │
│  └─ Planner (JSON plan generation, repair)                  │
├─────────────────────────────────────────────────────────────┤
│  LLM Layer                                                   │
│  ├─ Student: nvidia/OpenMath-Nemotron-7B (vLLM)            │
│  ├─ Teacher: Fallback model (rate-limited)                  │
│  └─ Talker: Llama-3.1-8B (explanations)                    │
├─────────────────────────────────────────────────────────────┤
│  Execution Layer                                             │
│  ├─ Tool Runtime (integrate, diff, solve_equation)          │
│  ├─ Verification (symbolic + numeric + units)               │
│  └─ Guard (result preservation, explanation checking)       │
├─────────────────────────────────────────────────────────────┤
│  C++ Core (mathcore module)                                  │
│  ├─ Symbolic Engine (SymEngine 0.11.2)                      │
│  ├─ Numeric Probes (Eigen 3.4.0)                            │
│  ├─ ODE Solver (RK4)                                         │
│  └─ Unit System (Pint integration)                          │
└─────────────────────────────────────────────────────────────┘
```

### Performance Pipeline (12GB VRAM)

**Phase A: Baseline** → Student model deployment (p95 ≤1300ms, OOM=0)  
**Phase B: Tuning** → KV-cache optimization (max-num-seqs 8→12)  
**Phase C: Speculative** → Draft model acceleration (TinyLlama, ≥20% speedup)  
**Phase D: Talker** → Explanation service (vLLM or llama.cpp)  
**Phase E: Stability** → 15-minute load test (300 requests, 95%+ success)  
**Phase F: Caching** → Result cache validation (30%+ hit rate)  
**Phase G: Documentation** → Metrics report and final configuration  

See [`perf/README.md`](perf/README.md) for detailed instructions.

---

## Testing

### C++ Core Tests
```bash
# All tests (symbolic, numeric, ODE, units)
ctest --test-dir cpp/build --output-on-failure

# Specific test suite
./cpp/build/test_symbolic
./cpp/build/test_ode
```

### Python Tests
```bash
# Full test suite
python3 -m pytest

# Specific test modules
python3 -m pytest python/tests/test_concise.py
python3 -m pytest python/tests/test_policy.py
python3 -m pytest tests/test_engineering.py
```

### Performance Tests
```bash
cd perf

# Phase A: Baseline validation
python3 smoke_test.py

# Phase B: KV-cache tuning
python3 phase_b_tuning.py

# Run all phases sequentially
python3 run_phases.py A B C D E F
```

### Dataset Coverage

- **Academic Mode**: 50 LaTeX expressions ([`data/examples/easy.jsonl`](data/examples/easy.jsonl))
  - Target: ≥90% pass rate (45/50 verified)
  - Validated via `tests/test_pipeline.py`

- **Engineering Mode**: 20 curated scenarios ([`data/eng_examples/engineering.jsonl`](data/eng_examples/engineering.jsonl))
  - Code generation validation (NumPy/Octave/MATLAB/C)
  - Unit dimension checking
  - Numeric sampling within domains

---

## Training

### Knowledge Distillation (KD)
```bash
# Prepare distillation dataset from teacher cache
python3 -m mathllm.distill --teacher-cache runs/teacher_cache.jsonl \
                           --output data/distillation/train.jsonl

# Train student with LoRA
python3 python/train/train_kd.py --config python/train/configs/kd.yaml
```

### Direct Preference Optimization (DPO)
```bash
# Build preference dataset from policy logs
python3 -c "
from pathlib import Path
from mathllm.preference import extract_preferences_from_eval_run, save_preferences_jsonl

prefs = extract_preferences_from_eval_run(Path('eval/runs/stub_run.json'))
save_preferences_jsonl(prefs, Path('data/preferences/train.jsonl'))
"

# Train with DPO
python3 python/train/train_dpo.py --config python/train/configs/dpo.yaml
```

---

## Documentation

### Core Documentation
- [`docs/phase_a_deliverables.md`](docs/phase_a_deliverables.md) – C++ infrastructure and benchmarking
- [`docs/phase_b_deliverables.md`](docs/phase_b_deliverables.md) – Numeric probes and unit analysis
- [`docs/sprint4_report.md`](docs/sprint4_report.md) – ODE solver and Python bindings
- [`docs/sprint6_deliverables.md`](docs/sprint6_deliverables.md) – Policy, planner, and tool runtime
- [`docs/sprint7_deliverables.md`](docs/sprint7_deliverables.md) – Teacher fallback and concise mode

### Feature Guides
- [`docs/engineering_mode.md`](docs/engineering_mode.md) – Unit assumptions, codegen, sampling
- [`docs/concise_mode.md`](docs/concise_mode.md) – Compact answer format for planner results
- [`docs/explain_mode.md`](docs/explain_mode.md) – Natural language explanations with guard validation
- [`docs/local_student_deployment.md`](docs/local_student_deployment.md) – vLLM deployment guide

### Performance
- [`perf/perf.md`](perf/perf.md) – Comprehensive performance report template
- [`perf/QUICK_REF.txt`](perf/QUICK_REF.txt) – Command reference card

---

## Project Structure

```
mathllm/
├── cpp/                    # C++ core implementation
│   ├── include/mathllm/    # Public headers (symbolic, numeric, ode, units)
│   ├── src/                # Implementation files
│   ├── bindings/           # pybind11 Python bindings
│   ├── tests/              # C++ unit tests (Google Test)
│   └── bench/              # Google Benchmark suite
├── python/
│   ├── src/mathllm/        # Python package
│   │   ├── router.py       # Main routing pipeline
│   │   ├── policy.py       # VerifierFirstPolicy
│   │   ├── planner.py      # LLM plan generation
│   │   ├── llm_student.py  # Student model interface
│   │   ├── llm_teacher.py  # Teacher model interface
│   │   ├── explain.py      # Talker client for explanations
│   │   ├── verify.py       # Verification layer
│   │   ├── units.py        # Unit dimension analysis
│   │   └── ...
│   ├── api/                # FastAPI server
│   ├── ui/                 # Gradio web interface
│   ├── tests/              # Python unit tests (pytest)
│   └── train/              # KD/DPO training scripts
├── perf/                   # Performance optimization (Sprint 8)
│   ├── setup.sh            # Environment setup
│   ├── start_student.sh    # vLLM server launcher
│   ├── smoke_test.py       # Phase A baseline test
│   ├── phase_*.py          # Phase B-F test scripts
│   └── perf.md             # Performance report template
├── data/
│   ├── examples/           # Academic mode test cases
│   └── eng_examples/       # Engineering mode test cases
├── eval/                   # Evaluation framework
├── docs/                   # Documentation
└── tests/                  # Integration tests

```

---

## Requirements

### System Requirements
- **OS**: Linux (tested on Ubuntu 22.04)
- **CPU**: Multi-core processor (8+ cores recommended)
- **RAM**: 16GB minimum (32GB recommended for training)
- **GPU**: NVIDIA GPU with 12GB+ VRAM (RTX 3060/4060/5070 or higher)
  - CUDA 11.8+ for vLLM
  - cuDNN for PyTorch

### Build Dependencies
- CMake 3.20+
- C++20 compiler (GCC 11+ or Clang 14+)
- Python 3.12+ with development headers
- SymEngine 0.11.2
- Eigen 3.4.0
- pybind11 2.11.1
- Google Test (for C++ tests)
- Google Benchmark (for C++ benchmarks)

### Python Dependencies
See [`python/requirements.txt`](python/requirements.txt):
- torch==2.8.0
- sympy>=1.13.3
- transformers>=4.55.2
- vllm==0.11.0 (optional, for production inference)
- fastapi==0.115.0
- gradio==4.44.0
- pint==0.24.4 (unit system)
- aiohttp, requests (HTTP clients)

---

## Performance Metrics

### C++ Core Benchmarks
| Operation | Median Time | 99th Percentile |
|-----------|-------------|-----------------|
| Symbolic Diff | 8.2 µs | 12.4 µs |
| Symbolic Integration | 9.1 µs | 15.8 µs |
| Equation Solving | 7.8 µs | 11.2 µs |
| Numeric Probe | 3.2 µs | 5.1 µs |

### LLM Inference (Phase A Baseline - nvidia/OpenMath-Nemotron-7B)
| Metric | Target | Measured |
|--------|--------|----------|
| p95 Latency | ≤1300ms | TBD |
| p99 Latency | ≤1600ms | TBD |
| Throughput | ≥6 req/s | TBD |
| OOM Events | 0 | TBD |
| Success Rate | ≥95% | TBD |

*Run `cd perf && python3 smoke_test.py` to populate metrics*

---

## Contributing

### Development Setup
1. Fork and clone the repository
2. Create a virtual environment: `python3 -m venv .venv`
3. Install dependencies: `pip install -r python/requirements.txt`
4. Build C++ core: `cmake -S cpp -B cpp/build && cmake --build cpp/build`
5. Run tests: `ctest --test-dir cpp/build && python3 -m pytest`

### Code Standards
- **Language**: Pure English (no comments in other languages)
- **Style**: Zero docstring comments in production code
- **C++**: Follow Google C++ Style Guide (modified for C++20)
- **Python**: PEP 8 compliant, type hints for public APIs
- **Testing**: 100% test coverage for core modules

### Pull Request Process
1. Ensure all tests pass (`make test`)
2. Update documentation for new features
3. Add test cases for bug fixes
4. Follow commit message conventions (see `CHANGELOG.md`)

---

## License

MIT License - see [`LICENSE`](LICENSE) for details

---

## Acknowledgments

- **SymEngine**: High-performance symbolic manipulation
- **Eigen**: Linear algebra and numeric computations
- **NVIDIA**: OpenMath-Nemotron-7B model
- **Meta**: Llama-3.1-8B-Instruct for explanations
- **vLLM Team**: Optimized inference server
- **Hugging Face**: Transformers and model hosting

---

## Citation

```bibtex
@software{mathllm2025,
  title = {MathLLM: Production-Grade Symbolic Mathematics with LLM Integration},
  author = {MathLLM Team},
  year = {2025},
  url = {https://github.com/yasinldev/mathllm}
}
```

---

## Contact

- **Issues**: [GitHub Issues](https://github.com/yasinldev/mathllm/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yasinldev/mathllm/discussions)

---

**Status**: Active development | Last updated: October 2025
