# MathLLM Core

MathLLM Core provides a C++20 symbolic and numeric kernel backed by SymEngine and Eigen with Python bindings powered by pybind11.

## Features

- Symbolic calculus APIs: `integrate`, `diff`, `solve_equation`, `verify_equal`
- Python bindings exposed as the `mathcore` module
- Command line utility `mathcore_cli` for quick experiments
- Unit tests in both C++ (ctest) and Python

## Requirements

- CMake 3.20+
- C++20 compiler (tested with GCC 14)
- Python 3.12 with development headers
- Python packages listed in `python/requirements.txt`

## Build

```
cmake -S cpp -B cpp/build
cmake --build cpp/build
```

Install Python dependencies in your virtual environment:

```
pip install -r python/requirements.txt
```

## CLI Usage

```
./cpp/build/mathcore_cli diff "x^2" "x"
```

Supported commands:

- `integrate <expr> <var>`
- `diff <expr> <var>`
- `solve_equation <lhs> <rhs> <var>`
- `verify_equal <lhs> <rhs>`

## Python Usage

```
python3 -c "import sys; sys.path.insert(0, 'cpp/build'); import mathcore; print(mathcore.diff('x^2', 'x'))"
```

### Router Pipeline

```
python3 -m python.src.mathllm.router
```

### FastAPI Service

```
uvicorn python.api.server:app --reload
```

Sample request:

```
curl -X POST http://127.0.0.1:8000/solve \
	-H "Content-Type: application/json" \
	-d '{"latex":"\\int x^{2}\\,dx","mode":"academic","objective":"integrate"}'
```

Engineering-mode request:

```
curl -X POST http://127.0.0.1:8000/solve \
	-H "Content-Type: application/json" \
	-d '{"latex":"\\int x^{2}\\,dx","mode":"eng","objective":"integrate","assumptions":{"x":{"unit":"meter","domain":[0.4,1.2]}},"sample_points":3,"emit_c_stub":true}'
```

### Gradio Demo

```
python3 python/ui/app.py
```

Switch the mode toggle to `eng` to unlock engineering controls (assumptions JSON, sampling count, optional C stub) and view NumPy/Octave/MATLAB previews, unit summaries, and sampled evaluations.

## Tests

### C++

```
ctest --test-dir cpp/build
```

### Python

```
python3 -m pytest
```

Run only the engineering regression suite:

```
python3 -m pytest tests/test_engineering.py
```

### Dataset

Sprint 2 ships a dataset of 50 LaTeX expressions in `data/examples/easy.jsonl`. The pipeline must successfully verify at least 45 entries; coverage is checked via `tests/test_pipeline.py`.

Engineering mode adds 20 curated scenarios in `data/eng_examples/engineering.jsonl`. The dataset powers `tests/test_engineering.py`, which asserts successful routing, code generation previews, numeric sampling, and unit status reporting for each entry.

See `docs/engineering_mode.md` for request/response anatomy and troubleshooting tips.

## Documentation

- `docs/engineering_mode.md` – in-depth guide to assumptions, codegen payloads, dataset structure, and validation workflows for engineering mode.
