# Technical Notes

## Build System

- Dependencies (SymEngine, Eigen, pybind11) are fetched via CMake FetchContent.
- SymEngine benchmarks and tests are disabled to keep MathLLM builds lean.
- Python dev headers are required for pybind11; `PYBIND11_FINDPYTHON` is enabled for modern discovery.

## Symbolic Engine

- Integration supports polynomials, sine, cosine, and exponential functions with simple product rules.
- Equation solving wraps `SymEngine::solve` and returns formatted solutions as strings.
- Equality verification simplifies the difference and checks for zero using SymEngine equality.
- Router re-stringifies SymPy expressions before delegating to SymEngine to keep the bridge simple.
- LaTeX inputs are sanitized with a deny-list (`\\input`, `\\include`, `\\write`, etc.) prior to parsing.

## MIR and Routing

- `MIRExpr` and `MIRProblem` dataclasses capture SymPy expressions, assumptions, objectives, and variables.
- Objective detection combines SymPy node inspection (`Integral`, `Derivative`, `Eq`) with raw LaTeX hints (`\\int`, `d/dx`, `=`).
- Integration results are verified by differentiating the candidate; differentiation compares to SymPy derivatives; solve validates each root with symbolic equality.

## Services and UI

- FastAPI `/solve` endpoint surfaces router responses with verification summaries and timing data.
- Gradio demo offers an interactive front-end with auto-objective detection and markdown rendering.

## Testing

- C++ tests cover 10+ calculus and solver scenarios.
- Python suite exercises router pipelines and validates ≥45 successes across 50 curated LaTeX samples in `data/examples/easy.jsonl`.

## Known Gaps

- Integration rule engine does not yet cover products such as `x e^{x}`; fallback to SymEngine integration is a future task.
- `prove` objective remains stubbed and is excluded from routing.
- Unit checking currently returns "skipped" pending Pint integration.
