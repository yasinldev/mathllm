#include <benchmark/benchmark.h>
#include "mathllm/symbolic.h"

static void BM_Integrate_Simple(benchmark::State& state) {
    for (auto _ : state) {
        std::string result = mathllm::integrate("x", "x");
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_Integrate_Simple);

static void BM_Integrate_Polynomial(benchmark::State& state) {
    for (auto _ : state) {
        std::string result = mathllm::integrate("x^2 + 2*x + 1", "x");
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_Integrate_Polynomial);

static void BM_Integrate_Trig(benchmark::State& state) {
    for (auto _ : state) {
        std::string result = mathllm::integrate("sin(x) + cos(x)", "x");
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_Integrate_Trig);

static void BM_Diff_Simple(benchmark::State& state) {
    for (auto _ : state) {
        std::string result = mathllm::diff("x^2", "x");
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_Diff_Simple);

static void BM_Diff_Polynomial(benchmark::State& state) {
    for (auto _ : state) {
        std::string result = mathllm::diff("x^3 + 2*x^2 + 3*x + 4", "x");
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_Diff_Polynomial);

static void BM_Solve_Linear(benchmark::State& state) {
    for (auto _ : state) {
        std::string result = mathllm::solve_equation("2*x + 1", "5", "x");
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_Solve_Linear);

static void BM_Solve_Quadratic(benchmark::State& state) {
    for (auto _ : state) {
        std::string result = mathllm::solve_equation("x^2", "4", "x");
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_Solve_Quadratic);

static void BM_Verify_Simple(benchmark::State& state) {
    for (auto _ : state) {
        bool result = mathllm::verify_equal("x + x", "2*x", 1000.0);
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_Verify_Simple);

static void BM_Verify_Polynomial(benchmark::State& state) {
    for (auto _ : state) {
        bool result = mathllm::verify_equal("(x + 1)^2", "x^2 + 2*x + 1", 1000.0);
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_Verify_Polynomial);

static void BM_Verify_Trig(benchmark::State& state) {
    for (auto _ : state) {
        bool result = mathllm::verify_equal("sin(x)^2 + cos(x)^2", "1", 1000.0);
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_Verify_Trig);

BENCHMARK_MAIN();
