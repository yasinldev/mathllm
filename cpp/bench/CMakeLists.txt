include(FetchContent)

FetchContent_Declare(
    googlebenchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG v1.8.3
)

set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(googlebenchmark)

add_executable(bench_symbolic bench_symbolic.cpp)
target_link_libraries(bench_symbolic PRIVATE mathcore benchmark::benchmark)

add_custom_target(run_benchmarks
    COMMAND bench_symbolic --benchmark_out=benchmark_results.json --benchmark_out_format=json
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS bench_symbolic
    COMMENT "Running symbolic benchmarks"
)
