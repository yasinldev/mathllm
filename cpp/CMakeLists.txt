cmake_minimum_required(VERSION 3.20)
project(MathLLMCore VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    add_compile_options(-O3 -DNDEBUG)
endif()

option(ENABLE_OPENMP "Enable OpenMP support" OFF)
option(ENABLE_TBB "Enable Intel TBB support" OFF)
option(ENABLE_BENCHMARKS "Build benchmarks with Google Benchmark" ON)

if(ENABLE_OPENMP)
    find_package(OpenMP REQUIRED)
endif()

if(ENABLE_TBB)
    find_package(TBB REQUIRED)
endif()

include(FetchContent)

set(BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
set(BUILD_BENCHMARKS_GOOGLE OFF CACHE BOOL "" FORCE)

FetchContent_Declare(symengine
    GIT_REPOSITORY https://github.com/symengine/symengine.git
    GIT_TAG v0.11.2
)

FetchContent_Declare(eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4.0
)

set(PYBIND11_FINDPYTHON ON)

FetchContent_Declare(pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.11.1
)

FetchContent_MakeAvailable(symengine pybind11)

FetchContent_GetProperties(eigen)
if(NOT eigen_POPULATED)
    FetchContent_Populate(eigen)
endif()

add_library(Eigen3::Eigen INTERFACE IMPORTED)
target_include_directories(Eigen3::Eigen INTERFACE ${eigen_SOURCE_DIR})

add_library(mathcore SHARED
    src/symbolic.cpp
    src/verifier.cpp
    src/numeric.cpp
    src/units.cpp
    src/ode.cpp
)

target_include_directories(mathcore
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${symengine_SOURCE_DIR}
    ${symengine_BINARY_DIR}
)

target_link_libraries(mathcore
    PUBLIC
        symengine
        Eigen3::Eigen
)

if(ENABLE_OPENMP)
    target_link_libraries(mathcore PUBLIC OpenMP::OpenMP_CXX)
endif()

if(ENABLE_TBB)
    target_link_libraries(mathcore PUBLIC TBB::tbb)
endif()

target_compile_options(mathcore PRIVATE
    -Wall -Wextra -Wpedantic
    $<$<CONFIG:Release>:-march=native>
)

pybind11_add_module(mathcore_python MODULE
    bindings/pybind.cpp
)

find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)

target_link_libraries(mathcore_python PRIVATE mathcore)
target_include_directories(mathcore_python PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${symengine_SOURCE_DIR}
    ${symengine_BINARY_DIR}
    ${Python3_INCLUDE_DIRS}
)
set_target_properties(mathcore_python PROPERTIES OUTPUT_NAME "mathcore")

add_executable(mathcore_cli src/cli.cpp)
target_link_libraries(mathcore_cli PRIVATE mathcore)

enable_testing()
add_subdirectory(tests)

if(ENABLE_BENCHMARKS)
    add_subdirectory(bench)
endif()

