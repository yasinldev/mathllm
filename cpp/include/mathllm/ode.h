#pragma once

#include <string>
#include <vector>
#include <map>
#include "errors.hpp"

namespace mathllm {

struct ODEResult {
    bool success;
    std::vector<double> t_values;
    std::vector<std::vector<double>> y_values;
    int steps_taken;
    std::string message;
};

ODEResult solve_ivp(
    const std::string& expr,
    double t0,
    double t1,
    const std::vector<double>& y0,
    const std::vector<std::string>& symbols,
    double rtol = 1e-6,
    double atol = 1e-8,
    int max_steps = 1000
);

}
