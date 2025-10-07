#pragma once

#include <string>
#include <vector>
#include <map>
#include "errors.hpp"

namespace mathllm {

struct ProbeResult {
    bool equal;
    int trials_executed;
    int failures;
    std::vector<double> max_errors;
};

ProbeResult probe_equal(
    const std::string& lhs_str,
    const std::string& rhs_str,
    const std::vector<std::string>& symbols,
    int trials = 10,
    unsigned int seed = 42,
    double domain_min = 0.5,
    double domain_max = 2.0,
    double threshold = 1e-6
);

}
