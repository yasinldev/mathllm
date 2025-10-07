#pragma once

#include <string>
#include "errors.hpp"

namespace mathllm {

std::string integrate(const std::string& expr, const std::string& var);
std::string diff(const std::string& expr, const std::string& var);
std::string solve_equation(const std::string& lhs, const std::string& rhs, const std::string& var);
bool verify_equal(const std::string& lhs, const std::string& rhs, double timeout_ms = 1000.0);

}

