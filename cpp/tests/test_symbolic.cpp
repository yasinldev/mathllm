#include "mathllm/symbolic.h"
#include "mathllm/verifier.h"

#include <cassert>
#include <stdexcept>
#include <string>

int main() {
    assert(mathllm::diff("x^2", "x") == "2*x");
    assert(mathllm::diff("sin(x)", "x") == "cos(x)");
    assert(mathllm::diff("exp(x)", "x") == "exp(x)");

    assert(mathllm::integrate("2*x", "x") == "x^2");
    assert(mathllm::integrate("cos(x)", "x") == "sin(x)");
    assert(mathllm::integrate("1", "x") == "x");

    const std::string solutions = mathllm::solve_equation("x^2", "4", "x");
    assert(solutions.find("2") != std::string::npos);
    assert(solutions.find("-2") != std::string::npos);

    assert(mathllm::solve_equation("x", "5", "x") == "[5]");

    assert(mathllm::verify_equal("x^2 + 2*x + 1", "(x + 1)^2"));
    assert(!mathllm::verify_equal("x^2", "x^3"));

    bool threw = false;
    try {
        mathllm::diff("sin(", "x");
    } catch (const std::runtime_error&) {
        threw = true;
    }
    assert(threw);

    return 0;
}
