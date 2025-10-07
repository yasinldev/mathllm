#include "mathllm/ode.h"
#include <cmath>
#include <iostream>
#include <iomanip>

using namespace mathllm;

void test_exponential_growth() {
    std::cout << "Test: Exponential growth (y' = y)" << std::endl;
    
    auto result = solve_ivp("y", 0.0, 1.0, {1.0}, {"t", "y"}, 1e-6, 1e-8, 100);
    
    if (!result.success) {
        std::cerr << "  FAILED: " << result.message << std::endl;
        return;
    }
    
    double t_final = result.t_values.back();
    double y_final = result.y_values.back()[0];
    double y_expected = std::exp(t_final);
    double error = std::abs(y_final - y_expected);
    
    std::cout << "  t_final: " << t_final << std::endl;
    std::cout << "  y_final: " << y_final << std::endl;
    std::cout << "  y_expected: " << y_expected << std::endl;
    std::cout << "  error: " << error << std::endl;
    
    if (error < 0.01) {
        std::cout << "  PASSED" << std::endl;
    } else {
        std::cerr << "  FAILED: Error too large" << std::endl;
    }
}

void test_exponential_decay() {
    std::cout << "\nTest: Exponential decay (y' = -y)" << std::endl;
    
    auto result = solve_ivp("-y", 0.0, 1.0, {1.0}, {"t", "y"}, 1e-6, 1e-8, 100);
    
    if (!result.success) {
        std::cerr << "  FAILED: " << result.message << std::endl;
        return;
    }
    
    double t_final = result.t_values.back();
    double y_final = result.y_values.back()[0];
    double y_expected = std::exp(-t_final);
    double error = std::abs(y_final - y_expected);
    
    std::cout << "  t_final: " << t_final << std::endl;
    std::cout << "  y_final: " << y_final << std::endl;
    std::cout << "  y_expected: " << y_expected << std::endl;
    std::cout << "  error: " << error << std::endl;
    
    if (error < 0.01) {
        std::cout << "  PASSED" << std::endl;
    } else {
        std::cerr << "  FAILED: Error too large" << std::endl;
    }
}

void test_linear_growth() {
    std::cout << "\nTest: Linear growth (y' = 1)" << std::endl;
    
    auto result = solve_ivp("1", 0.0, 2.0, {0.0}, {"t", "y"}, 1e-6, 1e-8, 100);
    
    if (!result.success) {
        std::cerr << "  FAILED: " << result.message << std::endl;
        return;
    }
    
    double t_final = result.t_values.back();
    double y_final = result.y_values.back()[0];
    double y_expected = t_final;
    double error = std::abs(y_final - y_expected);
    
    std::cout << "  t_final: " << t_final << std::endl;
    std::cout << "  y_final: " << y_final << std::endl;
    std::cout << "  y_expected: " << y_expected << std::endl;
    std::cout << "  error: " << error << std::endl;
    
    if (error < 0.01) {
        std::cout << "  PASSED" << std::endl;
    } else {
        std::cerr << "  FAILED: Error too large" << std::endl;
    }
}

void test_scaled_decay() {
    std::cout << "\nTest: Scaled decay (y' = -2*y)" << std::endl;
    
    auto result = solve_ivp("-2*y", 0.0, 1.0, {1.0}, {"t", "y"}, 1e-6, 1e-8, 100);
    
    if (!result.success) {
        std::cerr << "  FAILED: " << result.message << std::endl;
        return;
    }
    
    double t_final = result.t_values.back();
    double y_final = result.y_values.back()[0];
    double y_expected = std::exp(-2.0 * t_final);
    double error = std::abs(y_final - y_expected);
    
    std::cout << "  t_final: " << t_final << std::endl;
    std::cout << "  y_final: " << y_final << std::endl;
    std::cout << "  y_expected: " << y_expected << std::endl;
    std::cout << "  error: " << error << std::endl;
    
    if (error < 0.01) {
        std::cout << "  PASSED" << std::endl;
    } else {
        std::cerr << "  FAILED: Error too large" << std::endl;
    }
}

void test_quadratic() {
    std::cout << "\nTest: Quadratic (y' = 2*t)" << std::endl;
    
    auto result = solve_ivp("2*t", 0.0, 1.0, {0.0}, {"t", "y"}, 1e-6, 1e-8, 100);
    
    if (!result.success) {
        std::cerr << "  FAILED: " << result.message << std::endl;
        return;
    }
    
    double t_final = result.t_values.back();
    double y_final = result.y_values.back()[0];
    double y_expected = t_final * t_final;
    double error = std::abs(y_final - y_expected);
    
    std::cout << "  t_final: " << t_final << std::endl;
    std::cout << "  y_final: " << y_final << std::endl;
    std::cout << "  y_expected: " << y_expected << std::endl;
    std::cout << "  error: " << error << std::endl;
    
    if (error < 0.01) {
        std::cout << "  PASSED" << std::endl;
    } else {
        std::cerr << "  FAILED: Error too large" << std::endl;
    }
}

void test_invalid_interval() {
    std::cout << "\nTest: Invalid interval (t1 <= t0)" << std::endl;
    
    auto result = solve_ivp("y", 1.0, 0.0, {1.0}, {"t", "y"});
    
    if (!result.success && result.message.find("t1 must be greater than t0") != std::string::npos) {
        std::cout << "  PASSED: Correctly rejected invalid interval" << std::endl;
    } else {
        std::cerr << "  FAILED: Should have rejected invalid interval" << std::endl;
    }
}

void test_empty_initial_conditions() {
    std::cout << "\nTest: Empty initial conditions" << std::endl;
    
    auto result = solve_ivp("y", 0.0, 1.0, {}, {"t", "y"});
    
    if (!result.success && result.message.find("Initial conditions") != std::string::npos) {
        std::cout << "  PASSED: Correctly rejected empty y0" << std::endl;
    } else {
        std::cerr << "  FAILED: Should have rejected empty y0" << std::endl;
    }
}

void test_invalid_expression() {
    std::cout << "\nTest: Invalid expression" << std::endl;
    
    try {
        auto result = solve_ivp("invalid@#$", 0.0, 1.0, {1.0}, {"t", "y"});
        std::cerr << "  FAILED: Should have thrown ParseError" << std::endl;
    } catch (const ParseError& e) {
        std::cout << "  PASSED: Correctly threw ParseError" << std::endl;
    } catch (...) {
        std::cerr << "  FAILED: Threw wrong exception type" << std::endl;
    }
}

void test_explosion_detection() {
    std::cout << "\nTest: Explosion detection (y' = 10*y)" << std::endl;
    
    auto result = solve_ivp("10*y", 0.0, 5.0, {1.0}, {"t", "y"}, 1e-6, 1e-8, 1000);
    
    if (!result.success && result.message.find("exploded") != std::string::npos) {
        std::cout << "  PASSED: Correctly detected explosion" << std::endl;
    } else {
        std::cerr << "  FAILED: Should have detected explosion" << std::endl;
    }
}

void test_steps_count() {
    std::cout << "\nTest: Step count tracking" << std::endl;
    
    auto result = solve_ivp("y", 0.0, 1.0, {1.0}, {"t", "y"}, 1e-6, 1e-8, 50);
    
    if (result.success && result.steps_taken > 0 && result.steps_taken <= 50) {
        std::cout << "  steps_taken: " << result.steps_taken << std::endl;
        std::cout << "  PASSED" << std::endl;
    } else {
        std::cerr << "  FAILED: Invalid step count" << std::endl;
    }
}

int main() {
    std::cout << std::fixed << std::setprecision(6);
    
    test_exponential_growth();
    test_exponential_decay();
    test_linear_growth();
    test_scaled_decay();
    test_quadratic();
    test_invalid_interval();
    test_empty_initial_conditions();
    test_invalid_expression();
    test_explosion_detection();
    test_steps_count();
    
    return 0;
}
