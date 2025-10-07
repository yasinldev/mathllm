#include "mathllm/units.h"
#include <cassert>
#include <iostream>

void test_dimension_equality() {
    mathllm::Dimension d1(1, 0, -1);
    mathllm::Dimension d2(1, 0, -1);
    assert(d1 == d2);
    std::cout << "[PASS] test_dimension_equality\n";
}

void test_dimension_addition() {
    mathllm::Dimension velocity(1, 0, -1);
    mathllm::Dimension acceleration(1, 0, -2);
    
    auto result = velocity + acceleration;
    assert(result.length == 2);
    assert(result.time == -3);
    std::cout << "[PASS] test_dimension_addition\n";
}

void test_dimension_multiplication() {
    mathllm::Dimension length(1, 0, 0);
    auto area = length * 2;
    assert(area.length == 2);
    std::cout << "[PASS] test_dimension_multiplication\n";
}

void test_valid_addition() {
    std::map<std::string, mathllm::Dimension> dims;
    dims["x"] = mathllm::Dimension(1, 0, 0);
    dims["y"] = mathllm::Dimension(1, 0, 0);
    
    auto result = mathllm::unit_check("x + y", dims);
    assert(result.ok);
    assert(result.errors.empty());
    std::cout << "[PASS] test_valid_addition (same dimensions)\n";
}

void test_invalid_addition() {
    std::map<std::string, mathllm::Dimension> dims;
    dims["distance"] = mathllm::Dimension(1, 0, 0);
    dims["time"] = mathllm::Dimension(0, 0, 1);
    
    auto result = mathllm::unit_check("distance + time", dims);
    assert(!result.ok);
    assert(!result.errors.empty());
    std::cout << "[PASS] test_invalid_addition (mismatched dimensions)\n";
}

void test_multiplication() {
    std::map<std::string, mathllm::Dimension> dims;
    dims["length"] = mathllm::Dimension(1, 0, 0);
    dims["width"] = mathllm::Dimension(1, 0, 0);
    
    auto result = mathllm::unit_check("length * width", dims);
    assert(result.ok);
    assert(result.inferred_dimensions["result"].length == 2);
    std::cout << "[PASS] test_multiplication (L * L = L^2)\n";
}

void test_division() {
    std::map<std::string, mathllm::Dimension> dims;
    dims["distance"] = mathllm::Dimension(1, 0, 0);
    dims["time"] = mathllm::Dimension(0, 0, 1);
    
    auto result = mathllm::unit_check("distance / time", dims);
    assert(result.ok);
    assert(result.inferred_dimensions["result"].length == 1);
    assert(result.inferred_dimensions["result"].time == -1);
    std::cout << "[PASS] test_division (velocity = L/T)\n";
}

void test_power_integer() {
    std::map<std::string, mathllm::Dimension> dims;
    dims["r"] = mathllm::Dimension(1, 0, 0);
    
    auto result = mathllm::unit_check("r^2", dims);
    assert(result.ok);
    assert(result.inferred_dimensions["result"].length == 2);
    std::cout << "[PASS] test_power_integer (r^2 = L^2)\n";
}

void test_trig_dimensionless() {
    std::map<std::string, mathllm::Dimension> dims;
    dims["angle"] = mathllm::Dimension();
    
    auto result = mathllm::unit_check("sin(angle)", dims);
    assert(result.ok);
    assert(result.inferred_dimensions["result"].is_dimensionless());
    std::cout << "[PASS] test_trig_dimensionless (sin requires dimensionless)\n";
}

void test_trig_dimensional_error() {
    std::map<std::string, mathllm::Dimension> dims;
    dims["distance"] = mathllm::Dimension(1, 0, 0);
    
    auto result = mathllm::unit_check("sin(distance)", dims);
    assert(!result.ok);
    assert(!result.errors.empty());
    std::cout << "[PASS] test_trig_dimensional_error (sin of dimensional arg)\n";
}

void test_log_dimensionless() {
    std::map<std::string, mathllm::Dimension> dims;
    dims["ratio"] = mathllm::Dimension();
    
    auto result = mathllm::unit_check("log(ratio)", dims);
    assert(result.ok);
    std::cout << "[PASS] test_log_dimensionless\n";
}

void test_complex_expression() {
    std::map<std::string, mathllm::Dimension> dims;
    dims["m"] = mathllm::Dimension(0, 1, 0);
    dims["v"] = mathllm::Dimension(1, 0, -1);
    
    auto result = mathllm::unit_check("(1/2) * m * v^2", dims);
    assert(result.ok);
    assert(result.inferred_dimensions["result"].mass == 1);
    assert(result.inferred_dimensions["result"].length == 2);
    assert(result.inferred_dimensions["result"].time == -2);
    std::cout << "[PASS] test_complex_expression (kinetic energy = M L^2 T^-2)\n";
}

void test_unknown_symbol_warning() {
    std::map<std::string, mathllm::Dimension> dims;
    dims["x"] = mathllm::Dimension(1, 0, 0);
    
    auto result = mathllm::unit_check("x + y", dims);
    assert(!result.warnings.empty());
    std::cout << "[PASS] test_unknown_symbol_warning\n";
}

void test_dimensionless_constants() {
    std::map<std::string, mathllm::Dimension> dims;
    dims["L"] = mathllm::Dimension(1, 0, 0);
    
    auto result = mathllm::unit_check("2 * L + 3 * L", dims);
    assert(result.ok);
    assert(result.inferred_dimensions["result"].length == 1);
    std::cout << "[PASS] test_dimensionless_constants\n";
}

int main() {
    std::cout << "=== Phase D: Units/Dimensions Tests ===\n";
    
    test_dimension_equality();
    test_dimension_addition();
    test_dimension_multiplication();
    test_valid_addition();
    test_invalid_addition();
    test_multiplication();
    test_division();
    test_power_integer();
    test_trig_dimensionless();
    test_trig_dimensional_error();
    test_log_dimensionless();
    test_complex_expression();
    test_unknown_symbol_warning();
    test_dimensionless_constants();
    
    std::cout << "\n[SUCCESS] All units/dimensions tests passed\n";
    return 0;
}
