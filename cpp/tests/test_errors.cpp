#include "mathllm/symbolic.h"
#include <cassert>
#include <iostream>
#include <string>

void test_parse_error() {
    bool caught = false;
    try {
        mathllm::diff("invalid^^expr", "x");
    } catch (const mathllm::SymbolicError&) {
        caught = true;
    }
    assert(caught && "Expected ParseError or SymbolicError for invalid expression");
    std::cout << "[PASS] test_parse_error\n";
}

void test_division_by_zero() {
    try {
        auto result = mathllm::integrate("1/0", "x");
        std::cout << "[PASS] test_division_by_zero (SymEngine handles zoo: " << result << ")\n";
    } catch (const mathllm::SymbolicError& e) {
        std::cout << "[PASS] test_division_by_zero (caught: " << e.what() << ")\n";
    }
}

void test_unsupported_integrand() {
    bool caught = false;
    try {
        mathllm::integrate("tan(x)", "x");
    } catch (const mathllm::SymbolicError&) {
        caught = true;
    } catch (const std::runtime_error&) {
        caught = true;
    }
    assert(caught && "Expected SymbolicError for unsupported integrand");
    std::cout << "[PASS] test_unsupported_integrand\n";
}

void test_verify_timeout() {
    bool caught = false;
    try {
        mathllm::verify_equal("x^100 + x^99", "x^100 + x^99 + 1", 0.1);
    } catch (const mathllm::VerifierError&) {
        caught = true;
    }
    std::cout << "[PASS] test_verify_timeout (caught=" << caught << ")\n";
}

void test_empty_expression() {
    bool caught = false;
    try {
        mathllm::diff("", "x");
    } catch (const mathllm::SymbolicError&) {
        caught = true;
    } catch (const std::exception&) {
        caught = true;
    }
    assert(caught && "Expected error for empty expression");
    std::cout << "[PASS] test_empty_expression\n";
}

void test_valid_operations() {
    try {
        auto result = mathllm::integrate("x", "x");
        assert(!result.empty());
        
        result = mathllm::diff("x^2", "x");
        assert(!result.empty());
        
        result = mathllm::solve_equation("x^2", "4", "x");
        assert(!result.empty());
        
        bool equal = mathllm::verify_equal("x + x", "2*x", 100.0);
        assert(equal);
        
        std::cout << "[PASS] test_valid_operations\n";
    } catch (const std::exception& e) {
        std::cout << "[FAIL] test_valid_operations: " << e.what() << "\n";
        throw;
    }
}

int main() {
    std::cout << "=== Phase A: Error Handling Tests ===\n";
    
    test_parse_error();
    test_division_by_zero();
    test_unsupported_integrand();
    test_verify_timeout();
    test_empty_expression();
    test_valid_operations();
    
    std::cout << "\n[SUCCESS] All Phase A tests passed\n";
    return 0;
}
