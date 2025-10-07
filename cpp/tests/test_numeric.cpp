#include "mathllm/numeric.h"
#include <cassert>
#include <iostream>
#include <cmath>

void test_probe_simple_identity() {
    auto result = mathllm::probe_equal(
        "x + x",
        "2*x",
        {"x"},
        10,
        42,
        0.5,
        2.0,
        1e-6
    );
    assert(result.equal && "x + x should equal 2*x");
    assert(result.trials_executed == 10);
    assert(result.failures == 0);
    std::cout << "[PASS] test_probe_simple_identity\n";
}

void test_probe_polynomial() {
    auto result = mathllm::probe_equal(
        "(x + 1)^2",
        "x^2 + 2*x + 1",
        {"x"},
        20,
        123,
        0.5,
        2.0,
        1e-6
    );
    assert(result.equal && "(x+1)^2 should equal x^2+2x+1");
    assert(result.trials_executed == 20);
    std::cout << "[PASS] test_probe_polynomial\n";
}

void test_probe_trig_identity() {
    auto result = mathllm::probe_equal(
        "sin(x)^2 + cos(x)^2",
        "1",
        {"x"},
        15,
        456,
        0.5,
        2.0,
        1e-6
    );
    assert(result.equal && "sin^2 + cos^2 should equal 1");
    std::cout << "[PASS] test_probe_trig_identity\n";
}

void test_probe_multivariable() {
    auto result = mathllm::probe_equal(
        "x*y + x*z",
        "x*(y + z)",
        {"x", "y", "z"},
        10,
        789,
        0.5,
        2.0,
        1e-6
    );
    assert(result.equal && "Distributive property should hold");
    std::cout << "[PASS] test_probe_multivariable\n";
}

void test_probe_not_equal() {
    auto result = mathllm::probe_equal(
        "x^2",
        "x + 1",
        {"x"},
        5,
        999,
        0.5,
        2.0,
        1e-6
    );
    assert(!result.equal && "x^2 should not equal x+1");
    assert(result.failures > 0);
    std::cout << "[PASS] test_probe_not_equal\n";
}

void test_probe_deterministic() {
    auto result1 = mathllm::probe_equal(
        "x^2 + 2*x",
        "x*(x + 2)",
        {"x"},
        10,
        12345,
        0.5,
        2.0,
        1e-6
    );

    auto result2 = mathllm::probe_equal(
        "x^2 + 2*x",
        "x*(x + 2)",
        {"x"},
        10,
        12345,
        0.5,
        2.0,
        1e-6
    );

    assert(result1.equal == result2.equal);
    assert(result1.failures == result2.failures);
    assert(result1.max_errors.size() == result2.max_errors.size());
    
    for (size_t i = 0; i < result1.max_errors.size(); ++i) {
        assert(std::abs(result1.max_errors[i] - result2.max_errors[i]) < 1e-12);
    }
    
    std::cout << "[PASS] test_probe_deterministic (same seed produces same results)\n";
}

void test_probe_error_handling() {
    bool caught = false;
    try {
        mathllm::probe_equal("x", "y", {}, 10, 42, 0.5, 2.0, 1e-6);
    } catch (const mathllm::NumericError& e) {
        caught = true;
        std::cout << "  Caught expected error: " << e.what() << "\n";
    }
    assert(caught && "Should throw NumericError for empty symbols");
    
    caught = false;
    try {
        mathllm::probe_equal("x", "y", {"x"}, -5, 42, 0.5, 2.0, 1e-6);
    } catch (const mathllm::NumericError&) {
        caught = true;
    }
    assert(caught && "Should throw NumericError for negative trials");
    
    caught = false;
    try {
        mathllm::probe_equal("x", "y", {"x"}, 10, 42, 2.0, 0.5, 1e-6);
    } catch (const mathllm::NumericError&) {
        caught = true;
    }
    assert(caught && "Should throw NumericError for invalid domain");
    
    std::cout << "[PASS] test_probe_error_handling\n";
}

void test_probe_max_errors_tracking() {
    auto result = mathllm::probe_equal(
        "x",
        "x + 0.0001",
        {"x"},
        5,
        555,
        1.0,
        2.0,
        1e-3
    );
    
    assert(!result.equal && "Should detect small differences");
    assert(result.max_errors.size() == 5);
    
    for (double err : result.max_errors) {
        assert(err < 1.0 && "Errors should be reasonable for x vs x+0.0001");
    }
    
    std::cout << "[PASS] test_probe_max_errors_tracking\n";
}

int main() {
    std::cout << "=== Phase C: Numeric Probe Tests ===\n";
    
    test_probe_simple_identity();
    test_probe_polynomial();
    test_probe_trig_identity();
    test_probe_multivariable();
    test_probe_not_equal();
    test_probe_deterministic();
    test_probe_error_handling();
    test_probe_max_errors_tracking();
    
    std::cout << "\n[SUCCESS] All numeric probe tests passed\n";
    return 0;
}
