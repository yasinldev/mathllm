#include "mathllm/numeric.h"

#include <symengine/basic.h>
#include <symengine/parser.h>
#include <symengine/symbol.h>
#include <symengine/real_double.h>
#include <symengine/visitor.h>
#include <symengine/add.h>
#include <symengine/mul.h>
#include <symengine/pow.h>
#include <symengine/functions.h>
#include <symengine/integer.h>
#include <symengine/rational.h>

#include <Eigen/Dense>
#include <random>
#include <cmath>
#include <map>
#include <vector>
#include <string>
#include <limits>

namespace mathllm {

namespace {

using SymEngine::RCP;
using SymEngine::Basic;
using SymEngine::Symbol;

class NumericEvaluator : public SymEngine::BaseVisitor<NumericEvaluator> {
public:
    double result;
    const std::map<std::string, double>& values;

    NumericEvaluator(const std::map<std::string, double>& vals) 
        : result(0.0), values(vals) {}

    void bvisit(const SymEngine::Symbol& x) {
        auto it = values.find(x.get_name());
        if (it != values.end()) {
            result = it->second;
        } else {
            throw NumericError("Undefined symbol: " + x.get_name());
        }
    }

    void bvisit(const SymEngine::Integer& x) {
        result = SymEngine::mp_get_d(x.as_integer_class());
    }

    void bvisit(const SymEngine::RealDouble& x) {
        result = x.as_double();
    }

    void bvisit(const SymEngine::Rational& x) {
        result = SymEngine::mp_get_d(x.as_rational_class());
    }

    void bvisit(const SymEngine::Add& x) {
        result = 0.0;
        for (const auto& arg : x.get_args()) {
            double prev = result;
            arg->accept(*this);
            result = prev + result;
        }
    }

    void bvisit(const SymEngine::Mul& x) {
        result = 1.0;
        for (const auto& arg : x.get_args()) {
            double prev = result;
            arg->accept(*this);
            result = prev * result;
        }
    }

    void bvisit(const SymEngine::Pow& x) {
        x.get_base()->accept(*this);
        double base = result;
        x.get_exp()->accept(*this);
        double exp = result;
        result = std::pow(base, exp);
    }

    void bvisit(const SymEngine::Sin& x) {
        x.get_arg()->accept(*this);
        result = std::sin(result);
    }

    void bvisit(const SymEngine::Cos& x) {
        x.get_arg()->accept(*this);
        result = std::cos(result);
    }

    void bvisit(const SymEngine::Tan& x) {
        x.get_arg()->accept(*this);
        result = std::tan(result);
    }

    void bvisit(const SymEngine::Log& x) {
        x.get_arg()->accept(*this);
        result = std::log(result);
    }

    void bvisit(const Basic& x) {
        throw NumericError("Unsupported expression type for numeric evaluation");
    }
};

double evaluate_at_point(
    const RCP<const Basic>& expr,
    const std::map<std::string, double>& point
) {
    NumericEvaluator evaluator(point);
    expr->accept(evaluator);
    return evaluator.result;
}

}

ProbeResult probe_equal(
    const std::string& lhs_str,
    const std::string& rhs_str,
    const std::vector<std::string>& symbols,
    int trials,
    unsigned int seed,
    double domain_min,
    double domain_max,
    double threshold
) {
    if (symbols.empty()) {
        throw NumericError("No symbols provided for numeric probe");
    }

    if (trials <= 0) {
        throw NumericError("Number of trials must be positive");
    }

    if (domain_min >= domain_max) {
        throw NumericError("Invalid domain: min must be less than max");
    }

    RCP<const Basic> lhs;
    RCP<const Basic> rhs;

    try {
        lhs = SymEngine::parse(lhs_str);
        rhs = SymEngine::parse(rhs_str);
    } catch (const std::exception& e) {
        throw NumericError(std::string("Parse error: ") + e.what());
    }

    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dist(domain_min, domain_max);

    int failures = 0;
    std::vector<double> max_errors;
    max_errors.reserve(trials);

    for (int trial = 0; trial < trials; ++trial) {
        std::map<std::string, double> point;
        
        for (const auto& sym : symbols) {
            double value = dist(rng);
            
            if (std::abs(value) < 1e-10) {
                value = domain_min + 0.1;
            }
            
            point[sym] = value;
        }

        double lhs_val, rhs_val;
        try {
            lhs_val = evaluate_at_point(lhs, point);
            rhs_val = evaluate_at_point(rhs, point);
        } catch (const NumericError&) {
            ++failures;
            max_errors.push_back(std::numeric_limits<double>::infinity());
            continue;
        }

        if (!std::isfinite(lhs_val) || !std::isfinite(rhs_val)) {
            ++failures;
            max_errors.push_back(std::numeric_limits<double>::infinity());
            continue;
        }

        double abs_error = std::abs(lhs_val - rhs_val);
        double rel_error = abs_error / (std::abs(rhs_val) + 1e-10);
        double error = std::max(abs_error, rel_error);
        
        max_errors.push_back(error);

        if (error > threshold) {
            ++failures;
        }
    }

    bool equal = (failures == 0);

    return ProbeResult{
        equal,
        trials,
        failures,
        max_errors
    };
}

}
