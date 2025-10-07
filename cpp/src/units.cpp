#include "mathllm/units.h"

#include <symengine/basic.h>
#include <symengine/parser.h>
#include <symengine/symbol.h>
#include <symengine/visitor.h>
#include <symengine/add.h>
#include <symengine/mul.h>
#include <symengine/pow.h>
#include <symengine/functions.h>
#include <symengine/integer.h>
#include <symengine/rational.h>

#include <sstream>

namespace mathllm {

std::string Dimension::to_string() const {
    if (is_dimensionless()) {
        return "dimensionless";
    }

    std::ostringstream oss;
    bool first = true;

    auto append_dim = [&](const char* name, int power) {
        if (power != 0) {
            if (!first) oss << " ";
            first = false;
            oss << name;
            if (power != 1) {
                oss << "^" << power;
            }
        }
    };

    append_dim("L", length);
    append_dim("M", mass);
    append_dim("T", time);
    append_dim("A", current);
    append_dim("K", temperature);
    append_dim("mol", amount);
    append_dim("cd", luminosity);

    return oss.str();
}

namespace {

using SymEngine::RCP;
using SymEngine::Basic;
using SymEngine::Symbol;

class DimensionChecker : public SymEngine::BaseVisitor<DimensionChecker> {
public:
    Dimension result;
    const std::map<std::string, Dimension>& symbol_dims;
    std::vector<std::string>& errors;
    std::vector<std::string>& warnings;

    DimensionChecker(
        const std::map<std::string, Dimension>& dims,
        std::vector<std::string>& errs,
        std::vector<std::string>& warns
    ) : result(), symbol_dims(dims), errors(errs), warnings(warns) {}

    void bvisit(const SymEngine::Symbol& x) {
        auto it = symbol_dims.find(x.get_name());
        if (it != symbol_dims.end()) {
            result = it->second;
        } else {
            warnings.push_back("Unknown symbol dimension: " + x.get_name());
            result = Dimension();
        }
    }

    void bvisit(const SymEngine::Integer& x) {
        result = Dimension();
    }

    void bvisit(const SymEngine::RealDouble& x) {
        result = Dimension();
    }

    void bvisit(const SymEngine::Rational& x) {
        result = Dimension();
    }

    void bvisit(const SymEngine::Add& x) {
        const auto& args = x.get_args();
        if (args.empty()) {
            result = Dimension();
            return;
        }

        args[0]->accept(*this);
        Dimension first_dim = result;

        for (size_t i = 1; i < args.size(); ++i) {
            args[i]->accept(*this);
            if (result != first_dim) {
                errors.push_back("Addition/subtraction requires matching dimensions");
                result = Dimension();
                return;
            }
        }

        result = first_dim;
    }

    void bvisit(const SymEngine::Mul& x) {
        result = Dimension();
        
        for (const auto& arg : x.get_args()) {
            arg->accept(*this);
            result = result + this->result;
        }
    }

    void bvisit(const SymEngine::Pow& x) {
        x.get_base()->accept(*this);
        Dimension base_dim = result;

        x.get_exp()->accept(*this);
        Dimension exp_dim = result;

        if (!exp_dim.is_dimensionless()) {
            errors.push_back("Exponent must be dimensionless");
            result = Dimension();
            return;
        }

        if (SymEngine::is_a<SymEngine::Integer>(*x.get_exp())) {
            auto exp_int = SymEngine::rcp_static_cast<const SymEngine::Integer>(x.get_exp());
            int exp_val = SymEngine::mp_get_si(exp_int->as_integer_class());
            result = base_dim * exp_val;
        } else if (SymEngine::is_a<SymEngine::Rational>(*x.get_exp())) {
            if (!base_dim.is_dimensionless()) {
                warnings.push_back("Fractional power of dimensional quantity");
            }
            result = Dimension();
        } else {
            if (!base_dim.is_dimensionless()) {
                errors.push_back("Non-integer power requires dimensionless base");
            }
            result = Dimension();
        }
    }

    void bvisit(const SymEngine::Sin& x) {
        x.get_arg()->accept(*this);
        if (!result.is_dimensionless()) {
            errors.push_back("sin() argument must be dimensionless");
        }
        result = Dimension();
    }

    void bvisit(const SymEngine::Cos& x) {
        x.get_arg()->accept(*this);
        if (!result.is_dimensionless()) {
            errors.push_back("cos() argument must be dimensionless");
        }
        result = Dimension();
    }

    void bvisit(const SymEngine::Tan& x) {
        x.get_arg()->accept(*this);
        if (!result.is_dimensionless()) {
            errors.push_back("tan() argument must be dimensionless");
        }
        result = Dimension();
    }

    void bvisit(const SymEngine::Log& x) {
        x.get_arg()->accept(*this);
        if (!result.is_dimensionless()) {
            errors.push_back("log() argument must be dimensionless");
        }
        result = Dimension();
    }

    void bvisit(const Basic& x) {
        warnings.push_back("Unknown expression type for dimension analysis");
        result = Dimension();
    }
};

}

UnitCheckResult unit_check(
    const std::string& expr,
    const std::map<std::string, Dimension>& symbol_dimensions
) {
    UnitCheckResult result;
    result.ok = true;

    RCP<const Basic> parsed;
    try {
        parsed = SymEngine::parse(expr);
    } catch (const std::exception& e) {
        result.ok = false;
        result.errors.push_back(std::string("Parse error: ") + e.what());
        return result;
    }

    DimensionChecker checker(symbol_dimensions, result.errors, result.warnings);
    parsed->accept(checker);

    if (!result.errors.empty()) {
        result.ok = false;
    }

    result.inferred_dimensions["result"] = checker.result;

    return result;
}

}
