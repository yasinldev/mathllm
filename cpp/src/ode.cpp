#include "mathllm/ode.h"
#include <symengine/parser.h>
#include <symengine/eval_double.h>
#include <symengine/symbol.h>
#include <symengine/real_double.h>
#include <cmath>
#include <limits>

namespace mathllm {

using namespace SymEngine;

class ODEEvaluator {
public:
    ODEEvaluator(const RCP<const Basic>& expr, const std::vector<std::string>& symbols)
        : expr_(expr), symbols_(symbols) {
        for (const auto& sym : symbols) {
            symbol_map_[sym] = symbol(sym);
        }
    }
    
    std::vector<double> evaluate(double t, const std::vector<double>& y) {
        if (y.size() != symbols_.size() - 1) {
            throw ODEError("Mismatch between y values and symbols");
        }
        
        map_basic_basic subs;
        subs[symbol_map_[symbols_[0]]] = real_double(t);
        
        for (size_t i = 0; i < y.size(); ++i) {
            subs[symbol_map_[symbols_[i + 1]]] = real_double(y[i]);
        }
        
        auto result = expr_->subs(subs);
        
        double val = eval_double(*result);
        
        if (std::isnan(val) || std::isinf(val)) {
            throw ODEError("Invalid function evaluation: NaN or Inf");
        }
        
        return {val};
    }
    
private:
    RCP<const Basic> expr_;
    std::vector<std::string> symbols_;
    std::map<std::string, RCP<const Symbol>> symbol_map_;
};

ODEResult solve_ivp(
    const std::string& expr,
    double t0,
    double t1,
    const std::vector<double>& y0,
    const std::vector<std::string>& symbols,
    double rtol,
    double atol,
    int max_steps
) {
    ODEResult result;
    result.success = false;
    result.steps_taken = 0;
    
    if (t1 <= t0) {
        result.message = "t1 must be greater than t0";
        return result;
    }
    
    if (y0.empty()) {
        result.message = "Initial conditions y0 cannot be empty";
        return result;
    }
    
    if (symbols.empty()) {
        result.message = "Symbols list cannot be empty";
        return result;
    }
    
    if (max_steps <= 0) {
        result.message = "max_steps must be positive";
        return result;
    }
    
    try {
        auto parsed = parse(expr);
        ODEEvaluator evaluator(parsed, symbols);
        
        double h = (t1 - t0) / max_steps;
        double t = t0;
        std::vector<double> y = y0;
        
        result.t_values.push_back(t);
        result.y_values.push_back(y);
        
        const double explosion_threshold = 1e10;
        
        for (int step = 0; step < max_steps; ++step) {
            try {
                auto k1 = evaluator.evaluate(t, y);
                
                std::vector<double> y_temp(y.size());
                for (size_t i = 0; i < y.size(); ++i) {
                    y_temp[i] = y[i] + 0.5 * h * k1[0];
                }
                auto k2 = evaluator.evaluate(t + 0.5 * h, y_temp);
                
                for (size_t i = 0; i < y.size(); ++i) {
                    y_temp[i] = y[i] + 0.5 * h * k2[0];
                }
                auto k3 = evaluator.evaluate(t + 0.5 * h, y_temp);
                
                for (size_t i = 0; i < y.size(); ++i) {
                    y_temp[i] = y[i] + h * k3[0];
                }
                auto k4 = evaluator.evaluate(t + h, y_temp);
                
                for (size_t i = 0; i < y.size(); ++i) {
                    y[i] = y[i] + (h / 6.0) * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]);
                }
                
                t += h;
                result.steps_taken++;
                
                for (double val : y) {
                    if (std::abs(val) > explosion_threshold) {
                        result.message = "Solution exploded (exceeded threshold)";
                        result.success = false;
                        return result;
                    }
                }
                
                result.t_values.push_back(t);
                result.y_values.push_back(y);
                
            } catch (const ODEError& e) {
                result.message = std::string("ODE evaluation failed: ") + e.what();
                return result;
            }
            
            if (t >= t1 - 1e-10) {
                break;
            }
        }
        
        result.success = true;
        result.message = "Integration completed successfully";
        
    } catch (const SymEngine::ParseError& e) {
        throw ParseError(std::string("Failed to parse ODE expression: ") + e.what());
    } catch (const std::exception& e) {
        throw ODEError(std::string("ODE integration failed: ") + e.what());
    }
    
    return result;
}

}
