#include "mathllm/verifier.h"

#include <symengine/add.h>
#include <symengine/constants.h>
#include <symengine/parser.h>
#include <symengine/simplify.h>
#include <symengine/symbol.h>
#include <symengine/symengine_exception.h>

#include <stdexcept>

namespace mathllm {

bool verify_equal(const std::string& lhs, const std::string& rhs) {
    try {
        auto lhs_expr = SymEngine::parse(lhs);
        auto rhs_expr = SymEngine::parse(rhs);
        auto diff_expr = SymEngine::simplify(SymEngine::sub(lhs_expr, rhs_expr));
        return SymEngine::eq(*diff_expr, *SymEngine::integer(0));
    } catch (const SymEngine::SymEngineException& ex) {
        throw std::runtime_error(ex.what());
    } catch (const std::exception& ex) {
        throw std::runtime_error(ex.what());
    }
}

}
