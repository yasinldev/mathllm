#include "mathllm/symbolic.h"

#include <symengine/add.h>
#include <symengine/basic.h>
#include <symengine/constants.h>
#include <symengine/functions.h>
#include <symengine/mul.h>
#include <symengine/parser.h>
#include <symengine/pow.h>
#include <symengine/rational.h>
#include <symengine/sets.h>
#include <symengine/solve.h>
#include <symengine/symbol.h>
#include <symengine/symengine_exception.h>
#include <symengine/visitor.h>

#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <chrono>

namespace mathllm {

namespace {

using SymEngine::Basic;
using SymEngine::RCP;
using SymEngine::Symbol;

RCP<const Basic> parse_expression(const std::string& expr) {
	return SymEngine::parse(expr);
}

RCP<const Symbol> make_symbol(const std::string& name) {
	return SymEngine::symbol(name);
}

RCP<const Basic> integrate_basic(const RCP<const Basic>& expr, const RCP<const Symbol>& var);

RCP<const Basic> integrate_add(const SymEngine::Add& add_expr, const RCP<const Symbol>& var) {
	RCP<const Basic> result = SymEngine::integer(0);
	for (const auto& term : add_expr.get_args()) {
		result = SymEngine::add(result, integrate_basic(term, var));
	}
	return result;
}

RCP<const Basic> integrate_mul(const SymEngine::Mul& mul_expr, const RCP<const Symbol>& var) {
	RCP<const Basic> constant = SymEngine::one;
	RCP<const Basic> dependent = SymEngine::one;
	for (const auto& factor : mul_expr.get_args()) {
		if (SymEngine::has_symbol(*factor, *var)) {
			if (!SymEngine::eq(*dependent, *SymEngine::one)) {
				throw SymbolicError("Unsupported integrand");
			}
			dependent = factor;
		} else {
			constant = SymEngine::mul(constant, factor);
		}
	}
	if (SymEngine::eq(*dependent, *SymEngine::one)) {
		return SymEngine::mul(mul_expr.rcp_from_this(), var);
	}
	return SymEngine::mul(constant, integrate_basic(dependent, var));
}

RCP<const Basic> integrate_pow(const SymEngine::Pow& pow_expr, const RCP<const Symbol>& var) {
	const auto& base = pow_expr.get_base();
	const auto& exponent = pow_expr.get_exp();
	if (SymEngine::eq(*base, *SymEngine::E)) {
		if (SymEngine::eq(*exponent, *var)) {
			return SymEngine::pow(base, exponent);
		}
		throw SymbolicError("Unsupported integrand");
	}
	if (!SymEngine::eq(*base, *var)) {
		throw SymbolicError("Unsupported integrand");
	}
	if (SymEngine::is_a<SymEngine::Integer>(*exponent)) {
		const auto exp_int = SymEngine::rcp_static_cast<const SymEngine::Integer>(exponent);
		if (SymEngine::eq(*exponent, *SymEngine::minus_one)) {
			return SymEngine::log(var);
		}
		const auto exponent_plus_one = SymEngine::add(exponent, SymEngine::one);
		return SymEngine::div(SymEngine::pow(var, exponent_plus_one), exponent_plus_one);
	}
	throw SymbolicError("Unsupported integrand");
}

RCP<const Basic> integrate_basic(const RCP<const Basic>& expr, const RCP<const Symbol>& var) {
	if (!SymEngine::has_symbol(*expr, *var)) {
		return SymEngine::mul(expr, var);
	}
	if (SymEngine::is_a<Symbol>(*expr)) {
		if (SymEngine::eq(*expr, *var)) {
			const auto exponent_plus_one = SymEngine::integer(2);
			return SymEngine::div(SymEngine::pow(var, exponent_plus_one), exponent_plus_one);
		}
		return SymEngine::mul(expr, var);
	}
	if (SymEngine::is_a<SymEngine::Add>(*expr)) {
		return integrate_add(*SymEngine::rcp_static_cast<const SymEngine::Add>(expr), var);
	}
	if (SymEngine::is_a<SymEngine::Mul>(*expr)) {
		return integrate_mul(*SymEngine::rcp_static_cast<const SymEngine::Mul>(expr), var);
	}
	if (SymEngine::is_a<SymEngine::Pow>(*expr)) {
		return integrate_pow(*SymEngine::rcp_static_cast<const SymEngine::Pow>(expr), var);
	}
	if (SymEngine::is_a<SymEngine::Sin>(*expr)) {
		const auto func = SymEngine::rcp_static_cast<const SymEngine::Sin>(expr);
		const auto arg = func->get_arg();
		if (!SymEngine::eq(*arg, *var)) {
			throw SymbolicError("Unsupported integrand");
		}
		return SymEngine::mul(SymEngine::minus_one, SymEngine::cos(arg));
	}
	if (SymEngine::is_a<SymEngine::Cos>(*expr)) {
		const auto func = SymEngine::rcp_static_cast<const SymEngine::Cos>(expr);
		const auto arg = func->get_arg();
		if (!SymEngine::eq(*arg, *var)) {
			throw SymbolicError("Unsupported integrand");
		}
		return SymEngine::sin(arg);
	}
	if (SymEngine::is_a<SymEngine::FunctionSymbol>(*expr)) {
		throw SymbolicError("Unsupported integrand");
	}
	throw SymbolicError("Unsupported integrand");
}

std::string to_string(const RCP<const Basic>& expr) {
	return expr->__str__();
}

std::string solutions_to_string(const SymEngine::RCP<const SymEngine::Set>& set) {
	if (SymEngine::is_a<SymEngine::FiniteSet>(*set)) {
		const auto finite = SymEngine::rcp_static_cast<const SymEngine::FiniteSet>(set);
		const auto& elements = finite->get_container();
		if (elements.empty()) {
			return "[]";
		}
		std::vector<std::string> parts;
		parts.reserve(elements.size());
		for (const auto& element : elements) {
			parts.push_back(element->__str__());
		}
		std::ostringstream oss;
		oss << "[";
		for (std::size_t i = 0; i < parts.size(); ++i) {
			oss << parts[i];
			if (i + 1 < parts.size()) {
				oss << ", ";
			}
		}
		oss << "]";
		return oss.str();
	}
	return set->__str__();
}

}

std::string integrate(const std::string& expr, const std::string& var) {
	try {
		const auto parsed = parse_expression(expr);
		const auto symbol = make_symbol(var);
		const auto result = integrate_basic(parsed, symbol);
		return to_string(result);
	} catch (const SymEngine::SymEngineException& ex) {
		throw SymbolicError(ex.what());
	} catch (const std::exception& ex) {
		throw SymbolicError(ex.what());
	}
}

std::string diff(const std::string& expr, const std::string& var) {
	try {
		const auto parsed = parse_expression(expr);
		const auto symbol = make_symbol(var);
		const auto result = SymEngine::diff(parsed, symbol, false);
		return to_string(result);
	} catch (const SymEngine::SymEngineException& ex) {
		throw SymbolicError(ex.what());
	} catch (const std::exception& ex) {
		throw SymbolicError(ex.what());
	}
}

std::string solve_equation(const std::string& lhs, const std::string& rhs, const std::string& var) {
	try {
		const auto parsed_lhs = parse_expression(lhs);
		const auto parsed_rhs = parse_expression(rhs);
		const auto symbol = make_symbol(var);
		const auto equation = SymEngine::sub(parsed_lhs, parsed_rhs);
		const auto result_set = SymEngine::solve(equation, symbol);
		return solutions_to_string(result_set);
	} catch (const SymEngine::SymEngineException& ex) {
		throw SymbolicError(ex.what());
	} catch (const std::exception& ex) {
		throw SymbolicError(ex.what());
	}
}

bool verify_equal(const std::string& lhs, const std::string& rhs, double timeout_ms) {
	auto start = std::chrono::steady_clock::now();
	
	try {
		const auto parsed_lhs = parse_expression(lhs);
		const auto parsed_rhs = parse_expression(rhs);
		
		auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
			std::chrono::steady_clock::now() - start
		).count();
		
		if (elapsed > timeout_ms) {
			throw VerifierError("Verification timeout exceeded");
		}
		
		const auto diff = SymEngine::sub(parsed_lhs, parsed_rhs);
		const auto simplified = SymEngine::expand(diff);
		
		elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
			std::chrono::steady_clock::now() - start
		).count();
		
		if (elapsed > timeout_ms) {
			throw VerifierError("Verification timeout exceeded");
		}
		
		auto tribool_result = SymEngine::is_zero(*simplified);
		if (tribool_result == SymEngine::tribool::indeterminate) {
			return false;
		}
		return static_cast<bool>(tribool_result);
	} catch (const VerifierError&) {
		throw;
	} catch (const SymEngine::SymEngineException& ex) {
		throw VerifierError(ex.what());
	} catch (const std::exception& ex) {
		throw VerifierError(ex.what());
	}
}

}
