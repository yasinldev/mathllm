#pragma once

#include <stdexcept>
#include <string>

namespace mathllm {

class MathLLMError : public std::runtime_error {
public:
    explicit MathLLMError(const std::string& msg) : std::runtime_error(msg) {}
};

class ParseError : public MathLLMError {
public:
    explicit ParseError(const std::string& msg) : MathLLMError("ParseError: " + msg) {}
};

class SymbolicError : public MathLLMError {
public:
    explicit SymbolicError(const std::string& msg) : MathLLMError("SymbolicError: " + msg) {}
};

class NumericError : public MathLLMError {
public:
    explicit NumericError(const std::string& msg) : MathLLMError("NumericError: " + msg) {}
};

class VerifierError : public MathLLMError {
public:
    explicit VerifierError(const std::string& msg) : MathLLMError("VerifierError: " + msg) {}
};

class ODEError : public MathLLMError {
public:
    explicit ODEError(const std::string& msg) : MathLLMError("ODEError: " + msg) {}
};

class UnitError : public MathLLMError {
public:
    explicit UnitError(const std::string& msg) : MathLLMError("UnitError: " + msg) {}
};

}
