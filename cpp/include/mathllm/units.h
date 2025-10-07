#pragma once

#include <string>
#include <map>
#include <vector>
#include "errors.hpp"

namespace mathllm {

enum class BaseDimension {
    LENGTH,
    MASS,
    TIME,
    CURRENT,
    TEMPERATURE,
    AMOUNT,
    LUMINOSITY
};

struct Dimension {
    int length;
    int mass;
    int time;
    int current;
    int temperature;
    int amount;
    int luminosity;

    Dimension() 
        : length(0), mass(0), time(0), current(0), 
          temperature(0), amount(0), luminosity(0) {}

    Dimension(int L, int M, int T, int I = 0, int K = 0, int N = 0, int J = 0)
        : length(L), mass(M), time(T), current(I), 
          temperature(K), amount(N), luminosity(J) {}

    bool is_dimensionless() const {
        return length == 0 && mass == 0 && time == 0 && 
               current == 0 && temperature == 0 && 
               amount == 0 && luminosity == 0;
    }

    bool operator==(const Dimension& other) const {
        return length == other.length && 
               mass == other.mass && 
               time == other.time &&
               current == other.current && 
               temperature == other.temperature &&
               amount == other.amount && 
               luminosity == other.luminosity;
    }

    bool operator!=(const Dimension& other) const {
        return !(*this == other);
    }

    Dimension operator+(const Dimension& other) const {
        return Dimension(
            length + other.length,
            mass + other.mass,
            time + other.time,
            current + other.current,
            temperature + other.temperature,
            amount + other.amount,
            luminosity + other.luminosity
        );
    }

    Dimension operator-(const Dimension& other) const {
        return Dimension(
            length - other.length,
            mass - other.mass,
            time - other.time,
            current - other.current,
            temperature - other.temperature,
            amount - other.amount,
            luminosity - other.luminosity
        );
    }

    Dimension operator*(int scalar) const {
        return Dimension(
            length * scalar,
            mass * scalar,
            time * scalar,
            current * scalar,
            temperature * scalar,
            amount * scalar,
            luminosity * scalar
        );
    }

    std::string to_string() const;
};

struct UnitCheckResult {
    bool ok;
    std::vector<std::string> warnings;
    std::vector<std::string> errors;
    std::map<std::string, Dimension> inferred_dimensions;
};

UnitCheckResult unit_check(
    const std::string& expr,
    const std::map<std::string, Dimension>& symbol_dimensions
);

}
