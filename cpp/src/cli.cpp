#include "mathllm/symbolic.h"
#include "mathllm/verifier.h"

#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void print_usage() {
    std::cout << "Usage:\n";
    std::cout << "  mathcore_cli integrate <expr> <var>\n";
    std::cout << "  mathcore_cli diff <expr> <var>\n";
    std::cout << "  mathcore_cli solve_equation <lhs> <rhs> <var>\n";
    std::cout << "  mathcore_cli verify_equal <lhs> <rhs>\n";
}

}

int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage();
        return 1;
    }
    const std::string command = argv[1];
    try {
        if (command == "integrate" || command == "diff") {
            if (argc < 4) {
                print_usage();
                return 1;
            }
            const std::string expr = argv[2];
            const std::string var = argv[3];
            if (command == "integrate") {
                std::cout << mathllm::integrate(expr, var) << std::endl;
            } else {
                std::cout << mathllm::diff(expr, var) << std::endl;
            }
            return 0;
        }
        if (command == "solve_equation") {
            if (argc < 5) {
                print_usage();
                return 1;
            }
            const std::string lhs = argv[2];
            const std::string rhs = argv[3];
            const std::string var = argv[4];
            std::cout << mathllm::solve_equation(lhs, rhs, var) << std::endl;
            return 0;
        }
        if (command == "verify_equal") {
            if (argc < 4) {
                print_usage();
                return 1;
            }
            const std::string lhs = argv[2];
            const std::string rhs = argv[3];
            std::cout << (mathllm::verify_equal(lhs, rhs, 1000.0) ? "true" : "false") << std::endl;
            return 0;
        }
        print_usage();
        return 1;
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << std::endl;
        return 1;
    }
}
