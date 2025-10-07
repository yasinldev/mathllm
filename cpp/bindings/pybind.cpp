#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "mathllm/symbolic.h"
#include "mathllm/verifier.h"
#include "mathllm/numeric.h"
#include "mathllm/units.h"
#include "mathllm/ode.h"

namespace py = pybind11;

PYBIND11_MODULE(mathcore, m) {
    m.doc() = "MathLLM core symbolic bindings";
    
    py::register_exception<mathllm::ParseError>(m, "ParseError");
    py::register_exception<mathllm::SymbolicError>(m, "SymbolicError");
    py::register_exception<mathllm::VerifierError>(m, "VerifierError");
    py::register_exception<mathllm::NumericError>(m, "NumericError");
    py::register_exception<mathllm::UnitError>(m, "UnitError");
    py::register_exception<mathllm::ODEError>(m, "ODEError");
    
    m.def("integrate", &mathllm::integrate, 
          py::arg("expr"), py::arg("var"));
    m.def("diff", &mathllm::diff,
          py::arg("expr"), py::arg("var"));
    m.def("solve_equation", &mathllm::solve_equation,
          py::arg("lhs"), py::arg("rhs"), py::arg("var"));
    m.def("verify_equal", 
          py::overload_cast<const std::string&, const std::string&, double>(&mathllm::verify_equal),
          py::arg("lhs"), py::arg("rhs"), py::arg("timeout_ms") = 1000.0);
    
    py::class_<mathllm::ProbeResult>(m, "ProbeResult")
        .def_readonly("equal", &mathllm::ProbeResult::equal)
        .def_readonly("trials_executed", &mathllm::ProbeResult::trials_executed)
        .def_readonly("failures", &mathllm::ProbeResult::failures)
        .def_readonly("max_errors", &mathllm::ProbeResult::max_errors);
    
    m.def("probe_equal", &mathllm::probe_equal,
          py::arg("lhs"), py::arg("rhs"), py::arg("symbols"),
          py::arg("trials") = 10,
          py::arg("seed") = 42,
          py::arg("domain_min") = 0.5,
          py::arg("domain_max") = 2.0,
          py::arg("threshold") = 1e-6);
    
    py::class_<mathllm::Dimension>(m, "Dimension")
        .def(py::init<>())
        .def(py::init<int, int, int, int, int, int, int>(),
             py::arg("length") = 0, py::arg("mass") = 0, py::arg("time") = 0,
             py::arg("current") = 0, py::arg("temperature") = 0,
             py::arg("amount") = 0, py::arg("luminosity") = 0)
        .def_readwrite("length", &mathllm::Dimension::length)
        .def_readwrite("mass", &mathllm::Dimension::mass)
        .def_readwrite("time", &mathllm::Dimension::time)
        .def_readwrite("current", &mathllm::Dimension::current)
        .def_readwrite("temperature", &mathllm::Dimension::temperature)
        .def_readwrite("amount", &mathllm::Dimension::amount)
        .def_readwrite("luminosity", &mathllm::Dimension::luminosity)
        .def("is_dimensionless", &mathllm::Dimension::is_dimensionless)
        .def("__eq__", &mathllm::Dimension::operator==)
        .def("__str__", &mathllm::Dimension::to_string);
    
    py::class_<mathllm::UnitCheckResult>(m, "UnitCheckResult")
        .def_readonly("ok", &mathllm::UnitCheckResult::ok)
        .def_readonly("warnings", &mathllm::UnitCheckResult::warnings)
        .def_readonly("errors", &mathllm::UnitCheckResult::errors)
        .def_readonly("inferred_dimensions", &mathllm::UnitCheckResult::inferred_dimensions);
    
    m.def("unit_check", &mathllm::unit_check,
          py::arg("expr"), py::arg("symbol_dimensions"));
    
    py::class_<mathllm::ODEResult>(m, "ODEResult")
        .def_readonly("success", &mathllm::ODEResult::success)
        .def_readonly("t_values", &mathllm::ODEResult::t_values)
        .def_readonly("y_values", &mathllm::ODEResult::y_values)
        .def_readonly("steps_taken", &mathllm::ODEResult::steps_taken)
        .def_readonly("message", &mathllm::ODEResult::message);
    
    m.def("solve_ivp", &mathllm::solve_ivp,
          py::arg("expr"), py::arg("t0"), py::arg("t1"), py::arg("y0"),
          py::arg("symbols"),
          py::arg("rtol") = 1e-6,
          py::arg("atol") = 1e-8,
          py::arg("max_steps") = 1000);
}
