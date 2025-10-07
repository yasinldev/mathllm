import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "cpp", "build"))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import mathcore


def test_diff():
    assert mathcore.diff("x^2", "x") == "2*x"


def test_integrate():
    assert mathcore.integrate("2*x", "x") == "x**2"


def test_verify_equal_true():
    assert mathcore.verify_equal("x^2", "x*x") is True


def test_verify_equal_false():
    assert mathcore.verify_equal("x^2", "x^3") is False


def test_solve_equation_linear():
    assert mathcore.solve_equation("x", "3", "x") == "[3]"
