from __future__ import annotations

import pytest
import sympy as sp

from mathllm.explain import ExplanationStyle, TalkerClient, TalkerConfig
from mathllm.guard import GuardConfig, preserve_explanation


class MockTalkerClient(TalkerClient):
    def __init__(self, config=None):
        super().__init__(config)
        self._force_failure = False
    
    def generate_explanation(self, problem_latex, result_latex, style, code_preview=None, objective=None):
        if self._force_failure:
            return f"The result is ${result_latex} + 1$ which is wrong."
        
        if style == ExplanationStyle.ONE_MINUTE:
            return f"We solved the problem. The result is ${result_latex}$. Quick and easy."
        elif style == ExplanationStyle.FRIENDLY:
            return f"So basically, we worked through this problem and got ${result_latex}$ as our answer!"
        elif style == ExplanationStyle.ORAL_EXAM:
            return f"To solve this problem, I applied the relevant mathematical operations. The verified result is ${result_latex}$. This follows from the standard procedures."
        else:
            return f"The verified result for this mathematical problem is ${result_latex}$. This solution has been confirmed through symbolic verification."
    
    def redraft_explanation(self, result_latex, previous_text, style):
        return f"Corrected: The exact result is ${result_latex}$ without modification."


def test_explanation_guard_passes_correct_latex():
    x = sp.Symbol("x")
    result_latex = "x^{2} + 2x"
    explanation = f"The derivative is ${result_latex}$. This follows from the power rule."
    
    guard_result = preserve_explanation(result_latex, explanation, [x])
    assert guard_result.ok


def test_explanation_guard_detects_altered_latex():
    x = sp.Symbol("x")
    result_latex = "x^{2}"
    altered_explanation = "The result is $x^{3}$. Note the cubic term."
    
    guard_result = preserve_explanation(result_latex, altered_explanation, [x])
    assert not guard_result.ok
    assert "latex_result_altered" in (guard_result.reason or "")


def test_explanation_guard_detects_altered_numeric():
    x = sp.Symbol("x")
    result_latex = "x + 5"
    altered_explanation = "The result is $x + 7$. The constant is 7."
    
    guard_result = preserve_explanation(result_latex, altered_explanation, [x])
    assert not guard_result.ok


def test_explanation_guard_no_latex_in_text():
    x = sp.Symbol("x")
    result_latex = "x^{2}"
    no_latex_explanation = "The result is just x squared without any LaTeX formatting."
    
    guard_result = preserve_explanation(result_latex, no_latex_explanation, [x])
    assert not guard_result.ok
    assert "no_latex_in_explanation" in (guard_result.reason or "")


def test_explanation_style_one_minute_is_short():
    client = MockTalkerClient()
    text = client.generate_explanation(
        problem_latex="\\int x dx",
        result_latex="\\frac{x^{2}}{2}",
        style=ExplanationStyle.ONE_MINUTE,
    )
    assert len(text) <= 260
    assert "result" in text.lower() or "solved" in text.lower()


def test_explanation_style_friendly_tone():
    client = MockTalkerClient()
    text = client.generate_explanation(
        problem_latex="\\frac{d}{dx} x^3",
        result_latex="3x^{2}",
        style=ExplanationStyle.FRIENDLY,
    )
    assert "basically" in text.lower() or "so" in text.lower() or "got" in text.lower()


def test_explanation_style_oral_exam_formal():
    client = MockTalkerClient()
    text = client.generate_explanation(
        problem_latex="x^2 = 4",
        result_latex="x = \\pm 2",
        style=ExplanationStyle.ORAL_EXAM,
    )
    assert "verified" in text.lower() or "applied" in text.lower() or "procedures" in text.lower()


def test_explanation_style_academic_formal():
    client = MockTalkerClient()
    text = client.generate_explanation(
        problem_latex="\\int \\sin x dx",
        result_latex="-\\cos x",
        style=ExplanationStyle.ACADEMIC,
    )
    assert "verified" in text.lower() or "result" in text.lower()
    assert len(text) >= 40


def test_explanation_redraft_on_guard_failure():
    client = MockTalkerClient()
    client._force_failure = True
    
    x = sp.Symbol("x")
    result_latex = "x^{2}"
    
    text = client.generate_explanation(
        problem_latex="\\frac{d}{dx} \\frac{x^3}{3}",
        result_latex=result_latex,
        style=ExplanationStyle.ACADEMIC,
    )
    
    guard_result = preserve_explanation(result_latex, text, [x])
    assert not guard_result.ok
    
    redrafted = client.redraft_explanation(
        result_latex=result_latex,
        previous_text=text,
        style=ExplanationStyle.ACADEMIC,
    )
    
    guard_result_after = preserve_explanation(result_latex, redrafted, [x])
    assert guard_result_after.ok


def test_explanation_cache_behavior():
    client = MockTalkerClient()
    
    problem = "\\int x dx"
    result = "\\frac{x^{2}}{2}"
    style = ExplanationStyle.ACADEMIC
    
    text1 = client.generate_explanation(problem, result, style)
    cache_key = client._make_cache_key(problem, result, style)
    assert cache_key in client._cache
    
    text2 = client.generate_explanation(problem, result, style)
    assert text1 == text2


def test_explanation_different_styles_different_cache():
    client = MockTalkerClient()
    
    problem = "\\int x dx"
    result = "\\frac{x^{2}}{2}"
    
    text_academic = client.generate_explanation(problem, result, ExplanationStyle.ACADEMIC)
    text_friendly = client.generate_explanation(problem, result, ExplanationStyle.FRIENDLY)
    
    assert text_academic != text_friendly


def test_explanation_numeric_preservation():
    x = sp.Symbol("x")
    result_latex = "2.5 x + 3.14"
    explanation = "The result is $2.5 x + 3.14$. Note the coefficients 2.5 and 3.14."
    
    guard_result = preserve_explanation(result_latex, explanation, [x])
    assert guard_result.ok


def test_explanation_fraction_preservation():
    x = sp.Symbol("x")
    result_latex = "\\frac{x^{2}}{2} + \\frac{1}{3}"
    explanation = f"The antiderivative is ${result_latex}$."
    
    guard_result = preserve_explanation(result_latex, explanation, [x])
    assert guard_result.ok
