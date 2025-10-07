from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

from .llm_student import GenerationResult, StudentLLM
from .mir import MIRProblem, Objective

LOGGER = logging.getLogger(__name__)

ALLOWED_TOOLS = {"integrate", "diff", "solve_equation", "verify_equal", "simplify", "ode_solve_stub"}
STEP_TYPES = {"derive", "tool_call", "verify", "final"}

PLAN_PROMPT_TEMPLATE = (
    "You are a math planner. Return ONLY valid JSON with a 'steps' array.\n"
    "Allowed tools: integrate, diff, solve_equation, verify_equal, simplify, ode_solve_stub.\n"
    "Use SymPy-like strings in args. After each tool_call add a verify step.\n"
    "Problem (LaTeX): {latex}\n"
    "MIR summary: {mir_summary}\n"
    "Objective: {objective}\n"
    "JSON schema:\n"
    '{{ "steps": [ {{ "type": "derive"|"tool_call"|"verify"|"final", ... }} ] }}'
)

REPAIR_PROMPT_TEMPLATE = (
    "Previous plan failed at step {step_index} with error: {error}. Context: {context}.\n"
    "Return ONLY corrected JSON plan from step {step_index} onward. Keep schema & allowed tools."
)

JSON_REGEX = re.compile(r"\{.*\}", re.DOTALL)


class PlanError(RuntimeError):
    pass


@dataclass
class PlanStep:
    type: str
    payload: Dict[str, Any]


@dataclass
class Plan:
    steps: List[PlanStep]
    raw_text: str
    generation: Optional[GenerationResult] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def tool_calls(self) -> Iterable[PlanStep]:
        return (step for step in self.steps if step.type == "tool_call")


def summarize_mir(problem: MIRProblem) -> str:
    summary = {
        "objective": problem.objective.value,
        "expr": str(problem.expr.sympy_expr),
        "variables": [str(sym) for sym in problem.variables],
        "constraints": [str(c) for c in problem.constraints] if problem.constraints else None,
        "assumptions": problem.expr.assumptions,
    }
    return json.dumps(summary, ensure_ascii=False)


class Planner:
    def __init__(self, student: StudentLLM, *, tool_whitelist: Optional[Iterable[str]] = None) -> None:
        self.student = student
        self.tool_whitelist = set(tool_whitelist or ALLOWED_TOOLS)

    def propose(self, problem: MIRProblem, *, latex: str, objective: Optional[Objective] = None,
                extra_context: Optional[str] = None) -> Plan:
        prompt = self._build_plan_prompt(problem, latex=latex, objective=objective or problem.objective,
                                         extra_context=extra_context)
        generation = self.student.generate_plan(prompt)
        plan = self._parse_plan(generation.raw.get("completion") or generation.text)
        plan.generation = generation
        plan.metadata.setdefault("origin", "student")
        plan.metadata.setdefault("mode", "propose")
        return plan

    def repair(self, failed_plan: Plan, step_index: int, error: str, *, context: Optional[str] = None) -> Plan:
        context_snip = context or failed_plan.raw_text
        prompt = self._build_repair_prompt(step_index, error, context_snip)
        generation = self.student.repair_plan(prompt)
        plan = self._parse_plan(generation.raw.get("completion") or generation.text)
        plan.generation = generation
        plan.metadata.setdefault("origin", "student")
        plan.metadata["mode"] = "repair"
        plan.metadata["repair_step"] = step_index
        return plan

    def _build_plan_prompt(self, problem: MIRProblem, *, latex: str, objective: Objective,
                           extra_context: Optional[str]) -> str:
        mir_summary = summarize_mir(problem)
        prompt = PLAN_PROMPT_TEMPLATE.format(latex=latex, mir_summary=mir_summary, objective=objective.value)
        if extra_context:
            prompt += f"\nContext: {extra_context}"
        return prompt

    @staticmethod
    def _build_repair_prompt(step_index: int, error: str, context: str) -> str:
        return REPAIR_PROMPT_TEMPLATE.format(step_index=step_index, error=error, context=context)

    def _parse_plan(self, text: str) -> Plan:
        cleaned = self._extract_json_block(text)
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as exc:  # pragma: no cover, validated via unit tests
            LOGGER.error("Planner JSON decode failed: %s", cleaned)
            raise PlanError(f"Invalid planner JSON: {exc}") from exc
        steps = parsed.get("steps")
        if not isinstance(steps, list) or not steps:
            raise PlanError("Planner output missing steps")
        validated_steps = self._validate_steps(steps)
        return Plan(steps=validated_steps, raw_text=cleaned)

    def _validate_steps(self, steps: List[Dict[str, Any]]) -> List[PlanStep]:
        validated: List[PlanStep] = []
        last_type: Optional[str] = None
        for index, step in enumerate(steps):
            if not isinstance(step, dict):
                raise PlanError(f"Step {index} is not an object")
            step_type = step.get("type")
            if step_type not in STEP_TYPES:
                raise PlanError(f"Unsupported step type: {step_type}")
            payload = {k: v for k, v in step.items() if k != "type"}
            if step_type == "tool_call":
                self._validate_tool_step(payload, index)
                last_type = "tool_call"
            elif step_type == "verify":
                self._validate_verify_step(payload, index)
                if last_type != "tool_call":
                    raise PlanError(f"Verify step at {index} must follow a tool_call")
                last_type = "verify"
            elif step_type == "final":
                self._validate_final_step(payload, index)
                last_type = "final"
            else:  # derive or others
                last_type = step_type
            validated.append(PlanStep(type=step_type, payload=payload))
        if validated[-1].type != "final":
            raise PlanError("Plan must terminate with a final step")
        return validated

    def _validate_tool_step(self, payload: Dict[str, Any], index: int) -> None:
        tool = payload.get("tool")
        args = payload.get("args")
        bind = payload.get("bind")
        if tool not in self.tool_whitelist:
            raise PlanError(f"Tool '{tool}' not permitted at step {index}")
        if not isinstance(args, dict):
            raise PlanError(f"Tool step {index} args must be object")
        if not isinstance(bind, str) or not bind:
            raise PlanError(f"Tool step {index} requires non-empty bind")

    @staticmethod
    def _validate_verify_step(payload: Dict[str, Any], index: int) -> None:
        lhs = payload.get("lhs")
        rhs = payload.get("rhs")
        if not isinstance(lhs, str) or not isinstance(rhs, str):
            raise PlanError(f"Verify step {index} must include lhs and rhs strings")

    @staticmethod
    def _validate_final_step(payload: Dict[str, Any], index: int) -> None:
        result = payload.get("result")
        if not isinstance(result, str) or not result:
            raise PlanError(f"Final step {index} missing result binding")

    @staticmethod
    def _extract_json_block(text: str) -> str:
        stripped = text.strip()
        if stripped.startswith("```") and stripped.endswith("```"):
            stripped = stripped.strip("`")
            if stripped.startswith("json"):
                stripped = stripped[4:]
            stripped = stripped.strip()
        match = JSON_REGEX.search(stripped)
        if not match:
            raise PlanError("Planner response did not contain JSON object")
        return match.group(0)