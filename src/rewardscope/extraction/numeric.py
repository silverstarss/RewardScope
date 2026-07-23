"""Exact numeric answer extraction for rollout responses."""

from __future__ import annotations

import re
from fractions import Fraction

from rewardscope.schemas import ExtractionResult, ExtractionStatus


_EXPLICIT_PATTERNS = (
    re.compile(
        r"(?i)\b(?:the\s+)?final\s+answer\s*(?:is|:|=)\s*(?P<answer>.+?)\s*$"
    ),
    re.compile(r"(?i)^\s*answer\s*(?:is|:|=)\s*(?P<answer>.+?)\s*$"),
    re.compile(r"^\s*####\s*(?P<answer>.+?)\s*$"),
)
_LATEX_FRACTION_PATTERN = re.compile(
    r"\\(?:d?frac)\s*\{\s*([+-]?\d+)\s*\}\s*\{\s*([+-]?\d+)\s*\}"
)
_NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)"
_SIMPLE_FRACTION_PATTERN = re.compile(
    rf"(?P<numerator>{_NUMBER})\s*/\s*(?P<denominator>{_NUMBER})"
)
_NUMBER_PATTERN = re.compile(_NUMBER)
_NUMERIC_CANDIDATE_PATTERN = re.compile(
    rf"(?<![\w.])({_NUMBER}(?:\s*/\s*{_NUMBER})?)(?![\w.])"
)


def extract_numeric_answer(response: str) -> ExtractionResult:
    """Extract one exact numeric answer using answer-source precedence rules.

    The result retains the original candidate and a canonical ``Fraction`` value,
    so equivalent forms such as ``0.5`` and ``2/4`` compare exactly.
    """
    if not isinstance(response, str):
        raise TypeError("response must be a string.")

    explicit_answer = _find_last_explicit_answer(response)
    if explicit_answer is not None:
        return _result_from_candidate(
            explicit_answer, ExtractionStatus.EXPLICIT_FINAL
        )

    boxed_answers, malformed_box = _find_boxed_answers(response)
    if boxed_answers:
        return _result_from_boxed_answers(boxed_answers)
    if malformed_box:
        return _parse_error_result(malformed_box)

    terminal_line = _last_non_empty_line(response)
    if terminal_line is not None:
        terminal_result = _try_terminal_answer(terminal_line)
        if terminal_result is not None:
            return terminal_result

    if len(_NUMERIC_CANDIDATE_PATTERN.findall(response)) >= 2:
        return _failed_result(ExtractionStatus.AMBIGUOUS)
    return _failed_result(ExtractionStatus.MISSING)


def _find_last_explicit_answer(response: str) -> str | None:
    answer: str | None = None
    for line in response.splitlines():
        for pattern in _EXPLICIT_PATTERNS:
            match = pattern.search(line)
            if match:
                answer = match.group("answer")
                break
    return answer


def _find_boxed_answers(response: str) -> tuple[list[str], str | None]:
    answers: list[str] = []
    cursor = 0
    marker = r"\boxed{"

    while (start := response.find(marker, cursor)) != -1:
        content_start = start + len(marker)
        depth = 1
        index = content_start
        while index < len(response) and depth:
            if response[index] == "{":
                depth += 1
            elif response[index] == "}":
                depth -= 1
            index += 1

        if depth:
            return answers, response[content_start:]

        answers.append(response[content_start : index - 1])
        cursor = index

    return answers, None


def _result_from_boxed_answers(boxed_answers: list[str]) -> ExtractionResult:
    parsed_answers: list[tuple[str, str, Fraction]] = []
    for answer in boxed_answers:
        parsed = _parse_numeric_candidate(answer)
        if parsed is None:
            return _parse_error_result(answer)
        parsed_answers.append((answer, *parsed))

    values = {parsed_value for _, _, parsed_value in parsed_answers}
    if len(values) != 1:
        return _failed_result(ExtractionStatus.AMBIGUOUS)

    raw_answer, normalized_answer, parsed_value = parsed_answers[-1]
    return ExtractionResult(
        raw_answer=raw_answer,
        normalized_answer=normalized_answer,
        parsed_value=parsed_value,
        extraction_status=ExtractionStatus.BOXED,
        format_ok=True,
    )


def _try_terminal_answer(terminal_line: str) -> ExtractionResult | None:
    parsed = _parse_numeric_candidate(terminal_line)
    if parsed is not None:
        return _successful_result(
            terminal_line, parsed, ExtractionStatus.IMPLICIT_TERMINAL
        )

    if terminal_line.count("=") == 1:
        _, right_hand_side = terminal_line.split("=", maxsplit=1)
        candidate = right_hand_side.strip()
        if candidate:
            parsed = _parse_numeric_candidate(candidate)
            if parsed is not None:
                return _successful_result(
                    candidate, parsed, ExtractionStatus.IMPLICIT_TERMINAL
                )
            if len(_NUMERIC_CANDIDATE_PATTERN.findall(candidate)) >= 2:
                return None
            return _parse_error_result(candidate)

    return None


def _result_from_candidate(
    raw_answer: str, extraction_status: ExtractionStatus
) -> ExtractionResult:
    parsed = _parse_numeric_candidate(raw_answer)
    if parsed is None:
        return _parse_error_result(raw_answer)
    return _successful_result(raw_answer, parsed, extraction_status)


def _successful_result(
    raw_answer: str,
    parsed: tuple[str, Fraction],
    extraction_status: ExtractionStatus,
) -> ExtractionResult:
    normalized_answer, parsed_value = parsed
    return ExtractionResult(
        raw_answer=raw_answer,
        normalized_answer=normalized_answer,
        parsed_value=parsed_value,
        extraction_status=extraction_status,
        format_ok=extraction_status
        in {ExtractionStatus.EXPLICIT_FINAL, ExtractionStatus.BOXED},
    )


def _parse_numeric_candidate(raw_answer: str) -> tuple[str, Fraction] | None:
    candidate = raw_answer.strip().strip("$").strip()
    candidate = _strip_terminal_period(candidate)

    latex_fraction = _LATEX_FRACTION_PATTERN.fullmatch(candidate)
    if latex_fraction:
        numerator, denominator = latex_fraction.groups()
        try:
            return _normalize_fraction(Fraction(int(numerator), int(denominator)))
        except ZeroDivisionError:
            return None

    simple_fraction = _SIMPLE_FRACTION_PATTERN.fullmatch(candidate)
    if simple_fraction:
        numerator = Fraction(simple_fraction.group("numerator"))
        denominator = Fraction(simple_fraction.group("denominator"))
        if denominator == 0:
            return None
        return _normalize_fraction(numerator / denominator)

    if _NUMBER_PATTERN.fullmatch(candidate):
        return _normalize_fraction(Fraction(candidate))
    return None


def _normalize_fraction(value: Fraction) -> tuple[str, Fraction]:
    return str(value), value


def _strip_terminal_period(candidate: str) -> str:
    if candidate.endswith(".") and candidate.count(".") == 1:
        return candidate[:-1]
    return candidate


def _last_non_empty_line(response: str) -> str | None:
    for line in reversed(response.splitlines()):
        if line.strip():
            return line.strip()
    return None


def _failed_result(extraction_status: ExtractionStatus) -> ExtractionResult:
    return ExtractionResult(
        raw_answer=None,
        normalized_answer=None,
        parsed_value=None,
        extraction_status=extraction_status,
        format_ok=False,
    )


def _parse_error_result(raw_answer: str) -> ExtractionResult:
    return ExtractionResult(
        raw_answer=raw_answer,
        normalized_answer=None,
        parsed_value=None,
        extraction_status=ExtractionStatus.PARSE_ERROR,
        format_ok=False,
    )
