from fractions import Fraction

import pytest

from rewardscope import ExtractionStatus, extract_numeric_answer, parse_numeric_value


@pytest.mark.parametrize(
    ("response", "raw_answer", "normalized_answer", "parsed_value"),
    [
        ("The final answer is 42.", "42.", "42", Fraction(42)),
        ("Answer: -3", "-3", "-3", Fraction(-3)),
        ("#### 0.500", "0.500", "1/2", Fraction(1, 2)),
        ("The final answer is 1/2", "1/2", "1/2", Fraction(1, 2)),
        (
            r"The final answer is \frac{1}{2}",
            r"\frac{1}{2}",
            "1/2",
            Fraction(1, 2),
        ),
    ],
)
def test_explicit_final_answers_are_exactly_normalized(
    response, raw_answer, normalized_answer, parsed_value
):
    result = extract_numeric_answer(response)

    assert result.raw_answer == raw_answer
    assert result.normalized_answer == normalized_answer
    assert result.parsed_value == parsed_value
    assert result.extraction_status is ExtractionStatus.EXPLICIT_FINAL
    assert result.extraction_ok is True
    assert result.format_ok is True


def test_boxed_answer_is_extracted_with_nested_latex_fraction():
    result = extract_numeric_answer(r"Work shown here. \boxed{\frac{2}{4}}")

    assert result.raw_answer == r"\frac{2}{4}"
    assert result.normalized_answer == "1/2"
    assert result.parsed_value == Fraction(1, 2)
    assert result.extraction_status is ExtractionStatus.BOXED
    assert result.format_ok is True


def test_last_single_numeric_line_is_an_implicit_terminal_answer():
    result = extract_numeric_answer("We add the values.\n42")

    assert result.normalized_answer == "42"
    assert result.extraction_status is ExtractionStatus.IMPLICIT_TERMINAL
    assert result.format_ok is False


def test_terminal_equation_extracts_its_right_hand_side():
    result = extract_numeric_answer("We add the values.\n2 + 40 = 42")

    assert result.raw_answer == "42"
    assert result.parsed_value == Fraction(42)
    assert result.extraction_status is ExtractionStatus.IMPLICIT_TERMINAL


@pytest.mark.parametrize(
    "response",
    [
        "The answer may be 41 or 42.",
        "2 + 40 = 42, although another interpretation gives 43.",
        "x = 41 or x = 42",
        r"\boxed{41} and \boxed{42}",
    ],
)
def test_conflicting_candidates_are_ambiguous(response):
    result = extract_numeric_answer(response)

    assert result.extraction_status is ExtractionStatus.AMBIGUOUS
    assert result.extraction_ok is False
    assert result.raw_answer is None


def test_response_without_any_numeric_candidate_is_missing():
    result = extract_numeric_answer("I cannot solve this.")

    assert result.extraction_status is ExtractionStatus.MISSING
    assert result.extraction_ok is False


@pytest.mark.parametrize("response", ["Answer: sqrt(2)", r"\boxed{\sqrt{2}}"])
def test_unparseable_answer_candidate_is_a_parse_error(response):
    result = extract_numeric_answer(response)

    assert result.extraction_status is ExtractionStatus.PARSE_ERROR
    assert result.extraction_ok is False
    assert result.raw_answer is not None


@pytest.mark.parametrize("response", ["Answer: 1/0", r"\boxed{\frac{1}{0}}"])
def test_zero_denominator_is_a_parse_error(response):
    result = extract_numeric_answer(response)

    assert result.extraction_status is ExtractionStatus.PARSE_ERROR


def test_later_explicit_final_answer_overrides_earlier_reasoning_numbers():
    result = extract_numeric_answer(
        "I first got 42, but I reconsidered.\nThe final answer is 43."
    )

    assert result.normalized_answer == "43"


def test_response_must_be_a_string():
    with pytest.raises(TypeError, match="response must be a string"):
        extract_numeric_answer(42)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("raw_answer", "expected_value"),
    [
        ("0.500", Fraction(1, 2)),
        ("2/4", Fraction(1, 2)),
        (r"\frac{1}{2}", Fraction(1, 2)),
    ],
)
def test_parse_numeric_value_uses_exact_fraction_arithmetic(raw_answer, expected_value):
    assert parse_numeric_value(raw_answer) == expected_value
