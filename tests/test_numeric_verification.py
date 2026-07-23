from fractions import Fraction

import pytest

from rewardscope import (
    ExtractionStatus,
    extract_numeric_answer,
    verify_extracted_numeric_answer,
    verify_numeric_answer,
)


@pytest.mark.parametrize(
    ("response", "ground_truth", "expected_value"),
    [
        ("The final answer is 0.5.", "1/2", Fraction(1, 2)),
        (r"\boxed{\frac{2}{4}}", "0.500", Fraction(1, 2)),
        ("#### -3", "-3.0", Fraction(-3)),
    ],
)
def test_mathematically_equivalent_answers_are_correct(
    response, ground_truth, expected_value
):
    result = verify_numeric_answer(response, ground_truth)

    assert result.is_correct is True
    assert result.error_type is None
    assert result.extraction.parsed_value == expected_value


def test_wrong_numeric_answer_has_a_distinct_error_type():
    result = verify_numeric_answer("The final answer is 42.", "43")

    assert result.is_correct is False
    assert result.error_type == "wrong_answer"


def test_implicit_terminal_answer_can_be_correct_without_format_compliance():
    result = verify_numeric_answer("2 + 40 = 42", "42")

    assert result.is_correct is True
    assert result.extraction.extraction_status is ExtractionStatus.IMPLICIT_TERMINAL
    assert result.extraction.format_ok is False


@pytest.mark.parametrize(
    ("response", "expected_error_type"),
    [
        ("The answer may be 41 or 42.", "ambiguous_answer"),
        ("I cannot solve this.", "missing_answer"),
        ("Answer: sqrt(2)", "answer_parse_error"),
    ],
)
def test_extraction_failures_remain_distinguishable_in_verification(
    response, expected_error_type
):
    result = verify_numeric_answer(response, "42")

    assert result.is_correct is False
    assert result.error_type == expected_error_type


def test_invalid_ground_truth_is_not_reported_as_a_model_error():
    result = verify_numeric_answer("The final answer is 42.", "sqrt(42)")

    assert result.is_correct is False
    assert result.error_type == "invalid_ground_truth"


def test_existing_extraction_can_be_verified_without_reextracting_response():
    extraction = extract_numeric_answer("The final answer is 2/4")

    result = verify_extracted_numeric_answer(extraction, "0.5")

    assert result.is_correct is True
    assert result.extraction is extraction


@pytest.mark.parametrize(
    ("extraction", "ground_truth", "error_message"),
    [
        (None, "42", "extraction must be an ExtractionResult"),
        (extract_numeric_answer("Answer: 42"), None, "ground_truth must be a string"),
    ],
)
def test_verifier_validates_its_input_types(extraction, ground_truth, error_message):
    with pytest.raises(TypeError, match=error_message):
        verify_extracted_numeric_answer(extraction, ground_truth)  # type: ignore[arg-type]
