"""Tests for profile name extraction."""
import pytest

from core.memory import extract_person_name


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("my name is ram", "Ram"),
        ("My name is Ram Bahadur Thapa", "Ram Bahadur Thapa"),
        ("call me sita", "Sita"),
        ("i am called hari", "Hari"),
        ("मेरो नाम राम हो", "राम"),
        ("मेरो नाम सीता", "सीता"),
        # Descriptions must not be stored as names
        ("i am a student", None),
        ("i am from nepal", None),
        ("my name is a student", None),
        ("म नेपालमा बस्छु", None),
        ("my name is " + "x" * 50, None),
    ],
)
def test_extract_person_name(query: str, expected: str | None) -> None:
    assert extract_person_name(query) == expected
