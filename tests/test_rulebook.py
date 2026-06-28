# tests/test_rulebook.py
from ludus.rulebook import Rulebook


def test_add_and_render():
    rb = Rulebook()
    assert rb.add("avoid holes") is True
    assert rb.add("clear lines") is True
    assert rb.render() == "- avoid holes\n- clear lines"


def test_dedup():
    rb = Rulebook()
    rb.add("avoid holes")
    assert rb.add("avoid holes") is False
    assert rb.rules() == ["avoid holes"]


def test_none_is_ignored():
    rb = Rulebook()
    assert rb.add(None) is False
    assert rb.rules() == []


def test_caps_at_max_evicting_oldest():
    rb = Rulebook(max_rules=2)
    rb.add("a"); rb.add("b"); rb.add("c")
    assert rb.rules() == ["b", "c"]
