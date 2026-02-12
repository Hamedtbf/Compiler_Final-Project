# tests/test_parse_utils.py
import pytest
from plagiarism.parse_utils import safe_parse_module, extract_code_hierarchy

def test_safe_parse_module_skips_bad_blocks():
    src = (
        "def good():\n"
        "    return 1\n\n"
        "def bad(\n"  # syntax error here
        "    return 2\n\n"
        "def good2():\n"
        "    return 3\n"
    )
    tree = safe_parse_module(src)
    names = [n.name for n in tree.body if hasattr(n, "name")]
    assert "good" in names
    assert "good2" in names
    assert "bad" not in names

def test_extract_code_hierarchy_basic():
    src = (
        "import os\n"
        "x = 1\n"
        "def f():\n"
        "    return x\n"
        "class C:\n"
        "    def m(self):\n"
        "        return 2\n"
    )
    h = extract_code_hierarchy(src)
    assert "f" in h["functions"]
    assert "C" in h["classes"]
    assert "m" in h["classes"]["C"]["methods"]
    assert "x" in h["variables"]
