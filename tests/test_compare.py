# tests/test_compare.py
import tempfile
from plagiarism.compare import compare_two_files, compare_hierarchies
from plagiarism.config import DEFAULT_CONFIG

def write_temp(src):
    f = tempfile.NamedTemporaryFile("w", delete=False, suffix=".py", encoding="utf-8")
    f.write(src)
    f.flush()
    f.close()
    return f.name

def test_compare_two_simple_files():
    a = "def f():\n    return 1\n"
    b = "def f():\n    return 1\n"
    fa = write_temp(a)
    fb = write_temp(b)
    cfg = DEFAULT_CONFIG.copy()
    res = compare_two_files(fa, fb, cfg)
    assert "token" in res and "ast" in res and "cfg" in res and "final" in res

def test_compare_hierarchy_basic():
    a = (
        "x = 1\n"
        "def f():\n"
        "    return x\n"
        "class C:\n"
        "    def m(self):\n"
        "        return 2\n"
    )
    b = a  # identical
    fa = write_temp(a)
    fb = write_temp(b)
    cfg = DEFAULT_CONFIG.copy()
    report = compare_hierarchies(fa, fb, cfg)
    assert "functions" in report and "classes" in report and "variables" in report
    assert "f" in report["functions"]
    assert "C" in report["classes"]
