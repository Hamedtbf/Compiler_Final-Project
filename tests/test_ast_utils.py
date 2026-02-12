# tests/test_ast_utils.py
from plagiarism.ast_utils import ast_similarity

def test_ast_similarity_same():
    a = "def f(x):\n    return x + 1\n"
    b = "def f(y):\n    return y + 1\n"
    cfg = {"ast": {"method": "levenshtein"}}
    s = ast_similarity(a, b, cfg)
    assert s > 0.8
