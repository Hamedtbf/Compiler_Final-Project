# tests/test_cfg_builder.py
import pytest
from plagiarism.cfg_builder import CFGBuilder, export_cfgs_with_graph

try:
    import networkx as nx
    NX = True
except Exception:
    NX = False

@pytest.mark.skipif(not NX, reason="networkx not installed")
def test_cfg_builder_function_and_method():
    src = (
        "def f():\n"
        "    x = 1\n"
        "class C:\n"
        "    def m(self):\n"
        "        return 2\n"
    )
    import ast
    tree = ast.parse(src)
    b = CFGBuilder()
    res = b.build_for_module(tree, module_name="mod")
    exported = export_cfgs_with_graph(res, b.G)
    assert "f" in exported
    assert "C" in exported
    assert "C.m" in exported
    # subgraph for method
    Gm = exported["C.m"]["global_graph"].subgraph(exported["C.m"]["nodes"]).copy()
    assert len(Gm.nodes()) >= 1
