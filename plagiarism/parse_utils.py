# plagiarism/parse_utils.py
import ast
import textwrap
from typing import Dict, Any


def safe_parse_module(src: str) -> ast.Module:
    """
    Parse `src` into an ast.Module. If a full parse fails (SyntaxError),
    attempt to parse top-level blocks (blocks that start at column 0)
    individually and include the successfully parsed nodes into a synthetic
    module. Line numbers of nodes are adjusted to match the original source.
    This allows skipping functions/classes with syntax errors while keeping
    the rest of the module analyzable.
    """
    try:
        return ast.parse(src)
    except SyntaxError:
        pass

    lines = src.splitlines(True)
    n = len(lines)
    module = ast.Module(body=[], type_ignores=[])
    i = 0
    while i < n:
        # skip empty lines
        if lines[i].strip() == "":
            i += 1
            continue
        start = i
        # find block end: next top-level (indent == 0) or EOF
        j = i + 1
        while j < n:
            if lines[j].strip() == "":
                j += 1
                continue
            indent = len(lines[j]) - len(lines[j].lstrip(" \t"))
            if indent == 0:
                break
            j += 1
        block_src = "".join(lines[start:j])
        try:
            parsed = ast.parse(block_src)
            # adjust parsed node lineno attributes to global coordinates
            ast.increment_lineno(parsed, start)
            module.body.extend(parsed.body)
        except Exception:
            # skip this block if it doesn't parse
            pass
        i = j
    return module


def get_node_source(node: ast.AST, src: str) -> str:
    """
    Try to extract the source segment for node from src. Prefer ast.get_source_segment,
    but fall back to slicing by lineno/end_lineno if needed.
    """
    try:
        seg = ast.get_source_segment(src, node)
        if seg is not None:
            return seg
    except Exception:
        pass
    lines = src.splitlines(True)
    lineno = getattr(node, "lineno", None)
    end_lineno = getattr(node, "end_lineno", None)
    if lineno is not None and end_lineno is not None:
        # lineno and end_lineno are 1-based
        start_i = max(0, lineno - 1)
        end_i = min(len(lines), end_lineno)
        return "".join(lines[start_i:end_i])
    return ""


def extract_code_hierarchy(src: str) -> Dict[str, Any]:
    """
    Return a hierarchical description of top-level elements in `src`.
    Structure:
      {
        "module": {"source": src},
        "functions": {name: {"node": node, "source": text, "lineno": int}},
        "classes": {name: {"node": node, "source": text, "lineno": int, "methods": { ... }}},
        "variables": {name: {"node": node, "source": text, "lineno": int}}
      }
    Blocks which failed to parse are skipped.
    """
    tree = safe_parse_module(src)
    out = {"module": {"source": src}, "functions": {}, "classes": {}, "variables": {}}
    for node in getattr(tree, "body", []):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            seg = get_node_source(node, src)
            out["functions"][node.name] = {"node": node, "source": seg, "lineno": getattr(node, "lineno", None)}
        elif isinstance(node, ast.ClassDef):
            seg = get_node_source(node, src)
            methods = {}
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    mseg = get_node_source(child, src)
                    methods[child.name] = {"node": child, "source": mseg, "lineno": getattr(child, "lineno", None)}
            out["classes"][node.name] = {
                "node": node,
                "source": seg,
                "lineno": getattr(node, "lineno", None),
                "methods": methods,
            }
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            names = []
            if isinstance(node, ast.Assign):
                targets = node.targets
            else:
                targets = [getattr(node, "target", None)] if getattr(node, "target", None) is not None else []
            for t in targets:
                if isinstance(t, ast.Name):
                    names.append(t.id)
                elif isinstance(t, (ast.Tuple, ast.List)):
                    for elt in getattr(t, "elts", []):
                        if isinstance(elt, ast.Name):
                            names.append(elt.id)
            seg = get_node_source(node, src)
            for n in names:
                out["variables"][n] = {"node": node, "source": seg, "lineno": getattr(node, "lineno", None)}
    return out
