import ast
import difflib

try:
    import zss
    ZSS_AVAILABLE = True
except Exception:
    ZSS_AVAILABLE = False


class NameNormalizer(ast.NodeTransformer):
    def __init__(self, protected_names=None):
        super().__init__()
        self.protected = set(protected_names or [])
        self.map = {}
        self.counter = 0

    def _map(self, original):
        if original in self.protected:
            return original
        if original in ("True", "False", "None"):
            return original
        if original.startswith('__') and original.endswith('__'):
            return original
        if original not in self.map:
            self.counter += 1
            self.map[original] = f"VAR_{self.counter}"
        return self.map[original]

    def visit_Name(self, node):
        try:
            new_id = self._map(node.id)
            return ast.copy_location(ast.Name(id=new_id, ctx=node.ctx), node)
        except Exception:
            return node

    def visit_arg(self, node):
        if hasattr(node, "arg"):
            node.arg = self._map(node.arg)
        return node

    def visit_FunctionDef(self, node):
        if node.name not in self.protected:
            node.name = self._map(node.name)
        self.generic_visit(node)
        return node

    def visit_ClassDef(self, node):
        if node.name not in self.protected:
            node.name = self._map(node.name)
        self.generic_visit(node)
        return node


def ast_dump_normalized(src, protected_names=None):
    try:
        tree = ast.parse(src)
    except Exception:
        return ""
    normalizer = NameNormalizer(protected_names)
    norm_tree = normalizer.visit(tree)
    ast.fix_missing_locations(norm_tree)
    return ast.dump(norm_tree, include_attributes=False)


def ast_similarity_levenshtein(src_a, src_b, protected_names_a=None, protected_names_b=None):
    dump_a = ast_dump_normalized(src_a, protected_names_a)
    dump_b = ast_dump_normalized(src_b, protected_names_b)
    if not dump_a and not dump_b:
        return 1.0
    sm = difflib.SequenceMatcher(None, dump_a, dump_b)
    return sm.ratio()


# zss/zhang-shasha method
def _ast_to_zss(node):
    nd = zss.Node(node.__class__.__name__)
    # add extra label info for names & function/class names
    if isinstance(node, ast.Name):
        nd.label += f":{getattr(node, 'id', '')}"
    elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        nd.label += f":{getattr(node, 'name', '')}"
    elif isinstance(node, ast.arg):
        nd.label += f":{getattr(node, 'arg', '')}"
    for field, value in ast.iter_fields(node):
        if isinstance(value, ast.AST):
            nd.addkid(_ast_to_zss(value))
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, ast.AST):
                    nd.addkid(_ast_to_zss(item))
    return nd


def _zss_tree_size(node):
    if node is None:
        return 0
    cnt = 1
    for c in getattr(node, "children", []):
        cnt += _zss_tree_size(c)
    return cnt


def ast_similarity_zhang_shasha(src_a, src_b, protected_names_a=None, protected_names_b=None):
    if not ZSS_AVAILABLE:
        raise RuntimeError("zss library not available")
    try:
        tree_a = ast.parse(src_a)
        tree_b = ast.parse(src_b)
    except Exception:
        return 0.0
    norm_a = NameNormalizer(protected_names_a).visit(tree_a)
    norm_b = NameNormalizer(protected_names_b).visit(tree_b)
    ast.fix_missing_locations(norm_a)
    ast.fix_missing_locations(norm_b)
    za = _ast_to_zss(norm_a)
    zb = _ast_to_zss(norm_b)
    def label_dist(a, b):
        return 0 if a == b else 1
    ted = zss.simple_distance(za, zb, get_children=lambda n: n.children, label_dist=label_dist)
    size_a = _zss_tree_size(za)
    size_b = _zss_tree_size(zb)
    denom = float(size_a + size_b) if (size_a + size_b) > 0 else 1.0
    sim = 1.0 - (ted / denom)
    return max(0.0, min(1.0, sim))


def ast_similarity(src_a, src_b, cfg, protected_a=None, protected_b=None):
    method = cfg.get("ast", {}).get("method", "levenshtein")
    if method == "zhang_shasha":
        if not ZSS_AVAILABLE:
            print("[ast] zss not available; falling back to levenshtein")
        else:
            try:
                return ast_similarity_zhang_shasha(src_a, src_b, protected_a, protected_b)
            except Exception as e:
                print(f"[ast] zhang_shasha failed: {e}; falling back to levenshtein")
    return ast_similarity_levenshtein(src_a, src_b, protected_a, protected_b)
