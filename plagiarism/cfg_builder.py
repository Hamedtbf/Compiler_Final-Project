# plagiarism/cfg_builder.py
import ast
from collections import defaultdict

try:
    import networkx as nx
    NX_AVAILABLE = True
except Exception:
    NX_AVAILABLE = False

if not NX_AVAILABLE:
    print("[warning] networkx not available; CFG features will be limited.")


class CFGBuilder:
    def __init__(self):
        if not NX_AVAILABLE:
            raise RuntimeError("networkx required for CFGBuilder")
        self.G = nx.DiGraph()
        self._id = 0
        self.current_function = None
        self.loop_stack = []

    def _new_block(self, function_name=None):
        bid = f"B{self._id}"
        self._id += 1
        self.G.add_node(bid, stmts=[], label="", function=function_name)
        return bid

    def _stmt_label(self, stmt):
        if isinstance(stmt, ast.If):
            return "If"
        if isinstance(stmt, (ast.For, ast.AsyncFor)):
            return "For"
        if isinstance(stmt, ast.While):
            return "While"
        if isinstance(stmt, ast.Return):
            return "Return"
        if isinstance(stmt, ast.Break):
            return "Break"
        if isinstance(stmt, ast.Continue):
            return "Continue"
        if isinstance(stmt, ast.Expr):
            if isinstance(stmt.value, ast.Call):
                return "Call"
            return "Expr"
        if isinstance(stmt, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            return "Assign"
        if isinstance(stmt, ast.Try):
            return "Try"
        if isinstance(stmt, ast.With):
            return "With"
        return stmt.__class__.__name__

    def _add_stmt_to_block(self, bid, stmt):
        self.G.nodes[bid]['stmts'].append(stmt)
        labels = [self._stmt_label(s) for s in self.G.nodes[bid]['stmts']]
        lbl = "|".join(labels[-5:])
        self.G.nodes[bid]['label'] = lbl

    def _connect(self, a, b):
        if not self.G.has_node(a):
            self.G.add_node(a, stmts=[], label="", function=self.current_function)
        if not self.G.has_node(b):
            self.G.add_node(b, stmts=[], label="", function=self.current_function)
        if not self.G.has_edge(a, b):
            self.G.add_edge(a, b)

    def build_for_module(self, module_tree, module_name="__module__"):
        self.current_function = module_name
        results = {}
        module_entry = self._new_block(module_name)
        module_exits = self._process_statements(module_tree.body, module_entry)
        results[module_name] = {"entry": module_entry, "exits": list(module_exits)}
        # create per-function / per-class entries (including methods)
        for node in module_tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                name = node.name
                self.current_function = name
                entry = self._new_block(name)
                body = node.body
                exits = self._process_statements(body, entry)
                results[name] = {"entry": entry, "exits": list(exits)}
            elif isinstance(node, ast.ClassDef):
                # create entry for the class itself
                name = node.name
                self.current_function = name
                entry = self._new_block(name)
                body = node.body
                exits = self._process_statements(body, entry)
                results[name] = {"entry": entry, "exits": list(exits)}
                # also create entries for methods inside the class (name as ClassName.method)
                for child in node.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        method_name = f"{name}.{child.name}"
                        self.current_function = method_name
                        m_entry = self._new_block(method_name)
                        m_exits = self._process_statements(child.body, m_entry)
                        results[method_name] = {"entry": m_entry, "exits": list(m_exits)}
        # collect nodes per function
        fn_nodes = defaultdict(list)
        for n, data in self.G.nodes(data=True):
            fn_nodes[data.get('function', module_name)].append(n)
        for fn, info in results.items():
            info['nodes'] = fn_nodes.get(fn, [])
        # compress blocks
        self._compress_blocks()
        # refresh nodes mapping after compression
        fn_nodes = defaultdict(list)
        for n, data in self.G.nodes(data=True):
            fn_nodes[data.get('function', module_name)].append(n)
        for fn, info in results.items():
            info['nodes'] = fn_nodes.get(fn, [])
        return results

    def _process_statements(self, stmts, entry_block):
        exits = [entry_block]
        for stmt in stmts:
            new_exits = []
            for e in exits:
                out = self._process_statement(e, stmt)
                new_exits.extend(out)
            exits = new_exits
            if not exits:
                break
        return exits

    def _process_statement(self, entry_block, stmt):
        simple_stmts = (ast.Assign, ast.AugAssign, ast.AnnAssign, ast.Expr, ast.Pass,
                        ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal)
        if isinstance(stmt, simple_stmts):
            self._add_stmt_to_block(entry_block, stmt)
            return [entry_block]

        if isinstance(stmt, ast.Return):
            self._add_stmt_to_block(entry_block, stmt)
            return []

        if isinstance(stmt, ast.If):
            cond_block = self._new_block(self.current_function)
            self._connect(entry_block, cond_block)
            self._add_stmt_to_block(cond_block, stmt)
            body_entry = self._new_block(self.current_function)
            self._connect(cond_block, body_entry)
            body_exits = self._process_statements(stmt.body, body_entry)
            if stmt.orelse:
                else_entry = self._new_block(self.current_function)
                self._connect(cond_block, else_entry)
                else_exits = self._process_statements(stmt.orelse, else_entry)
            else:
                else_exits = [cond_block]
            post = self._new_block(self.current_function)
            for be in body_exits:
                self._connect(be, post)
            for ee in else_exits:
                self._connect(ee, post)
            return [post]

        if isinstance(stmt, (ast.For, ast.AsyncFor, ast.While)):
            loop_header = self._new_block(self.current_function)
            self._connect(entry_block, loop_header)
            self._add_stmt_to_block(loop_header, stmt)
            body_entry = self._new_block(self.current_function)
            self._connect(loop_header, body_entry)
            post = self._new_block(self.current_function)
            self.loop_stack.append((post, loop_header))
            body_exits = self._process_statements(stmt.body, body_entry)
            for be in body_exits:
                self._connect(be, loop_header)
            self._connect(loop_header, post)
            self.loop_stack.pop()
            return [post]

        if isinstance(stmt, ast.Break):
            self._add_stmt_to_block(entry_block, stmt)
            if not self.loop_stack:
                return []
            break_target, _ = self.loop_stack[-1]
            self._connect(entry_block, break_target)
            return []

        if isinstance(stmt, ast.Continue):
            self._add_stmt_to_block(entry_block, stmt)
            if not self.loop_stack:
                return []
            _, continue_target = self.loop_stack[-1]
            self._connect(entry_block, continue_target)
            return []

        if isinstance(stmt, ast.With):
            self._add_stmt_to_block(entry_block, stmt)
            return [entry_block]

        if isinstance(stmt, ast.Try):
            self._add_stmt_to_block(entry_block, stmt)
            body_entry = self._new_block(self.current_function)
            self._connect(entry_block, body_entry)
            body_exits = self._process_statements(stmt.body, body_entry)
            post = self._new_block(self.current_function)
            for be in body_exits:
                self._connect(be, post)
            for handler in stmt.handlers:
                h_entry = self._new_block(self.current_function)
                self._connect(entry_block, h_entry)
                h_exits = self._process_statements(handler.body, h_entry)
                for he in h_exits:
                    self._connect(he, post)
            if stmt.orelse:
                o_entry = self._new_block(self.current_function)
                self._connect(entry_block, o_entry)
                o_exits = self._process_statements(stmt.orelse, o_entry)
                for oe in o_exits:
                    self._connect(oe, post)
            if stmt.finalbody:
                f_entry = self._new_block(self.current_function)
                self._connect(entry_block, f_entry)
                f_exits = self._process_statements(stmt.finalbody, f_entry)
                for fe in f_exits:
                    self._connect(fe, post)
            return [post]

        # default fallback: attach to current block
        self._add_stmt_to_block(entry_block, stmt)
        return [entry_block]

    def _compress_blocks(self):
        changed = True
        while changed:
            changed = False
            for n in list(self.G.nodes()):
                if n not in self.G:
                    continue
                succ = list(self.G.successors(n))
                if len(succ) != 1:
                    continue
                s = succ[0]
                if s not in self.G:
                    continue
                preds = list(self.G.predecessors(s))
                if len(preds) != 1:
                    continue
                fn_n = self.G.nodes[n].get('function', None)
                fn_s = self.G.nodes[s].get('function', None)
                if fn_n != fn_s:
                    continue
                n_stmts = self.G.nodes[n].get('stmts', [])
                s_stmts = self.G.nodes[s].get('stmts', [])
                merged_stmts = n_stmts + s_stmts
                self.G.nodes[n]['stmts'] = merged_stmts
                labels = [self._stmt_label(s_) for s_ in merged_stmts]
                self.G.nodes[n]['label'] = "|".join(labels[-5:])
                s_succ = list(self.G.successors(s))
                for ss in s_succ:
                    if ss != n:
                        self._connect(n, ss)
                if self.G.has_edge(n, s):
                    self.G.remove_edge(n, s)
                if s in self.G:
                    try:
                        self.G.remove_node(s)
                    except Exception:
                        pass
                changed = True
                break


def export_cfgs_with_graph(builder_result, global_graph):
    out = {}
    for fn, info in builder_result.items():
        out[fn] = {
            "entry": info.get("entry"),
            "exits": info.get("exits", []),
            "nodes": info.get("nodes", []),
            "global_graph": global_graph
        }
    return out


def cfg_subgraph_from_nodes(cfgs_by_name, function_name):
    """
    Reconstruct a subgraph for the given function_name using stored 'global_graph' pointer.
    Returns an empty DiGraph if networkx is not available or function not present.
    """
    try:
        import networkx as nx
    except Exception:
        raise RuntimeError("networkx required for cfg_subgraph_from_nodes")

    if not isinstance(cfgs_by_name, dict):
        return nx.DiGraph()
    info = cfgs_by_name.get(function_name)
    if not info:
        return nx.DiGraph()
    nodes = info.get("nodes", [])
    global_graph = info.get("global_graph")
    if global_graph is None:
        G = nx.DiGraph()
        for n in nodes:
            G.add_node(n, label=info.get("label", ""), function=function_name)
        return G
    sub = global_graph.subgraph(nodes).copy()
    return sub
