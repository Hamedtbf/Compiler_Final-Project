# plagiarism_pipeline.py
# Requires: antlr4-python3-runtime, zss, networkx (optional), editdistance (optional)
# Place this file in the same directory as your generated Python3Lexer.py / Python3Parser.py

import os
import argparse
import csv
import difflib
import ast as py_ast

from antlr4 import FileStream, CommonTokenStream, ParserRuleContext, TerminalNode

# imports from your generated ANTLR files (must be in PYTHONPATH / same folder)
from Python3Lexer import Python3Lexer
from Python3Parser import Python3Parser

# optional packages
try:
    import editdistance
except Exception:
    editdistance = None

from zss import Node as ZSSNode, simple_distance

# ANTLR token EOF value (usually -1)
EOF = -1

# ---------- Token extraction and normalization ----------

# keep the EOF constant already defined in your file (e.g., EOF = -1)

def tokenize_file_with_antlr(path):
    """
    Return (tokens_list, lexer_instance).
    Compatible with various antlr4 runtime versions:
    - tries token_stream.getTokens()
    - falls back to token_stream.tokens
    - final fallback: consume tokens via lexer.nextToken()
    """
    stream = FileStream(path, encoding='utf-8')
    lexer = Python3Lexer(stream)
    token_stream = CommonTokenStream(lexer)
    token_stream.fill()

    # try CommonTokenStream.getTokens() (some runtime versions require args)
    tokens_all = None
    try:
        tokens_all = token_stream.getTokens()  # may raise TypeError if signature differs
    except TypeError:
        # try attribute 'tokens' (common)
        tokens_all = getattr(token_stream, 'tokens', None)
    except Exception:
        tokens_all = None

    if tokens_all is None:
        # Last-resort: consume tokens directly from the lexer
        tokens_all = []
        tok = lexer.nextToken()
        while tok is not None and tok.type != EOF:
            tokens_all.append(tok)
            tok = lexer.nextToken()

    tokens = [t for t in tokens_all if t.type != EOF]
    return tokens, lexer


def normalize_token_sequence(tokens, lexer):
    """
    Map tokens to a normalized token sequence.
    - Replace identifier token (NAME) with IDn mapping (preserving first-appearance mapping).
    - Replace numbers and strings with placeholders NUM, STR.
    - Drop purely formatting tokens (INDENT/DEDENT/NEWLINE/NL) to reduce noise.
    """
    sym = getattr(lexer, 'symbolicNames', None)
    seq = []
    id_map = {}
    next_id = 1
    for t in tokens:
        ttype = t.type
        tok_name = None
        if sym and ttype < len(sym):
            tok_name = sym[ttype]
        if not tok_name:
            tok_name = t.text

        # Normalization rules (adjust to your grammar token names if different)
        if tok_name == 'NAME':
            if t.text not in id_map:
                id_map[t.text] = f'ID{next_id}'
                next_id += 1
            seq.append(id_map[t.text])
        elif tok_name in ('NUMBER', 'INTEGER', 'DECIMAL_INTEGER', 'FLOAT_NUMBER'):
            seq.append('NUM')
        elif tok_name and 'STRING' in tok_name:
            seq.append('STR')
        elif tok_name in ('INDENT', 'DEDENT', 'NEWLINE', 'NL'):
            # skip formatting tokens
            continue
        else:
            seq.append(tok_name if isinstance(tok_name, str) else str(tok_name))
    return seq

def token_sequence_similarity(seq1, seq2):
    """Return similarity in [0,1]. Prefer editdistance if installed, else difflib ratio."""
    if len(seq1) == 0 and len(seq2) == 0:
        return 1.0
    if editdistance:
        # editdistance.eval works on lists (if available)
        d = editdistance.eval(seq1, seq2)
        sim = 1.0 - d / max(len(seq1), len(seq2))
        return max(0.0, min(1.0, sim))
    else:
        # difflib works on sequences and returns ratio in [0,1]
        return difflib.SequenceMatcher(None, seq1, seq2).ratio()

# ---------- AST extraction (from ANTLR parse tree) and tree edit distance ----------

class SimpleASTNode:
    def __init__(self, label, children=None):
        self.label = label
        self.children = children or []

def parse_file_to_parse_tree(path, start_rule='file_input'):
    """Parse using ANTLR generated parser. If the start_rule doesn't exist, try common Python top rules."""
    stream = FileStream(path, encoding='utf-8')
    lexer = Python3Lexer(stream)
    tokens = CommonTokenStream(lexer)
    parser = Python3Parser(tokens)
    # choose a top rule by name; common Python grammars use 'file_input' or 'module'
    if not hasattr(parser, start_rule):
        for cand in ('file_input', 'module', 'file_input_stmt', 'start'):
            if hasattr(parser, cand):
                start_rule = cand
                break
    tree = getattr(parser, start_rule)()
    return tree

def parse_tree_to_simple_ast(node, lexer_cls=Python3Lexer, parser_cls=Python3Parser):
    """
    Convert ANTLR parse tree into a simplified labelled tree:
    - rule nodes use parser rule names
    - terminal nodes use normalized token names (ID/NUM/STR or token symbolic name)
    """
    # TerminalNode check: using antlr4.TerminalNode
    if isinstance(node, TerminalNode):
        sym = getattr(lexer_cls, 'symbolicNames', None)
        symbol = node.getSymbol()
        if symbol is None:
            label = node.getText()
        else:
            ttype = symbol.type
            label = None
            if sym and ttype < len(sym):
                label = sym[ttype]
            if not label:
                label = symbol.text or node.getText()
        # normalize
        if label == 'NAME':
            return SimpleASTNode('ID')
        if label and 'STRING' in label:
            return SimpleASTNode('STR')
        if label and label in ('NUMBER', 'INTEGER'):
            return SimpleASTNode('NUM')
        return SimpleASTNode(label)
    # Parser rule node
    if isinstance(node, ParserRuleContext):
        rindex = node.getRuleIndex()
        rule_name = parser_cls.ruleNames[rindex] if (parser_cls and rindex < len(parser_cls.ruleNames)) else f'R{rindex}'
        s = SimpleASTNode(rule_name)
        for i in range(node.getChildCount()):
            child = node.getChild(i)
            child_s = parse_tree_to_simple_ast(child, lexer_cls, parser_cls)
            if child_s is not None:
                s.children.append(child_s)
        return s
    # fallback (shouldn't normally happen)
    return None

def simple_ast_to_zss(node):
    """Convert our SimpleASTNode to a zss.Node for tree edit distance."""
    z = ZSSNode(node.label)
    for c in node.children:
        z.addkid(simple_ast_to_zss(c))
    return z

def count_simple_ast_nodes(node):
    return 1 + sum(count_simple_ast_nodes(c) for c in node.children)

def ast_tree_similarity(ast1, ast2):
    """Compute normalized similarity in [0,1] using tree edit distance (zss)."""
    if ast1 is None or ast2 is None:
        return 0.0
    z1 = simple_ast_to_zss(ast1)
    z2 = simple_ast_to_zss(ast2)
    dist = simple_distance(z1, z2)  # default costs: insert=1, delete=1, substitute=0/1
    n1 = count_simple_ast_nodes(ast1)
    n2 = count_simple_ast_nodes(ast2)
    denom = max(n1, n2)
    if denom == 0:
        return 1.0
    sim = 1.0 - dist / denom
    return max(0.0, min(1.0, sim))

# ---------- Lightweight "CFG" / flow extraction using Python builtin ast (practical fallback) ----------

def code_to_flow_sequence(source_code):
    """
    Create a flattened sequence of control-flow-relevant tokens extracted using Python's builtin ast module.
    This is simpler than a full CFG but captures branching/loop structure for similarity.
    """
    try:
        tree = py_ast.parse(source_code)
    except Exception:
        return []

    seq = []

    def visit(node):
        # classify important statement types
        if isinstance(node, py_ast.FunctionDef):
            seq.append('FUNCDEF')
            for s in node.body:
                visit(s)
        elif isinstance(node, py_ast.Assign):
            seq.append('ASSIGN')
        elif isinstance(node, py_ast.AugAssign):
            seq.append('ASSIGN')
        elif isinstance(node, py_ast.Return):
            seq.append('RETURN')
        elif isinstance(node, py_ast.For):
            seq.append('FOR')
            for s in node.body: visit(s)
            if node.orelse:
                seq.append('FOR_ELSE')
                for s in node.orelse: visit(s)
        elif isinstance(node, py_ast.While):
            seq.append('WHILE')
            for s in node.body: visit(s)
            if node.orelse:
                seq.append('WHILE_ELSE')
                for s in node.orelse: visit(s)
        elif isinstance(node, py_ast.If):
            seq.append('IF')
            for s in node.body: visit(s)
            if node.orelse:
                seq.append('ELSE')
                for s in node.orelse: visit(s)
        elif isinstance(node, py_ast.Expr):
            if isinstance(node.value, py_ast.Call):
                seq.append('CALL')
            else:
                seq.append('EXPR')
        elif isinstance(node, (py_ast.Try,)):
            seq.append('TRY')
            for s in getattr(node, 'body', []): visit(s)
            for h in getattr(node, 'handlers', []): visit(h)
            for s in getattr(node, 'finalbody', []): visit(s)
        else:
            # recursively visit children
            for c in py_ast.iter_child_nodes(node):
                visit(c)

    visit(tree)
    return seq

def sequence_similarity(seq1, seq2):
    return difflib.SequenceMatcher(None, seq1, seq2).ratio()

# ---------- Pipeline compare function ----------

def compare_pair(file1, file2, thresholds=(0.6, 0.6), weights=(0.3, 0.5, 0.2)):
    """
    Compare two files:
    - token similarity always computed
    - AST similarity computed if token similarity >= thresholds[0]
    - flow similarity computed if AST similarity >= thresholds[1]
    Return dict with token/ast/flow/final scores.
    """
    tok_threshold, ast_threshold = thresholds
    w_t, w_a, w_c = weights

    toks1, lexer1 = tokenize_file_with_antlr(file1)
    toks2, lexer2 = tokenize_file_with_antlr(file2)
    seq1 = normalize_token_sequence(toks1, lexer1)
    seq2 = normalize_token_sequence(toks2, lexer2)
    tok_sim = token_sequence_similarity(seq1, seq2)

    ast_sim = 0.0
    if tok_sim >= tok_threshold:
        # parse trees
        tree1 = parse_file_to_parse_tree(file1)
        tree2 = parse_file_to_parse_tree(file2)
        simple1 = parse_tree_to_simple_ast(tree1)
        simple2 = parse_tree_to_simple_ast(tree2)
        ast_sim = ast_tree_similarity(simple1, simple2)

    flow_sim = 0.0
    if ast_sim >= ast_threshold:
        with open(file1, 'r', encoding='utf-8', errors='ignore') as f:
            code1 = f.read()
        with open(file2, 'r', encoding='utf-8', errors='ignore') as f:
            code2 = f.read()
        fseq1 = code_to_flow_sequence(code1)
        fseq2 = code_to_flow_sequence(code2)
        flow_sim = sequence_similarity(fseq1, fseq2)

    final_score = w_t * tok_sim + w_a * ast_sim + w_c * flow_sim
    return {'token': tok_sim, 'ast': ast_sim, 'flow': flow_sim, 'final': final_score}

# ---------- Batch / CLI ----------

def batch_compare_folder(folder, out_csv, thresholds=(0.6, 0.6), weights=(0.3, 0.5, 0.2), extensions=('.py',)):
    files = sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(extensions)])
    results = []
    n = len(files)
    for i in range(n):
        for j in range(i+1, n):
            f1 = files[i]; f2 = files[j]
            res = compare_pair(f1, f2, thresholds, weights)
            results.append({
                'file1': os.path.basename(f1), 'file2': os.path.basename(f2),
                'token': round(res['token'],4), 'ast': round(res['ast'],4),
                'flow': round(res['flow'],4), 'final': round(res['final'],4)
            })
            print(f"Compared {os.path.basename(f1)} <> {os.path.basename(f2)}: final={res['final']:.4f}")
    # write CSV
    keys = ['file1','file2','token','ast','flow','final']
    with open(out_csv, 'w', newline='', encoding='utf-8') as csvf:
        writer = csv.DictWriter(csvf, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved results to {out_csv}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True, help='Folder with source files (.py)')
    ap.add_argument('--out', required=True, help='Output CSV file')
    ap.add_argument('--tok-th', type=float, default=0.6, help='Token-stage threshold to trigger AST')
    ap.add_argument('--ast-th', type=float, default=0.6, help='AST-stage threshold to trigger flow')
    ap.add_argument('--wt', type=float, default=0.3, help='weight token (default 0.3)')
    ap.add_argument('--wa', type=float, default=0.5, help='weight AST (default 0.5)')
    ap.add_argument('--wc', type=float, default=0.2, help='weight flow/cfg (default 0.2)')
    args = ap.parse_args()
    weights = (args.wt, args.wa, args.wc)
    batch_compare_folder(args.dir, args.out, thresholds=(args.tok_th, args.ast_th), weights=weights)

if __name__ == '__main__':
    main()
