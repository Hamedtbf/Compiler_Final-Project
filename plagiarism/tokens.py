import io
import tokenize
import keyword
import ast
import difflib
import re


def extract_protected_names_from_source(src):
    """
    Function and class names, also imported names are considered protected by default.
    Try AST extraction; if parsing fails, use regexp heuristics to extract function/class/import names.
    """
    protected = set()
    try:
        tree = ast.parse(src)
    except Exception:

        # fallback heuristics using regex
        for m in re.finditer(r'^\s*def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(', src, flags=re.MULTILINE):
            protected.add(m.group(1))
        for m in re.finditer(r'^\s*class\s+([A-Za-z_][A-Za-z0-9_]*)\s*(\(|:)', src, flags=re.MULTILINE):
            protected.add(m.group(1))
        for m in re.finditer(r'^\s*import\s+(.*)$', src, flags=re.MULTILINE):
            items = m.group(1).split(',')

            for item in items:

                item = item.strip()
                if not item:
                    continue
                if ' as ' in item:
                    alias = item.split(' as ')[1].strip()
                    if alias:
                        protected.add(alias)
                else:
                    top = item.split('.')[0].strip()
                    if top:
                        protected.add(top)
        for m in re.finditer(r'^\s*from\s+([^\s]+)\s+import\s+(.*)$', src, flags=re.MULTILINE):
            imported = m.group(2).strip()
            imported = imported.strip('() ')
            parts = [p.strip() for p in imported.split(',') if p.strip()]
            for p in parts:
                if ' as ' in p:
                    alias = p.split(' as ')[1].strip()
                    if alias:
                        protected.add(alias)
                else:
                    name = p.split('.')[0].strip()
                    if name:
                        protected.add(name)
        return protected

    for node in ast.walk(tree):

        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            protected.add(node.name)

        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname:
                    protected.add(alias.asname)
                else:
                    protected.add(alias.name.split('.')[0])

        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.asname:
                    protected.add(alias.asname)
                else:
                    protected.add(alias.name)

    return protected


def tokenize_python_source(src):
    """
    Return list of tuples (toknum, tokname, tokval)
    """
    tokens = []
    try:
        gen = tokenize.generate_tokens(io.StringIO(src).readline)
        for toknum, tokval, start, end, line in gen:

            tokname = tokenize.tok_name.get(toknum, str(toknum))
            tokens.append((toknum, tokname, tokval))

    except Exception as e:
        print(f"[tokenize] warning: tokenize failed: {e}")
    return tokens


def normalize_tokens(tokens, protected_names=None, ignore_literal_values=True):
    if protected_names is None:
        protected_names = set()

    name_map = {}
    name_counter = 0
    normalized = []

    for toknum, tokname, tokval in tokens:

        if tokname == "NAME":
            if tokval in protected_names or keyword.iskeyword(tokval) or tokval in ("True", "False", "None"):
                normalized.append(tokval)
            else:
                if tokval not in name_map:
                    name_counter += 1
                    name_map[tokval] = f"VAR_{name_counter}"
                normalized.append(name_map[tokval])

        elif tokname == "NUMBER":
            normalized.append("NUMBER" if ignore_literal_values else tokval)
        elif tokname == "STRING":
            normalized.append("STRING" if ignore_literal_values else tokval)
        elif tokname in ("NEWLINE", "NL", "INDENT", "DEDENT", "COMMENT"):
            continue
        else:
            normalized.append(tokval)

    return normalized


def token_sequence_similarity(seq1, seq2):
    sm = difflib.SequenceMatcher(None, seq1, seq2)
    return sm.ratio()
