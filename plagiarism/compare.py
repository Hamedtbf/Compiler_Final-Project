from typing import Dict, Any, Tuple, Optional

from .tokens import tokenize_python_source, normalize_tokens, extract_protected_names_from_source, \
    token_sequence_similarity
from .ast_utils import ast_similarity
from .cfg_builder import CFGBuilder, export_cfgs_with_graph, cfg_subgraph_from_nodes
from .graph_match import compute_cfg_similarity
from .utils import read_source_file
from .parse_utils import safe_parse_module, extract_code_hierarchy


def _compute_weighted_score(scores: dict, weights: dict):
    """
    Given a dict scores that may contain keys 'token','ast','cfg' (float values) and
    weights dict with same keys, compute the weighted average using only available components.
    If no component available, return 0.0.
    """
    avail = {}
    total_w = 0.0
    for k in ("token", "ast", "cfg"):
        if k in scores and scores[k] is not None:
            w = float(weights.get(k, 0.0))
            avail[k] = (scores[k], w)
            total_w += w
    if total_w <= 0.0:
        return 0.0
    s = 0.0
    for k, (val, w) in avail.items():
        s += val * w
    return s / total_w


def _safe_token_similarity(src_a: str, src_b: str, prot_a, prot_b, cfg: dict) -> float:
    """
    Token-based similarity for two source strings; returns 1/0 for both/one empty.
    Uses tokenize_python_source + normalize_tokens + token_sequence_similarity.
    """
    if not src_a and not src_b:
        return 1.0
    if not src_a or not src_b:
        return 0.0

    ta = tokenize_python_source(src_a)
    tb = tokenize_python_source(src_b)
    union_prot = set(prot_a or set()) | set(prot_b or set())

    na = normalize_tokens(
        ta,
        protected_names=union_prot,
        ignore_literal_values=cfg.get("token", {}).get("ignore_literal_values", True),
    )
    nb = normalize_tokens(
        tb,
        protected_names=union_prot,
        ignore_literal_values=cfg.get("token", {}).get("ignore_literal_values", True),
    )
    return token_sequence_similarity(na, nb)


# --- small helpers used by compare_two_files and compare_hierarchies ---


def _build_exported_cfgs_from_source(src: str, module_name: str) -> dict:
    """
    Attempt to build CFGs from source using CFGBuilder and safe parsing.
    On any error, return empty dict.
    """
    try:
        builder = CFGBuilder()
        tree = safe_parse_module(src)
        res = builder.build_for_module(tree, module_name=module_name)
        return export_cfgs_with_graph(res, builder.G)
    except Exception:
        return {}


def _compute_cfg_similarity_safe(G1, G2, cfg_options) -> Optional[float]:
    """
    Compute CFG similarity while catching exceptions and returning None if not computable.
    """
    if G1 is None or G2 is None:
        return None
    try:
        return compute_cfg_similarity(G1, G2, cfg_options)
    except Exception:
        return None


def _get_graphs_from_exported(cfgs: dict) -> Dict[str, Any]:
    """
    Build a mapping name -> subgraph (or None if cannot be built).
    Uses cfg_subgraph_from_nodes for canonical extraction.
    """
    graphs = {}
    for name in cfgs:
        try:
            graphs[name] = cfg_subgraph_from_nodes(cfgs, name)
        except Exception:
            graphs[name] = None
    return graphs


def _find_cfg_key_for_name(cfgs: dict, entity_name: str, entity_info: Optional[dict]) -> Optional[str]:
    """
    Thin wrapper (kept name) to match the logic inside compare_hierarchies.
    See compare_hierarchies for full original behavior. This wrapper is used by other helpers.
    """
    if not cfgs:
        return None

    # 1) exact key
    if entity_name in cfgs:
        return entity_name

    # 2) last-segment match (module.ename or Class.name)
    for k in cfgs:
        if k.split('.')[-1] == entity_name:
            return k

    # 3) suffix/prefix common patterns
    for k in cfgs:
        if k.endswith('.' + entity_name) or k.startswith(entity_name + '.'):
            return k

    # 4) substring match in key
    for k in cfgs:
        if entity_name in k:
            return k

    # 5) search node attributes/labels in stored graphs for the entity name
    for k, info in cfgs.items():
        G = info.get('global_graph') or info.get('graph') or info.get('g')
        if G is None:
            node_labels = info.get('node_labels') or info.get('nodes')
            if node_labels:
                try:
                    for nl in node_labels:
                        if isinstance(nl, str) and entity_name in nl:
                            return k
                except Exception:
                    pass
            continue
        try:
            for _, data in G.nodes(data=True):
                # common attribute names that might contain entity text
                for field in ('label', 'name', 'code', 'src'):
                    v = data.get(field)
                    if isinstance(v, str) and entity_name in v:
                        return k
                # fallback: any string value containing the name
                for v in data.values():
                    if isinstance(v, str) and entity_name in v:
                        return k
        except Exception:
            pass

    # 6) line-range matching if entity_info contains lineno/end_lineno and exporter stored ranges
    if entity_info:
        start = entity_info.get('lineno') or entity_info.get('start_line') or entity_info.get('start')
        end = entity_info.get('end_lineno') or entity_info.get('end_line') or entity_info.get('end')
        if start and end:
            for k, info in cfgs.items():
                si = info.get('start_line') or info.get('lineno') or info.get('lno') or info.get('line')
                ei = info.get('end_line') or info.get('end_lineno')
                if si and ei:
                    try:
                        if (start >= si) and (end <= ei):
                            return k
                    except Exception:
                        pass

    return None


def _get_subgraph_for_cfg_key(cfgs: dict, key: Optional[str]):
    """
    Return a networkx subgraph or graph object for the function/class identified by key,
    trying cfg_subgraph_from_nodes first, then falling back to stored graph objects.
    """
    if not key:
        return None
    try:
        return cfg_subgraph_from_nodes(cfgs, key)
    except Exception:
        info = cfgs.get(key, {}) or {}
        for gname in ('global_graph', 'graph', 'g'):
            g = info.get(gname)
            if g is not None:
                return g
        # last resort: maybe info itself is a graph
        if hasattr(info, "nodes") and hasattr(info, "edges"):
            return info
        return None


# --- main compare function (refactored to use helpers) ---


def compare_two_files(path_a, path_b, config):
    # read sources
    src_a = read_source_file(path_a)
    src_b = read_source_file(path_b)

    prot_a = extract_protected_names_from_source(src_a)
    prot_b = extract_protected_names_from_source(src_b)

    # tokens (full-file)
    tokens_a = tokenize_python_source(src_a)
    tokens_b = tokenize_python_source(src_b)
    norm_a = normalize_tokens(
        tokens_a,
        protected_names=prot_a,
        ignore_literal_values=config.get("token", {}).get("ignore_literal_values", True),
    )
    norm_b = normalize_tokens(
        tokens_b,
        protected_names=prot_b,
        ignore_literal_values=config.get("token", {}).get("ignore_literal_values", True),
    )
    token_score = token_sequence_similarity(norm_a, norm_b)

    # AST (full-file)
    ast_score = ast_similarity(src_a, src_b, config, prot_a, prot_b)

    # CFG (best effort using safe parsing; skip failing blocks)
    cfg_score = 0.0
    try:
        exported_a = _build_exported_cfgs_from_source(src_a, "__module_a__")
        exported_b = _build_exported_cfgs_from_source(src_b, "__module_b__")
        cfg_score = compare_cfgs_by_functions(exported_a, exported_b, config.get("cfg", {}))
    except Exception as e:
        # if anything goes wrong, fall back gracefully (preserve original message + behavior)
        print("[cfg] building/comparing CFGs failed; falling back to AST-based approximation:", e)
        cfg_score = ast_score * 0.9

    w = config.get("weights", {})
    final = (w.get("token", 0.0) * token_score +
             w.get("ast", 0.0) * ast_score +
             w.get("cfg", 0.0) * cfg_score)

    return {
        "token": token_score,
        "ast": ast_score,
        "cfg": cfg_score,
        "final": final,
        "normalized_tokens_a": norm_a,
        "normalized_tokens_b": norm_b
    }


# --- CFG comparison by functions (kept behavior, extracted small helpers) ---


def compare_cfgs_by_functions(cfgs_a, cfgs_b, cfg_options):
    """
    cfgs_a / cfgs_b: exported dicts (from export_cfgs_with_graph), mapping function_name -> info (with 'global_graph').
    cfg_options: dict passed to compute_cfg_similarity
    """
    names_a = set(cfgs_a.keys())
    names_b = set(cfgs_b.keys())

    graphs_a = _get_graphs_from_exported(cfgs_a)
    graphs_b = _get_graphs_from_exported(cfgs_b)

    matched_pairs = []
    used_a = set()
    used_b = set()

    # match identical names
    for name in sorted(names_a.intersection(names_b)):
        G1 = graphs_a.get(name, None)
        G2 = graphs_b.get(name, None)
        sim = _compute_cfg_similarity_safe(G1, G2, cfg_options) or 0.0
        matched_pairs.append((name, name, sim, (G1.number_of_nodes() if G1 is not None else 0),
                              (G2.number_of_nodes() if G2 is not None else 0)))
        used_a.add(name)
        used_b.add(name)

    remaining_a = [n for n in names_a if n not in used_a]
    remaining_b = [n for n in names_b if n not in used_b]

    # compute pairwise sims for remaining items
    pairs = []
    for a in remaining_a:
        for b in remaining_b:
            G1 = graphs_a.get(a)
            G2 = graphs_b.get(b)
            sim = _compute_cfg_similarity_safe(G1, G2, cfg_options) or 0.0
            pairs.append((sim, a, b))
    pairs.sort(reverse=True, key=lambda x: x[0])
    for sim, a, b in pairs:
        if a in used_a or b in used_b:
            continue
        matched_pairs.append((a, b, sim, graphs_a[a].number_of_nodes(), graphs_b[b].number_of_nodes()))
        used_a.add(a)
        used_b.add(b)

    unmatched_a = [n for n in names_a if n not in used_a]
    unmatched_b = [n for n in names_b if n not in used_b]

    total_weight = 0.0
    weighted_sum = 0.0
    for a_name, b_name, sim, na, nb in matched_pairs:
        w = float(na + nb) if (na + nb) > 0 else 1.0
        weighted_sum += sim * w
        total_weight += w
    for a in unmatched_a:
        na = graphs_a[a].number_of_nodes()
        nb = 0
        w = float(na + nb) if (na + nb) > 0 else 1.0
        total_weight += w
    for b in unmatched_b:
        nb = graphs_b[b].number_of_nodes()
        na = 0
        w = float(na + nb) if (na + nb) > 0 else 1.0
        total_weight += w
    if total_weight == 0.0:
        return 1.0
    return weighted_sum / total_weight


# --- hierarchy compare (refactored for readability) ---


def compare_hierarchies(path_a, path_b, config):
    """
    Return a hierarchical comparison structure with per-function, per-class (and per-method),
    and per-variable similarity breakdowns. We attempt to compute CFG similarities when available,
    and greedily pair differently-named but similar entities.
    """
    src_a = read_source_file(path_a)
    src_b = read_source_file(path_b)

    prot_a = extract_protected_names_from_source(src_a)
    prot_b = extract_protected_names_from_source(src_b)
    union_prot = set(prot_a) | set(prot_b)

    # Build CFGs for both files (best-effort; safe_parse_module used)
    exported_a = _build_exported_cfgs_from_source(src_a, "__module_a__")
    exported_b = _build_exported_cfgs_from_source(src_b, "__module_b__")

    # Extract hierarchical elements
    h_a = extract_code_hierarchy(src_a)
    h_b = extract_code_hierarchy(src_b)

    result = {
        "file_a": path_a,
        "file_b": path_b,
        "functions": {},
        "classes": {},
        "variables": {},
    }

    cfg_opts = config.get("cfg", {})

    # Helper inline functions (kept local to preserve original structure)
    def get_subgraph_for_cfg_key_local(cfgs, key):
        return _get_subgraph_for_cfg_key(cfgs, key)

    def find_cfg_key_for_name_local(cfgs, entity_name, entity_info=None):
        return _find_cfg_key_for_name(cfgs, entity_name, entity_info)

    def get_pair_token_ast_scores(sa: str, sb: str) -> Tuple[float, float]:
        token_sim = _safe_token_similarity(sa, sb, prot_a, prot_b, config)
        ast_sim = ast_similarity(sa, sb, config, prot_a, prot_b)
        return token_sim, ast_sim

    # Generic greedy pairing by a provided score function
    def greedy_pair_entities(dict_a, dict_b, score_fn):
        """
        dict_a / dict_b: mapping name -> info-dict
        score_fn(a_name, a_info, b_name, b_info) -> float score in [0,1]
        Returns: matched_list [(a_name,b_name,score)], unmatched_a_list, unmatched_b_list
        """
        names_a = list(dict_a.keys())
        names_b = list(dict_b.keys())
        pairs = []
        for a in names_a:
            for b in names_b:
                try:
                    s = float(score_fn(a, dict_a.get(a, {}), b, dict_b.get(b, {})) or 0.0)
                except Exception:
                    s = 0.0
                pairs.append((s, a, b))
        pairs.sort(reverse=True, key=lambda x: x[0])
        used_a = set()
        used_b = set()
        matched = []
        for s, a, b in pairs:
            if a in used_a or b in used_b:
                continue
            used_a.add(a)
            used_b.add(b)
            matched.append((a, b, s))
        unmatched_a = [n for n in names_a if n not in used_a]
        unmatched_b = [n for n in names_b if n not in used_b]
        return matched, unmatched_a, unmatched_b

    # -------------------------
    # Functions: pair & compute per-pair sims
    # -------------------------
    functions_a = h_a.get("functions", {}) or {}
    functions_b = h_b.get("functions", {}) or {}

    # score closure: uses token+ast (normalize weights to only token+ast for pairing)
    w_token = config.get("weights", {}).get("token", 0.0)
    w_ast = config.get("weights", {}).get("ast", 0.0)
    denom_pair = (w_token + w_ast) if (w_token + w_ast) > 0.0 else 1.0

    def pair_score(a_name, a_info, b_name, b_info):
        sa = a_info.get("source", "")
        sb = b_info.get("source", "")
        token_sim = _safe_token_similarity(sa, sb, prot_a, prot_b, config)
        ast_sim = ast_similarity(sa, sb, config, prot_a, prot_b)
        combined = (w_token * token_sim + w_ast * ast_sim) / denom_pair
        # small boosts for name-equality or substring matches (helps short names)
        if a_name == b_name:
            combined = min(1.0, combined + 0.10)
        else:
            la = a_name.lower() if a_name else ""
            lb = b_name.lower() if b_name else ""
            if la and lb and (la in lb or lb in la):
                combined = min(1.0, combined + 0.05)
        return combined

    matched_funcs, unmatched_funcs_a, unmatched_funcs_b = greedy_pair_entities(functions_a, functions_b, pair_score)

    # fill in matched function entries (assign same pair result to both keys to keep interface)
    for a_name, b_name, _score in matched_funcs:
        a_info = functions_a.get(a_name, {})
        b_info = functions_b.get(b_name, {})
        sa = a_info.get("source", "")
        sb = b_info.get("source", "")
        token_sim = _safe_token_similarity(sa, sb, prot_a, prot_b, config)
        ast_sim = ast_similarity(sa, sb, config, prot_a, prot_b)

        # compute cfg similarity using heuristics
        cfg_sim = None
        try:
            key_a = find_cfg_key_for_name_local(exported_a, a_name, a_info)
            key_b = find_cfg_key_for_name_local(exported_b, b_name, b_info)
            G1 = get_subgraph_for_cfg_key_local(exported_a, key_a)
            G2 = get_subgraph_for_cfg_key_local(exported_b, key_b)
            cfg_sim = _compute_cfg_similarity_safe(G1, G2, cfg_opts)
        except Exception:
            cfg_sim = None

        scores = {"token": token_sim, "ast": ast_sim, "cfg": cfg_sim}
        final = _compute_weighted_score(scores, config.get("weights", {}))

        # assign same summary to both names to keep previous union-style structure
        result["functions"][a_name] = {"token": token_sim, "ast": ast_sim, "cfg": cfg_sim, "final": final}
        result["functions"][b_name] = {"token": token_sim, "ast": ast_sim, "cfg": cfg_sim, "final": final}

    # handle unmatched functions (present on only one side)
    for a in unmatched_funcs_a:
        sa = functions_a.get(a, {}).get("source", "")
        token_sim = _safe_token_similarity(sa, "", prot_a, prot_b, config)
        ast_sim = ast_similarity(sa, "", config, prot_a, prot_b)
        result["functions"][a] = {"token": token_sim, "ast": ast_sim, "cfg": None,
                                  "final": _compute_weighted_score({"token": token_sim, "ast": ast_sim, "cfg": None},
                                                                  config.get("weights", {}))}

    for b in unmatched_funcs_b:
        sb = functions_b.get(b, {}).get("source", "")
        token_sim = _safe_token_similarity("", sb, prot_a, prot_b, config)
        ast_sim = ast_similarity("", sb, config, prot_a, prot_b)
        result["functions"][b] = {"token": token_sim, "ast": ast_sim, "cfg": None,
                                  "final": _compute_weighted_score({"token": token_sim, "ast": ast_sim, "cfg": None},
                                                                  config.get("weights", {}))}

    # -------------------------
    # Classes: pair & compute per-pair sims (and methods inside classes)
    # -------------------------
    classes_a = h_a.get("classes", {}) or {}
    classes_b = h_b.get("classes", {}) or {}

    matched_classes, unmatched_classes_a, unmatched_classes_b = greedy_pair_entities(classes_a, classes_b, pair_score)

    for a_name, b_name, _score in matched_classes:
        a_info = classes_a.get(a_name, {})
        b_info = classes_b.get(b_name, {})
        sa = a_info.get("source", "")
        sb = b_info.get("source", "")

        token_sim = _safe_token_similarity(sa, sb, prot_a, prot_b, config)
        ast_sim = ast_similarity(sa, sb, config, prot_a, prot_b)

        # compute class-level cfg
        cfg_sim = None
        try:
            key_a = find_cfg_key_for_name_local(exported_a, a_name, a_info)
            key_b = find_cfg_key_for_name_local(exported_b, b_name, b_info)
            G1 = get_subgraph_for_cfg_key_local(exported_a, key_a)
            G2 = get_subgraph_for_cfg_key_local(exported_b, key_b)
            cfg_sim = _compute_cfg_similarity_safe(G1, G2, cfg_opts)
        except Exception:
            cfg_sim = None

        class_entry_a = {"token": token_sim, "ast": ast_sim, "cfg": cfg_sim,
                         "final": _compute_weighted_score({"token": token_sim, "ast": ast_sim, "cfg": cfg_sim},
                                                         config.get("weights", {})), "methods": {}}
        class_entry_b = {"token": token_sim, "ast": ast_sim, "cfg": cfg_sim,
                         "final": _compute_weighted_score({"token": token_sim, "ast": ast_sim, "cfg": cfg_sim},
                                                         config.get("weights", {})), "methods": {}}

        # Methods: pair by similarity inside the class
        methods_a = a_info.get("methods", {}) or {}
        methods_b = b_info.get("methods", {}) or {}

        matched_methods, unmatched_methods_a, unmatched_methods_b = greedy_pair_entities(methods_a, methods_b,
                                                                                         pair_score)

        # fill matched methods (and set cfg using qualified names if possible)
        for ma, mb, _s in matched_methods:
            mai = methods_a.get(ma, {})
            mbi = methods_b.get(mb, {})
            sa_m = mai.get("source", "")
            sb_m = mbi.get("source", "")
            token_sim_m = _safe_token_similarity(sa_m, sb_m, prot_a, prot_b, config)
            ast_sim_m = ast_similarity(sa_m, sb_m, config, prot_a, prot_b)

            # try to find method-level cfg keys; prefer qualified name "Class.method"
            cfg_sim_m = None
            try:
                qualified_a = f"{a_name}.{ma}"
                qualified_b = f"{b_name}.{mb}"
                key_ma = find_cfg_key_for_name_local(exported_a, qualified_a, mai) or find_cfg_key_for_name_local(
                    exported_a, ma, mai)
                key_mb = find_cfg_key_for_name_local(exported_b, qualified_b, mbi) or find_cfg_key_for_name_local(
                    exported_b, mb, mbi)
                Gma = get_subgraph_for_cfg_key_local(exported_a, key_ma)
                Gmb = get_subgraph_for_cfg_key_local(exported_b, key_mb)
                cfg_sim_m = _compute_cfg_similarity_safe(Gma, Gmb, cfg_opts)
            except Exception:
                cfg_sim_m = None

            scores_m = {"token": token_sim_m, "ast": ast_sim_m, "cfg": cfg_sim_m}
            final_m = _compute_weighted_score(scores_m, config.get("weights", {}))
            class_entry_a["methods"][ma] = {"token": token_sim_m, "ast": ast_sim_m, "cfg": cfg_sim_m, "final": final_m}
            class_entry_b["methods"][mb] = {"token": token_sim_m, "ast": ast_sim_m, "cfg": cfg_sim_m, "final": final_m}

        # unmatched methods
        for ma in unmatched_methods_a:
            mai = methods_a.get(ma, {})
            sa_m = mai.get("source", "")
            token_sim_m = _safe_token_similarity(sa_m, "", prot_a, prot_b, config)
            ast_sim_m = ast_similarity(sa_m, "", config, prot_a, prot_b)
            class_entry_a["methods"][ma] = {"token": token_sim_m, "ast": ast_sim_m, "cfg": None,
                                            "final": _compute_weighted_score({"token": token_sim_m, "ast": ast_sim_m}, {
                                                "token": config.get("weights", {}).get("token", 0.0),
                                                "ast": config.get("weights", {}).get("ast", 0.0)})}

        for mb in unmatched_methods_b:
            mbi = methods_b.get(mb, {})
            sb_m = mbi.get("source", "")
            token_sim_m = _safe_token_similarity("", sb_m, prot_a, prot_b, config)
            ast_sim_m = ast_similarity("", sb_m, config, prot_a, prot_b)
            class_entry_b["methods"][mb] = {"token": token_sim_m, "ast": ast_sim_m, "cfg": None,
                                            "final": _compute_weighted_score({"token": token_sim_m, "ast": ast_sim_m}, {
                                                "token": config.get("weights", {}).get("token", 0.0),
                                                "ast": config.get("weights", {}).get("ast", 0.0)})}

        result["classes"][a_name] = class_entry_a
        result["classes"][b_name] = class_entry_b

    # unmatched classes: populate as-is (no pair)
    for ca in unmatched_classes_a:
        cai = classes_a.get(ca, {})
        sa = cai.get("source", "")
        token_sim = _safe_token_similarity(sa, "", prot_a, prot_b, config)
        ast_sim = ast_similarity(sa, "", config, prot_a, prot_b)
        # methods: show them as unmatched
        methods_entry = {}
        for mname, minfo in (cai.get("methods", {}) or {}).items():
            sm = minfo.get("source", "")
            token_sim_m = _safe_token_similarity(sm, "", prot_a, prot_b, config)
            ast_sim_m = ast_similarity(sm, "", config, prot_a, prot_b)
            methods_entry[mname] = {"token": token_sim_m, "ast": ast_sim_m, "cfg": None,
                                    "final": _compute_weighted_score({"token": token_sim_m, "ast": ast_sim_m}, {
                                        "token": config.get("weights", {}).get("token", 0.0),
                                        "ast": config.get("weights", {}).get("ast", 0.0)})}
        result["classes"][ca] = {"token": token_sim, "ast": ast_sim, "cfg": None,
                                 "final": _compute_weighted_score({"token": token_sim, "ast": ast_sim},
                                                                  {"token": config.get("weights", {}).get("token", 0.0),
                                                                   "ast": config.get("weights", {}).get("ast", 0.0)}),
                                 "methods": methods_entry}

    for cb in unmatched_classes_b:
        cbi = classes_b.get(cb, {})
        sb = cbi.get("source", "")
        token_sim = _safe_token_similarity("", sb, prot_a, prot_b, config)
        ast_sim = ast_similarity("", sb, config, prot_a, prot_b)
        methods_entry = {}
        for mname, minfo in (cbi.get("methods", {}) or {}).items():
            sm = minfo.get("source", "")
            token_sim_m = _safe_token_similarity("", sm, prot_a, prot_b, config)
            ast_sim_m = ast_similarity("", sm, config, prot_a, prot_b)
            methods_entry[mname] = {"token": token_sim_m, "ast": ast_sim_m, "cfg": None,
                                    "final": _compute_weighted_score({"token": token_sim_m, "ast": ast_sim_m}, {
                                        "token": config.get("weights", {}).get("token", 0.0),
                                        "ast": config.get("weights", {}).get("ast", 0.0)})}
        result["classes"][cb] = {"token": token_sim, "ast": ast_sim, "cfg": None,
                                 "final": _compute_weighted_score({"token": token_sim, "ast": ast_sim},
                                                                  {"token": config.get("weights", {}).get("token", 0.0),
                                                                   "ast": config.get("weights", {}).get("ast", 0.0)}),
                                 "methods": methods_entry}

    # Variables (global) - keep the original simple union behavior
    var_union = set(h_a.get("variables", {}).keys()) | set(h_b.get("variables", {}).keys())
    for v in sorted(var_union):
        sa = h_a.get("variables", {}).get(v, {}).get("source", "")
        sb = h_b.get("variables", {}).get(v, {}).get("source", "")
        token_sim = _safe_token_similarity(sa, sb, prot_a, prot_b, config)
        ast_sim = ast_similarity(sa, sb, config, prot_a, prot_b)
        final = _compute_weighted_score({"token": token_sim, "ast": ast_sim},
                                        {"token": config.get("weights", {}).get("token", 0.0),
                                         "ast": config.get("weights", {}).get("ast", 0.0)})
        result["variables"][v] = {"token": token_sim, "ast": ast_sim, "final": final}

    return result
