from typing import Dict, Any, Tuple, Optional, List, Set

from .tokens import (
    tokenize_python_source,
    normalize_tokens,
    extract_protected_names_from_source,
    token_sequence_similarity,
)
from .ast_utils import ast_similarity
from .cfg_builder import CFGBuilder, export_cfgs_with_graph, cfg_subgraph_from_nodes
from .graph_match import compute_cfg_similarity
from .utils import read_source_file
from .parse_utils import safe_parse_module, extract_code_hierarchy


def _compute_weighted_score(scores: dict, weights: dict) -> float:
    """
    Weighted average over present components in `scores` (keys: 'token','ast','cfg').
    If no weightable component available, returns 0.0.
    """
    total_w = 0.0
    acc = 0.0
    for k in ("token", "ast", "cfg"):
        if k in scores and scores[k] is not None:
            w = float(weights.get(k, 0.0))
            if w > 0.0:
                acc += float(scores[k]) * w
                total_w += w
    return (acc / total_w) if total_w > 0.0 else 0.0


def _safe_token_similarity(src_a: str, src_b: str, prot_a, prot_b, cfg: dict) -> float:
    """
    Token similarity with guards for empty inputs. Uses tokenize -> normalize -> sequence similarity.
    Uses token config flags from cfg["token"] (ignore_literal_values, normalize_names).
    """
    if not src_a and not src_b:
        return 1.0
    if not src_a or not src_b:
        return 0.0

    ta = tokenize_python_source(src_a)
    tb = tokenize_python_source(src_b)
    union_prot = set(prot_a or set()) | set(prot_b or set())

    tok_conf = cfg.get("token", {})
    na = normalize_tokens(
        ta,
        protected_names=union_prot,
        ignore_literal_values=tok_conf.get("ignore_literal_values", True),
        normalize_names=tok_conf.get("normalize_names", False),
    )
    nb = normalize_tokens(
        tb,
        protected_names=union_prot,
        ignore_literal_values=tok_conf.get("ignore_literal_values", True),
        normalize_names=tok_conf.get("normalize_names", False),
    )
    return token_sequence_similarity(na, nb)


def _build_exported_cfgs_from_source(src: str, module_name: str) -> dict:
    """
    Try to parse and export CFGs using CFGBuilder; return {} on error.
    """
    try:
        builder = CFGBuilder()
        tree = safe_parse_module(src)
        raw = builder.build_for_module(tree, module_name=module_name)
        return export_cfgs_with_graph(raw, builder.G)
    except Exception:
        return {}


def _compute_cfg_similarity_safe(G1, G2, cfg_options) -> Optional[float]:
    """
    Compute CFG similarity, returning None if G1/G2 absent or on exception.
    """
    if G1 is None or G2 is None:
        return None
    try:
        return compute_cfg_similarity(G1, G2, cfg_options)
    except Exception:
        return None


def _get_subgraph_for_cfg_key(cfgs: dict, key: Optional[str]):
    """
    Return a subgraph object for a given exported key.
    Prefer cfg_subgraph_from_nodes; fall back to stored graph objects if present.
    """
    if not key:
        return None
    try:
        return cfg_subgraph_from_nodes(cfgs, key)
    except Exception:
        info = cfgs.get(key, {}) or {}
        for field in ("global_graph", "graph", "g"):
            g = info.get(field)
            if g is not None:
                return g
        # last-resort: maybe the info itself is a graph-like object
        if hasattr(info, "nodes") and hasattr(info, "edges"):
            return info
        return None


def _get_graphs_from_exported(cfgs: dict) -> Dict[str, Any]:
    """
    Build mapping name -> (sub)graph or None if extraction failed.
    """
    graphs: Dict[str, Any] = {}
    for name in cfgs:
        try:
            graphs[name] = cfg_subgraph_from_nodes(cfgs, name)
        except Exception:
            # fallback to stored graph objects if available
            info = cfgs.get(name, {}) or {}
            graphs[name] = info.get("global_graph") or info.get("graph") or info.get("g") or None
    return graphs


def _find_cfg_key_for_name(cfgs: dict, entity_name: str, entity_info: Optional[dict]) -> Optional[str]:
    """
    Heuristic key lookup to match an entity name to an exported cfg key.
    """
    if not cfgs or not entity_name:
        return None

    # exact
    if entity_name in cfgs:
        return entity_name

    # last segment, prefix/suffix, or substring
    for k in cfgs:
        if k.split(".")[-1] == entity_name:
            return k
    for k in cfgs:
        if k.endswith("." + entity_name) or k.startswith(entity_name + "."):
            return k
    for k in cfgs:
        if entity_name in k:
            return k

    # search node labels / stored node lists
    for k, info in cfgs.items():
        G = info.get("global_graph") or info.get("graph") or info.get("g")
        node_labels = None
        if G is None:
            node_labels = info.get("node_labels") or info.get("nodes")
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
                for fld in ("label", "name", "code", "src"):
                    v = data.get(fld)
                    if isinstance(v, str) and entity_name in v:
                        return k
                for v in data.values():
                    if isinstance(v, str) and entity_name in v:
                        return k
        except Exception:
            pass

    # line-range matching if entity_info contains lineno / end_lineno
    if entity_info:
        start = entity_info.get("lineno") or entity_info.get("start_line") or entity_info.get("start")
        end = entity_info.get("end_lineno") or entity_info.get("end_line") or entity_info.get("end")
        if start and end:
            for k, info in cfgs.items():
                si = info.get("start_line") or info.get("lineno") or info.get("lno") or info.get("line")
                ei = info.get("end_line") or info.get("end_lineno")
                if si and ei:
                    try:
                        if (start >= si) and (end <= ei):
                            return k
                    except Exception:
                        pass

    return None


def compare_cfgs_by_functions(cfgs_a, cfgs_b, cfg_options):
    """
    Compute a single similarity score between two exported-CFG sets by matching functions.
    match identical names first, then greedy best cross-matches,
    compute a weighted average where weight is proportional to node counts (fallback to 1).
    Returns a float in [0,1], with fallback to 1.0 when no nodes/weights are available.
    """
    names_a = set(cfgs_a.keys())
    names_b = set(cfgs_b.keys())

    graphs_a = _get_graphs_from_exported(cfgs_a)
    graphs_b = _get_graphs_from_exported(cfgs_b)

    matched_pairs: List[Tuple[str, str, float, int, int]] = []
    used_a = set()
    used_b = set()

    # match identical keys first
    for name in sorted(names_a.intersection(names_b)):
        G1 = graphs_a.get(name)
        G2 = graphs_b.get(name)
        sim = _compute_cfg_similarity_safe(G1, G2, cfg_options) or 0.0
        na = G1.number_of_nodes() if (G1 is not None and hasattr(G1, "number_of_nodes")) else 0
        nb = G2.number_of_nodes() if (G2 is not None and hasattr(G2, "number_of_nodes")) else 0
        matched_pairs.append((name, name, sim, na, nb))
        used_a.add(name)
        used_b.add(name)

    remaining_a = [n for n in names_a if n not in used_a]
    remaining_b = [n for n in names_b if n not in used_b]

    # compute pairwise sims and greedily match best remaining pairs
    pairs: List[Tuple[float, str, str]] = []
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
        na = graphs_a.get(a)
        nb = graphs_b.get(b)
        na_count = na.number_of_nodes() if (na is not None and hasattr(na, "number_of_nodes")) else 0
        nb_count = nb.number_of_nodes() if (nb is not None and hasattr(nb, "number_of_nodes")) else 0
        matched_pairs.append((a, b, sim, na_count, nb_count))
        used_a.add(a)
        used_b.add(b)

    unmatched_a = [n for n in names_a if n not in used_a]
    unmatched_b = [n for n in names_b if n not in used_b]

    total_weight = 0.0
    weighted_sum = 0.0
    for a_name, b_name, sim, na, nb in matched_pairs:
        w = float((na + nb) if (na + nb) > 0 else 1.0)
        weighted_sum += sim * w
        total_weight += w

    for a in unmatched_a:
        na_graph = graphs_a.get(a)
        na = na_graph.number_of_nodes() if (na_graph is not None and hasattr(na_graph, "number_of_nodes")) else 0
        w = float(na if na > 0 else 1.0)
        total_weight += w

    for b in unmatched_b:
        nb_graph = graphs_b.get(b)
        nb = nb_graph.number_of_nodes() if (nb_graph is not None and hasattr(nb_graph, "number_of_nodes")) else 0
        w = float(nb if nb > 0 else 1.0)
        total_weight += w

    if total_weight == 0.0:
        # no graphs / no nodes => treat as identical (as before)
        return 1.0
    return weighted_sum / total_weight


def compare_two_files(path_a, path_b, config):
    """
    Flat file comparison entry point (required by CLI): token/AST/CFG comparisons on entire files.
    Returns normalized token streams as part of the result.
    """
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
        normalize_names=config.get("token", {}).get("normalize_names", False),
    )
    norm_b = normalize_tokens(
        tokens_b,
        protected_names=prot_b,
        ignore_literal_values=config.get("token", {}).get("ignore_literal_values", True),
        normalize_names=config.get("token", {}).get("normalize_names", False),
    )
    token_score = token_sequence_similarity(norm_a, norm_b)

    # AST (full-file)
    ast_score = ast_similarity(src_a, src_b, config, prot_a, prot_b)

    # CFG (best-effort)
    cfg_score = 0.0
    try:
        exported_a = _build_exported_cfgs_from_source(src_a, "__module_a__")
        exported_b = _build_exported_cfgs_from_source(src_b, "__module_b__")
        cfg_score = compare_cfgs_by_functions(exported_a, exported_b, config.get("cfg", {}))
    except Exception as e:
        # graceful fallback (preserving the previous behavior)
        print("[cfg] building/comparing CFGs failed; falling back to AST-based approximation:", e)
        cfg_score = ast_score * 0.9

    # final weighted score (straight linear combination; keep original behavior)
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
        "normalized_tokens_b": norm_b,
    }


def compare_hierarchies(path_a, path_b, config):
    """
    Hierarchical comparison (functions, classes + methods, variables).
    """
    src_a = read_source_file(path_a)
    src_b = read_source_file(path_b)
    prot_a = extract_protected_names_from_source(src_a)
    prot_b = extract_protected_names_from_source(src_b)

    exported_a = _build_exported_cfgs_from_source(src_a, "__module_a__")
    exported_b = _build_exported_cfgs_from_source(src_b, "__module_b__")
    cfg_opts = config.get("cfg", {})

    h_a = extract_code_hierarchy(src_a)
    h_b = extract_code_hierarchy(src_b)

    result = {
        "file_a": path_a,
        "file_b": path_b,
        "functions": {},
        "classes": {},
        "variables": {},
    }

    weights = config.get("weights", {})

    def make_entry(
        a_name: Optional[str],
        a_info: Optional[Dict[str, Any]],
        b_name: Optional[str],
        b_info: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Compute token/AST/CFG similarity for a single entity pair and final weighted score.
        """
        src1 = (a_info or {}).get("source", "")
        src2 = (b_info or {}).get("source", "")

        tok = _safe_token_similarity(src1, src2, prot_a, prot_b, config)
        ast_s = ast_similarity(src1, src2, config, prot_a, prot_b)

        cfg_s = None
        try:
            key_a = _find_cfg_key_for_name(exported_a, a_name or "", a_info or {})
            key_b = _find_cfg_key_for_name(exported_b, b_name or "", b_info or {})
            G1 = _get_subgraph_for_cfg_key(exported_a, key_a)
            G2 = _get_subgraph_for_cfg_key(exported_b, key_b)
            cfg_s = _compute_cfg_similarity_safe(G1, G2, cfg_opts)
        except Exception:
            cfg_s = None

        final = _compute_weighted_score({"token": tok, "ast": ast_s, "cfg": cfg_s}, weights)
        return {"token": tok, "ast": ast_s, "cfg": cfg_s, "final": final}

    def greedy_pair_entities(dict_a, dict_b, score_fn):
        """
        Generic greedy matching helper used for functions, classes and methods.
        """
        names_a = list(dict_a.keys())
        names_b = list(dict_b.keys())
        pairs: List[Tuple[float, str, str]] = []
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

    # Prepare pairing score: token + ast (weights normalized to token+ast only for pairing)
    w_token = config.get("weights", {}).get("token", 0.0)
    w_ast = config.get("weights", {}).get("ast", 0.0)
    denom_pair = (w_token + w_ast) if (w_token + w_ast) > 0.0 else 1.0

    def pair_score(a_name, a_info, b_name, b_info):
        sa = (a_info or {}).get("source", "")
        sb = (b_info or {}).get("source", "")
        token_sim = _safe_token_similarity(sa, sb, prot_a, prot_b, config)
        ast_sim = ast_similarity(sa, sb, config, prot_a, prot_b)
        combined = (w_token * token_sim + w_ast * ast_sim) / denom_pair
        # small heuristic boosts for name similarity
        if a_name == b_name:
            combined = min(1.0, combined + 0.10)
        else:
            la = (a_name or "").lower()
            lb = (b_name or "").lower()
            if la and lb and (la in lb or lb in la):
                combined = min(1.0, combined + 0.05)
        return combined

    # FUNCTIONS
    functions_a = h_a.get("functions", {}) or {}
    functions_b = h_b.get("functions", {}) or {}

    matched_funcs, unmatched_funcs_a, unmatched_funcs_b = greedy_pair_entities(functions_a, functions_b, pair_score)

    for a_name, b_name, _ in matched_funcs:
        a_info = functions_a.get(a_name, {})
        b_info = functions_b.get(b_name, {})
        entry = make_entry(a_name, a_info, b_name, b_info)
        # assign same summary to both names (union-style interface preserved)
        result["functions"][a_name] = entry
        result["functions"][b_name] = entry

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

    # CLASSES + METHODS
    classes_a = h_a.get("classes", {}) or {}
    classes_b = h_b.get("classes", {}) or {}

    matched_classes, unmatched_classes_a, unmatched_classes_b = greedy_pair_entities(classes_a, classes_b, pair_score)

    for a_name, b_name, _ in matched_classes:
        a_info = classes_a.get(a_name, {})
        b_info = classes_b.get(b_name, {})
        entry = make_entry(a_name, a_info, b_name, b_info)
        # prepare method containers
        entry_a = dict(entry)
        entry_b = dict(entry)
        entry_a["methods"] = {}
        entry_b["methods"] = {}

        # pair methods inside the class
        methods_a = a_info.get("methods", {}) or {}
        methods_b = b_info.get("methods", {}) or {}
        m_matched, mu_a, mu_b = greedy_pair_entities(methods_a, methods_b, pair_score)

        for ma, mb, _ in m_matched:
            mai = methods_a.get(ma, {})
            mbi = methods_b.get(mb, {})
            me = make_entry(ma, mai, mb, mbi)
            entry_a["methods"][ma] = me
            entry_b["methods"][mb] = me

        for ma in mu_a:
            mai = methods_a.get(ma, {})
            sm = mai.get("source", "")
            token_sim_m = _safe_token_similarity(sm, "", prot_a, prot_b, config)
            ast_sim_m = ast_similarity(sm, "", config, prot_a, prot_b)
            entry_a["methods"][ma] = {"token": token_sim_m, "ast": ast_sim_m, "cfg": None,
                                      "final": _compute_weighted_score({"token": token_sim_m, "ast": ast_sim_m},
                                                                      config.get("weights", {}))}

        for mb in mu_b:
            mbi = methods_b.get(mb, {})
            sm = mbi.get("source", "")
            token_sim_m = _safe_token_similarity("", sm, prot_a, prot_b, config)
            ast_sim_m = ast_similarity("", sm, config, prot_a, prot_b)
            entry_b["methods"][mb] = {"token": token_sim_m, "ast": ast_sim_m, "cfg": None,
                                      "final": _compute_weighted_score({"token": token_sim_m, "ast": ast_sim_m},
                                                                      config.get("weights", {}))}

        result["classes"][a_name] = entry_a
        result["classes"][b_name] = entry_b

    # unmatched classes (A)
    for ca in unmatched_classes_a:
        cai = classes_a.get(ca, {})
        sa = cai.get("source", "")
        token_sim = _safe_token_similarity(sa, "", prot_a, prot_b, config)
        ast_sim = ast_similarity(sa, "", config, prot_a, prot_b)
        methods_entry = {}
        for mname, minfo in (cai.get("methods", {}) or {}).items():
            sm = minfo.get("source", "")
            token_sim_m = _safe_token_similarity(sm, "", prot_a, prot_b, config)
            ast_sim_m = ast_similarity(sm, "", config, prot_a, prot_b)
            methods_entry[mname] = {"token": token_sim_m, "ast": ast_sim_m, "cfg": None,
                                    "final": _compute_weighted_score({"token": token_sim_m, "ast": ast_sim_m},
                                                                    config.get("weights", {}))}
        result["classes"][ca] = {"token": token_sim, "ast": ast_sim, "cfg": None,
                                 "final": _compute_weighted_score({"token": token_sim, "ast": ast_sim},
                                                                 config.get("weights", {})),
                                 "methods": methods_entry}

    # unmatched classes (B)
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
                                    "final": _compute_weighted_score({"token": token_sim_m, "ast": ast_sim_m},
                                                                    config.get("weights", {}))}
        result["classes"][cb] = {"token": token_sim, "ast": ast_sim, "cfg": None,
                                 "final": _compute_weighted_score({"token": token_sim, "ast": ast_sim},
                                                                 config.get("weights", {})),
                                 "methods": methods_entry}

    # VARIABLES (union)
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
