from .tokens import tokenize_python_source, normalize_tokens, extract_protected_names_from_source, token_sequence_similarity
from .ast_utils import ast_similarity
from .cfg_builder import CFGBuilder, export_cfgs_with_graph, cfg_subgraph_from_nodes
from .graph_match import compute_cfg_similarity
from .utils import read_source_file

def compare_cfgs_by_functions(cfgs_a, cfgs_b, cfg_options):
    """
    cfgs_a / cfgs_b: exported dicts (from export_cfgs_with_graph), mapping function_name -> info (with 'global_graph').
    cfg_options: dict passed to compute_cfg_similarity
    """
    names_a = set(cfgs_a.keys())
    names_b = set(cfgs_b.keys())

    graphs_a = {}
    graphs_b = {}

    for name in cfgs_a:
        graphs_a[name] = cfg_subgraph_from_nodes(cfgs_a, name)
    for name in cfgs_b:
        graphs_b[name] = cfg_subgraph_from_nodes(cfgs_b, name)

    matched_pairs = []
    used_a = set()
    used_b = set()

    # match identical names
    for name in sorted(names_a.intersection(names_b)):
        G1 = graphs_a.get(name, None) or None
        G2 = graphs_b.get(name, None) or None
        if G1 is None or G2 is None:
            sim = 0.0
        else:
            sim = compute_cfg_similarity(G1, G2, cfg_options)
        matched_pairs.append((name, name, sim, (G1.number_of_nodes() if G1 is not None else 0), (G2.number_of_nodes() if G2 is not None else 0)))
        used_a.add(name)
        used_b.add(name)

    remaining_a = [n for n in names_a if n not in used_a]
    remaining_b = [n for n in names_b if n not in used_b]

    # compute pairwise sims
    pairs = []
    for a in remaining_a:
        for b in remaining_b:
            G1 = graphs_a.get(a)
            G2 = graphs_b.get(b)
            if (G1 is None) or (G2 is None):
                sim = 0.0
            else:
                sim = compute_cfg_similarity(G1, G2, cfg_options)
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


def compare_two_files(path_a, path_b, config):
    # read sources
    src_a = read_source_file(path_a)
    src_b = read_source_file(path_b)

    prot_a = extract_protected_names_from_source(src_a)
    prot_b = extract_protected_names_from_source(src_b)

    # tokens
    tokens_a = tokenize_python_source(src_a)
    tokens_b = tokenize_python_source(src_b)
    norm_a = normalize_tokens(tokens_a, protected_names=prot_a,
                              ignore_literal_values=config.get("token", {}).get("ignore_literal_values", True))
    norm_b = normalize_tokens(tokens_b, protected_names=prot_b,
                              ignore_literal_values=config.get("token", {}).get("ignore_literal_values", True))
    token_score = token_sequence_similarity(norm_a, norm_b)

    # AST
    ast_score = ast_similarity(src_a, src_b, config, prot_a, prot_b)

    # CFG
    cfg_score = 0.0
    try:
        # build CFGs
        builder_a = CFGBuilder()
        builder_b = CFGBuilder()
        tree_a = __import__("ast").parse(src_a)
        tree_b = __import__("ast").parse(src_b)
        res_a = builder_a.build_for_module(tree_a, module_name="__module_a__")
        res_b = builder_b.build_for_module(tree_b, module_name="__module_b__")
        exported_a = export_cfgs_with_graph(res_a, builder_a.G)
        exported_b = export_cfgs_with_graph(res_b, builder_b.G)
        cfg_score = compare_cfgs_by_functions(exported_a, exported_b, config.get("cfg", {}))
    except Exception as e:
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
