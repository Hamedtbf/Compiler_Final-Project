import time
from collections import Counter

try:
    import networkx as nx
    NX_AVAILABLE = True
except Exception:
    NX_AVAILABLE = False

# safe label creation for node/edge attribute dicts
def _safe_label_from_attrs(attrs, max_len=240):
    if not attrs:
        return ""
    parts = []
    for k, v in sorted(attrs.items()):
        if isinstance(v, (str, int, float, bool)) or v is None:
            parts.append(f"{k}={v}")
        elif isinstance(v, (list, tuple, set)):
            parts.append(f"{k}=[{','.join(map(str, v))}]")
        elif isinstance(v, dict):
            inner = ",".join(f"{ik}:{str(iv)}" for ik, iv in sorted(v.items()))
            parts.append(f"{k}={{ {inner} }}")
        else:
            parts.append(f"{k}={type(v).__name__}")
    label = "|".join(parts)
    if len(label) > max_len:
        label = label[: max_len - 3] + "..."
    return label

def _make_ged_friendly(G):
    """
    Build an int-keyed DiGraph/Graph with node and edge attribute 'label' as plain strings.
    """
    if not NX_AVAILABLE:
        raise RuntimeError("networkx required")
    H = nx.DiGraph() if isinstance(G, nx.DiGraph) else nx.Graph()
    mapping = {}
    for i, n in enumerate(G.nodes()):
        attrs = dict(G.nodes[n]) if G.nodes[n] else {}
        # prefer an existing 'label' attribute (string), else create safe label
        label = attrs.get("label", None)
        if label is None:
            label = _safe_label_from_attrs(attrs)
        H.add_node(i, label=str(label))
        mapping[n] = i
    for u, v, ed in G.edges(data=True):
        if u not in mapping or v not in mapping:
            continue
        edge_label = _safe_label_from_attrs(dict(ed) if ed else {})
        H.add_edge(mapping[u], mapping[v], label=str(edge_label))
    return H

def _greedy_cfg_similarity(G1, G2):
    """
    Greedy similarity working on canonical graphs (nodes carry 'label' strings).
    """
    # Node label counts
    lab1 = Counter(G1.nodes[n].get('label', '') for n in G1.nodes())
    lab2 = Counter(G2.nodes[n].get('label', '') for n in G2.nodes())
    node_matches = sum(min(lab1[k], lab2.get(k, 0)) for k in lab1.keys())
    node_denom = max(1, max(G1.number_of_nodes(), G2.number_of_nodes()))
    node_sim = node_matches / node_denom

    def _edge_sig(G, u, v):
        lu = G.nodes[u].get('label', '')
        lv = G.nodes[v].get('label', '')
        el = G.edges[u, v].get('label', '')
        return (lu, el, lv)

    e1 = Counter(_edge_sig(G1, u, v) for u, v in G1.edges())
    e2 = Counter(_edge_sig(G2, u, v) for u, v in G2.edges())
    edge_matches = sum(min(e1[k], e2.get(k, 0)) for k in e1.keys())
    edge_denom = max(1, max(G1.number_of_edges(), G2.number_of_edges()))
    edge_sim = edge_matches / edge_denom

    return 0.5 * (node_sim + edge_sim)


def compute_cfg_similarity(G1, G2, cfg_options=None):
    """
    Compute CFG similarity. cfg_options is a dict (or None) with keys:
      - ged_timeout (float)
      - ged_max_nodes (int)
      - greedy_fallback (bool)
    If networkx not available, raises RuntimeError.
    """
    if not NX_AVAILABLE:
        raise RuntimeError("networkx required for compute_cfg_similarity")
    cfg_options = cfg_options or {}
    try:
        ged_timeout = float(cfg_options.get("ged_timeout", 5.0))
    except Exception:
        ged_timeout = 5.0
    try:
        ged_max_nodes = int(cfg_options.get("ged_max_nodes", 60))
    except Exception:
        ged_max_nodes = 60

    H1 = _make_ged_friendly(G1)
    H2 = _make_ged_friendly(G2)

    # If graphs are too big, skip GED
    if max(H1.number_of_nodes(), H2.number_of_nodes()) > ged_max_nodes:
        return _greedy_cfg_similarity(H1, H2)

    node_match = lambda a, b: a.get("label", "") == b.get("label", "")
    edge_match = lambda a, b: a.get("label", "") == b.get("label", "")

    # try optimize_graph_edit_distance (generator)
    try:
        ged_iter = nx.optimize_graph_edit_distance(H1, H2, node_match=node_match, edge_match=edge_match)
        start = time.time()
        best = float("inf")
        for d in ged_iter:
            best = min(best, d)
            if time.time() - start > ged_timeout:
                raise TimeoutError("GED timed out")
        denom = max(1, (max(H1.number_of_nodes(), H2.number_of_nodes()) + max(H1.number_of_edges(), H2.number_of_edges())))
        sim = 1.0 - (best / denom)
        return max(0.0, min(1.0, sim))
    except Exception as e:
        # try older graph_edit_distance with timeout parameter
        try:
            ged = nx.graph_edit_distance(H1, H2, node_match=node_match, edge_match=edge_match, timeout=ged_timeout)
            if ged is None:
                raise TimeoutError("GED returned None (likely timed out)")
            best = float(ged)
            denom = max(1, (max(H1.number_of_nodes(), H2.number_of_nodes()) + max(H1.number_of_edges(), H2.number_of_edges())))
            sim = 1.0 - (best / denom)
            return max(0.0, min(1.0, sim))
        except Exception as e2:
            print("[cfg] graph_edit_distance failed or timed out:", e2, "; falling back to greedy matching.")
            return _greedy_cfg_similarity(H1, H2)
