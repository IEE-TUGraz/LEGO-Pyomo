import pyomo.environ as pyo

try:
    import networkx as nx
except ImportError:
    nx = None

from InOutModule.printer import Printer

printer = Printer.getInstance()


def _unordered_pair_key(i, j):
    return tuple(sorted((str(i), str(j))))


def get_first_circuit_for_pair(model: pyo.ConcreteModel, i, j):
    if not hasattr(model, "_first_circuit_by_pair"):
        first_circuit_by_pair = {}
        for edge in model.la:
            edge_i, edge_j, circuit = edge
            first_circuit_by_pair.setdefault(
                _unordered_pair_key(edge_i, edge_j),
                (edge_i, edge_j, circuit),
            )
        model._first_circuit_by_pair = first_circuit_by_pair

    return model._first_circuit_by_pair.get(_unordered_pair_key(i, j))


def _build_cycle_basis_without_networkx(model: pyo.ConcreteModel) -> list[list[str]]:
    adjacency = {node: set() for node in model.i}
    for i, j, _ in model.la:
        adjacency[i].add(j)
        adjacency[j].add(i)

    parent = {}
    depth = {}
    visited = set()
    tree_edges = set()
    back_edges = []

    for root in model.i:
        if root in visited:
            continue

        parent[root] = None
        depth[root] = 0
        stack = [root]
        visited.add(root)

        while stack:
            node = stack.pop()
            for neighbor in adjacency[node]:
                edge_key = frozenset((node, neighbor))
                if neighbor not in visited:
                    visited.add(neighbor)
                    parent[neighbor] = node
                    depth[neighbor] = depth[node] + 1
                    tree_edges.add(edge_key)
                    stack.append(neighbor)
                elif (
                    parent[node] != neighbor
                    and edge_key not in tree_edges
                    and edge_key not in {frozenset(edge) for edge in back_edges}
                ):
                    back_edges.append((node, neighbor))

    def build_path_to_root(node):
        path = []
        while node is not None:
            path.append(node)
            node = parent[node]
        return path

    cycles = []
    seen = set()
    for start, end in back_edges:
        path_start = build_path_to_root(start)
        path_end = build_path_to_root(end)
        common = next((node for node in path_start if node in set(path_end)), None)
        if common is None:
            continue

        start_to_common = path_start[:path_start.index(common) + 1]
        end_to_common = path_end[:path_end.index(common) + 1]
        cycle = start_to_common + list(reversed(end_to_common[:-1]))
        key = tuple(sorted(str(node) for node in cycle))
        if len(cycle) >= 3 and key not in seen:
            seen.add(key)
            cycles.append(cycle)

    return cycles


def build_directed_cycle_edges(model: pyo.ConcreteModel, cycle: list[str]):
    directed_edges = []
    for index, start in enumerate(cycle):
        end = cycle[(index + 1) % len(cycle)]
        stored_edge = get_first_circuit_for_pair(model, start, end)
        if stored_edge is None:
            printer.warning(
                f"Skipping invalid cycle edge between {start} and {end}: no matching line found"
            )
            return None

        edge_i, edge_j, circuit = stored_edge
        if edge_i == start and edge_j == end:
            sign = 1
        elif edge_i == end and edge_j == start:
            sign = -1
        else:
            printer.warning(
                f"Skipping invalid cycle edge between {start} and {end}: orientation mismatch"
            )
            return None

        directed_edges.append((sign, edge_i, edge_j, circuit))

    return directed_edges


def detect_cycle_basis(model: pyo.ConcreteModel) -> list[list[tuple[int, str, str, str]]]:
    unique_pairs = set()
    if nx is not None:
        graph = nx.Graph()
        graph.add_nodes_from(model.i)
        for i, j, _ in model.la:
            pair_key = _unordered_pair_key(i, j)
            if pair_key in unique_pairs:
                continue
            unique_pairs.add(pair_key)
            graph.add_edge(i, j)
        raw_cycles = nx.cycle_basis(graph)
    else:
        printer.warning("networkx not available; using fallback cycle detection for BFM cycle basis")
        raw_cycles = _build_cycle_basis_without_networkx(model)

    detected_cycles = []
    seen_cycle_keys = set()
    if not raw_cycles:
        printer.information("No meshed network cycles detected for SOCP BFM")
        return detected_cycles

    for cycle in raw_cycles:
        directed_edges = build_directed_cycle_edges(model, cycle)
        if not directed_edges:
            continue

        cycle_key = tuple(
            sorted((min(str(i), str(j)), max(str(i), str(j)), str(c)) for _, i, j, c in directed_edges)
        )
        if cycle_key in seen_cycle_keys:
            continue
        seen_cycle_keys.add(cycle_key)
        detected_cycles.append(directed_edges)

        cycle_path = " -> ".join([*(str(node) for node in cycle), str(cycle[0])])
        cycle_edges = ", ".join(f"{'+' if sign > 0 else '-'}({i}, {j}, {c})" for sign, i, j, c in directed_edges)
        printer.information(f"Detected SOCP cycle {cycle_path}: {cycle_edges}")

    return detected_cycles