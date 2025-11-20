from collections import defaultdict


base_edges = {
    ("A", "B"): "AB",
    ("B", "E"): "BE",
    ("A", "D"): "AD",
    ("B", "D"): "BD",
    ("D", "E"): "DE",
    ("A", "C"): "AC",
    ("C", "D"): "CD",
    ("D", "F"): "DF",
    ("F", "E"): "FE",
}

# making edges bidirectional
edges = dict(base_edges)  # copy original
for (u, v), name in list(base_edges.items()):
    rev_name = name[::-1]  # reverse label, e.g. "AB" -> "BA"
    edges[(v, u)] = rev_name

# list of nodes you can move to directly from a given node
adj = defaultdict(list)
for (u, v), name in edges.items():
    adj[u].append(v)


def edge_cost(edge_name, k):
    # 24 edges
    const_edges = {"AB", "BA", "CD", "DC", "FE", "EF"}
    # 2x edges
    double_edges = {"BE", "EB", "DE", "ED"}
    # x edges
    single_edges = {"AD", "DA", "BD", "DB", "AC", "CA", "DF", "FD"}

    if edge_name in const_edges:
        return 24

    if edge_name in double_edges:
        return 2 * k

    if edge_name in single_edges:
        return k

    raise ValueError(f"Unknown edge name: {edge_name}")


# generate all simple paths from A to E
def all_paths(start, goal):
    paths = []
    stack = [(start, [start])]
    while stack:
        node, path = stack.pop()
        if node == goal:
            paths.append(path)
            continue
        for nxt in adj[node]:
            if nxt not in path:  # simple paths only
                stack.append((nxt, path + [nxt]))
    return paths


paths = all_paths("A", "E")

# node paths to edge paths
edge_paths = []
for p in paths:
    ep = []
    for i in range(len(p) - 1):
        ep.append(edges[(p[i], p[i + 1])])
    edge_paths.append(ep)


def path_cost(edge_path, loads):
    return sum(edge_cost(e, loads[e]) for e in edge_path)


# calculate cost if a player moves from old_edge_path to edge_path
def path_cost_if_player_moves(edge_path, old_edge_path, loads):
    new_loads = loads.copy()

    for e in old_edge_path:
        new_loads[e] -= 1

    for e in edge_path:
        new_loads[e] += 1

    return sum(edge_cost(e, new_loads[e]) for e in edge_path)


N = 24  # number of players

# initial assignment: all players use path 0
player_paths = [0] * N
loads = defaultdict(int)
for _ in range(N):
    for e in edge_paths[0]:
        loads[e] += 1


def best_response_dynamics():
    improved = True
    step = 0

    while improved:
        improved = False
        step += 1
        print(f"Step {step}")

        # for each player, see if they can improve their cost by switching paths
        for player in range(N):
            current_p = player_paths[player]  # current path index
            current_ep = edge_paths[current_p]  # current edge path
            current_cost = path_cost(current_ep, loads)  # current cost

            best_p = current_p
            best_cost = current_cost

            # try switching to each other path
            for p_idx, new_ep in enumerate(edge_paths):
                if p_idx == current_p:
                    continue

                new_cost = path_cost_if_player_moves(new_ep, current_ep, loads)
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_p = p_idx

            # if a better path was found, switch to it
            if best_p != current_p:
                for e in current_ep:
                    loads[e] -= 1
                for e in edge_paths[best_p]:
                    loads[e] += 1

                player_paths[player] = best_p
                improved = True
                print(
                    f"  Player {player} switches path {current_p} → {best_p} "
                    f"({current_cost:.2f} → {best_cost:.2f})"
                )

    print("I hope we reached a Nash equilibrium!")
    return player_paths, loads


final_paths, final_loads = best_response_dynamics()

print("\nFinal path usage:")
usage = defaultdict(int)
for p in final_paths:
    usage[p] += 1
for p, u in usage.items():
    print(f"Path {p}: {u} players\n  Path nodes: {paths[p]}")

print("\nFinal edge loads:")
for e in sorted(edges.values()):
    print(f"{e}: {final_loads[e]} users (cost {edge_cost(e, final_loads[e])})")
