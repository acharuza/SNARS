"""louvain algorithm community detection

reference:
https://en.wikipedia.org/wiki/Louvain_method#Algorithm_description

1. Initialization

Each node starts in its own community.

2. Local Moving Phase

For each node, evaluate the modularity gain if it moves to a neighbor’s community.

Move to the community that maximizes modularity.

3. Aggregation Phase

Collapse each community into a super-node.

Recompute edges between super-nodes (weights = sum of edges between communities).

4. Repeat

Repeat local moving + aggregation until modularity no longer improves.

"""

import networkx as nx
import random
from collections import defaultdict


# compute modularity
def compute_modularity(G, communities):
    m = G.size(weight="weight")
    Q = 0.0
    degrees = dict(G.degree(weight="weight"))

    # group nodes by community
    comm_nodes = defaultdict(set)
    for node, comm in communities.items():
        comm_nodes[comm].add(node)

    for nodes in comm_nodes.values():
        L_c = 0.0  # sum of internal edge weights
        K_c = 0.0  # sum of degrees
        for u in nodes:
            K_c += degrees[u]
            for v in G.neighbors(u):
                if v in nodes:
                    L_c += G[u][v].get("weight", 1)
        Q += (L_c / (2 * m)) - (K_c / (2 * m)) ** 2

    return Q


# local moving phase
def local_moving(G, communities):
    nodes = list(G.nodes())
    random.shuffle(nodes)

    for node in nodes:
        current_comm = communities[node]
        best_comm = current_comm
        best_gain = 0

        # neighbor communities to consider
        neighbor_comms = set(communities[neigh] for neigh in G.neighbors(node))
        for comm in neighbor_comms:
            if comm == current_comm:
                continue

            # temporarily move node
            communities[node] = comm
            # compute modularity gain
            Q_new = compute_modularity(G, communities)
            communities[node] = current_comm
            Q_old = compute_modularity(G, communities)
            gain = Q_new - Q_old

            # if gain is better, update best
            if gain > best_gain:
                best_gain = gain
                best_comm = comm

        # move to best community found
        if best_comm != current_comm:
            communities[node] = best_comm


# aggregation phase
def aggregate_graph(G, communities):
    agg_G = nx.Graph()
    comm_nodes = defaultdict(list)

    # group nodes by community
    for node, comm in communities.items():
        comm_nodes[comm].append(node)

    # add super-nodes for each community
    for comm in comm_nodes.keys():
        agg_G.add_node(comm)

    # add edges between super-nodes
    for u, v, data in G.edges(data=True):
        w = data.get("weight", 1)
        cu = communities[u]
        cv = communities[v]
        if agg_G.has_edge(cu, cv):
            agg_G[cu][cv]["weight"] += w
        else:
            agg_G.add_edge(cu, cv, weight=w)

    return agg_G


def louvain_community_detection(G, eps=1e-5):
    # initialization
    communities = {node: idx for idx, node in enumerate(G.nodes())}
    prev_modularity = compute_modularity(G, communities)
    curr_modularity = prev_modularity + 1

    # tracking nodes to communities

    node2community = {node: node for node in G.nodes()}

    while abs(curr_modularity - prev_modularity) > eps:
        prev_modularity = curr_modularity

        # local moving phase
        local_moving(G, communities)  # modifies communities in-place

        for node, comm in communities.items():
            node2community[node] = comm

        # compute modularity
        curr_modularity = compute_modularity(G, communities)

        # aggregation phase
        G = aggregate_graph(G, communities)

        # reinitialize communities for super-nodes
        communities = {node: node for node in G.nodes()}

    # relabel communities to be consecutive integers
    unique_comms = sorted(set(node2community.values()))
    comm_mapping = {old: new for new, old in enumerate(unique_comms)}
    for node in node2community:
        node2community[node] = comm_mapping[node2community[node]]

    return node2community


# because louvain con is that it can get stuck in local optima, I run it multiple times and take the best result
def best_of_louvain(G, n_iter=10, eps=1e-5):
    best_communities = None
    best_modularity = float("-inf")

    for _ in range(n_iter):
        communities = louvain_community_detection(G, eps)
        modularity = compute_modularity(G, communities)

        if modularity > best_modularity:
            best_modularity = modularity
            best_communities = communities

    return best_communities


# now let's check it

if __name__ == "__main__":
    # Define community sizes
    sizes = [10, 15, 20]  # 3 communities

    # probability matrix: higher probability inside communities
    p_intra = 0.9  # edges within community
    p_inter = 0.01  # edges between communities
    probs = [
        [p_intra if i == j else p_inter for j in range(len(sizes))]
        for i in range(len(sizes))
    ]

    # generate SBM graph
    G = nx.stochastic_block_model(sizes, probs, seed=42)

    # run louvain community detection
    detected_communities = best_of_louvain(G, n_iter=500, eps=1e-9)
    print("Detected communities:")
    for node, comm in detected_communities.items():
        print(f"Node {node}: Community {comm}")

    # sizes of detected communities
    comm_sizes = defaultdict(int)
    for comm in detected_communities.values():
        comm_sizes[comm] += 1
    print("\nDetected community sizes:")
    for comm, size in comm_sizes.items():
        print(f"Community {comm}: Size {size}")
