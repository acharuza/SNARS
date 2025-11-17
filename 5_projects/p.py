import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


def check_clustering_er(n=2000, p=0.05, trials=5):
    clustering_values = []

    for t in range(trials):
        G = nx.erdos_renyi_graph(n, p)

        C = nx.average_clustering(G)
        clustering_values.append(C)

        print(f"Trial {t+1}/{trials}: C = {C:.5f}")

    meanC = np.mean(clustering_values)
    stdC = np.std(clustering_values)

    print("\n--- Results ---")
    print(f"Analytical expectation:  C = p = {p}")
    print(f"Simulated mean:          C = {meanC:.5f}")
    print(f"Standard deviation:      {stdC:.5f}")


def generate_community_graph(num_communities=4, N=20, p_in=0.3, p_out=0.02):
    total_nodes = num_communities * N
    G = nx.Graph()

    # assign nodes to communities
    community = {}
    for c in range(num_communities):
        for i in range(N):
            community[c * N + i] = c

    G.add_nodes_from(range(total_nodes))

    rng = np.random.default_rng()

    for i in range(total_nodes):
        for j in range(i + 1, total_nodes):
            if community[i] == community[j]:
                # same community
                if rng.random() < p_in:
                    G.add_edge(i, j)
            else:
                # different communities
                if rng.random() < p_out:
                    G.add_edge(i, j)

    return G, community


def draw_community_graph(G, community):
    pos = nx.spring_layout(G)
    colors = [community[n] for n in G.nodes()]
    degree_sequence = [d for n, d in G.degree()]

    plt.figure(figsize=(7, 7))
    nx.draw(
        G,
        pos,
        node_color=colors,
        cmap="tab10",
        node_size=[50 + 20 * d for d in degree_sequence],
        with_labels=False,
        edge_color="gray",
        alpha=0.7,
    )
    plt.title("Graph with 4 Communities")
    plt.show()


# how watts-strogatz clustering works?
# https://en.wikipedia.org/wiki/Watts%E2%80%93Strogatz_model#Algorithm
# first create a ring lattice where each node is connected to k nearest neighbors K/2 on each side
# for each edge, rewire it with probability p to a randomly chosen node
def plot_ws_clustering_vs_p(n=100, k=4, trials=10):
    p_values = np.linspace(0, 1, 20)

    mean_clust = []
    std_clust = []

    for p in p_values:
        clustering_vals = []
        for _ in range(trials):
            G = nx.watts_strogatz_graph(n, k, p)
            clustering_vals.append(nx.average_clustering(G))
        mean_clust.append(np.mean(clustering_vals))
        std_clust.append(np.std(clustering_vals))

    mean_clust = np.array(mean_clust)
    std_clust = np.array(std_clust)

    plt.figure(figsize=(8, 5))
    plt.errorbar(
        p_values,
        mean_clust,
        yerr=std_clust,
        fmt="o",
        capsize=3,
        markersize=5,
        label="WS network",
    )
    plt.xlabel("p")
    plt.ylabel("Average clustering coefficient ⟨C⟩")
    plt.title(f"Watts-Strogatz Model: ⟨C⟩ vs p")
    plt.grid(True, ls="--", alpha=0.5)
    plt.show()
