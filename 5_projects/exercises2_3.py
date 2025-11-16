import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


# exercise 2
def er_adjacency_matrix(N, p):
    rand_matrix = np.random.rand(N, N)
    # we take the upper triangle to avoid self-loops and double edges
    # we check where random values are less than p to create edges
    upper = np.triu(rand_matrix < p, 1)

    # merge upper and lower triangles to create a symmetric adjacency matrix
    A = upper + upper.T

    return A.astype(int)


# exercise 3
def draw_er_graph(G, ax):
    nx.draw(
        G,
        node_size=30,
        with_labels=False,
        pos=nx.kamada_kawai_layout(G),
        ax=ax,
        edge_color="gray",
    )
    ax.set_title("Erdős-Rényi graph\nbased on adjacency matrix")


def plot_degree_distribution(G, ax):
    degrees = [d for _, d in G.degree()]
    ax.hist(degrees, bins=range(max(degrees) + 1), edgecolor="black", alpha=0.7)
    ax.set_title("Degree Distribution")
    ax.set_xlabel("Degree")
    ax.set_ylabel("Frequency")


def plot_over_many_er_graphs(num_graphs, N, p, ax):
    all_degrees = []

    for _ in range(num_graphs):
        A = er_adjacency_matrix(N, p)
        G = nx.from_numpy_array(A)
        all_degrees.extend([d for _, d in G.degree()])

    ax.hist(all_degrees, bins=range(max(all_degrees) + 1), edgecolor="black", alpha=0.7)
    ax.set_title(
        f"Degree Distribution over {num_graphs} ER Graphs\n(N={N}, p={p})\nexpected avg degree={(N-1)*p:.1f}"
    )
    ax.set_xlabel("Degree")
    ax.set_ylabel("Frequency")


if __name__ == "__main__":
    N = 50
    p = 0.1
    num_graphs = 100

    fig, axs = plt.subplots(1, 3, figsize=(12, 5))

    A = er_adjacency_matrix(N, p)
    G = nx.from_numpy_array(A)

    draw_er_graph(G, axs[0])
    plot_degree_distribution(G, axs[1])
    plot_over_many_er_graphs(num_graphs, N, p, axs[2])

    plt.tight_layout()
    plt.savefig("ex_2_3.png")
