import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import networkx as nx


def ba_model(T=100, m0=3, m=2):
    # initial complete graph K_m0 (adj list representation)
    adj = {i: set(j for j in range(m0) if j != i) for i in range(m0)}
    # initial degrees each node has degree m0 - 1
    degree = np.array([m0 - 1] * m0, dtype=float)

    deg_t = np.zeros((T, T))  # track degrees over time for the purpose of ki(t)
    for i in range(m0):
        deg_t[i, :m0] = degree[i]

    # at each step we add a new node with m edges
    # probability proportional to degree
    for new_node in range(m0, T):
        # degrees of existing nodes over total degree sum
        probs = degree[:new_node] / degree[:new_node].sum()
        # choose m unique vertices to connect to based on probs
        targets = np.random.choice(new_node, size=m, replace=False, p=probs)

        # update adjacency list and degrees
        adj[new_node] = set()
        for t in targets:
            adj[new_node].add(t)
            adj[t].add(new_node)
            degree[t] += 1

        degree = np.append(degree, m)

        for i in range(new_node + 1):
            deg_t[i, new_node] = degree[i]

    return adj, degree, deg_t


def animate_ba(T=50, m0=3, m=2):
    adj, _, deg_t = ba_model(T, m0, m)

    fig, ax = plt.subplots(figsize=(7, 7))
    G = nx.Graph()

    # precompute full layout for consistency
    G_full = nx.Graph()
    for i in range(T):
        G_full.add_node(i)
    for i in adj:
        for j in adj[i]:
            G_full.add_edge(i, j)
    pos = nx.spring_layout(G_full, seed=42)

    def update(frame):
        ax.clear()
        G.clear()

        for i in range(frame):
            G.add_node(i)

        for i in range(frame):
            for j in adj.get(i, []):
                if j < frame:
                    G.add_edge(i, j)

        node_sizes = deg_t[:frame, frame] * 25

        nx.draw(
            G,
            pos,
            node_size=node_sizes,
            node_color="skyblue",
            edge_color="lightgrey",
            ax=ax,
        )

        ax.set_title(f"BA Model — t = {frame}", fontsize=14)
        ax.axis("off")

    anim = FuncAnimation(fig, update, frames=range(m0, T), interval=200)
    anim.save("ba_animation.gif", writer="pillow", fps=2)


def averaged_histogram(T=100, m0=3, m=2, runs=1000, bins=30, plot=True):
    all_hist = []

    bin_edges = np.linspace(0, T, bins + 1)

    for _ in range(runs):
        _, degrees, _ = ba_model(T, m0, m)
        hist, _ = np.histogram(degrees, bins=bin_edges, density=False)
        all_hist.append(hist)

    avg_hist = np.mean(all_hist, axis=0)

    # compute bin centers
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # remove trailing zeros
    nonzero_index = np.max(np.where(avg_hist > 0)) + 1
    avg_hist = avg_hist[:nonzero_index]
    bin_centers = bin_centers[:nonzero_index]

    if plot:
        # plot
        plt.figure(figsize=(7, 5))
        plt.bar(
            bin_centers,
            avg_hist,
            width=(bin_edges[1] - bin_edges[0]) * 0.9,
            color="skyblue",
            edgecolor="black",
        )
        plt.yscale("log")
        plt.xlabel("Degree k")
        plt.ylabel("Average count (log scale)")
        plt.title("Averaged histogram")
        plt.grid(True, which="both", ls="--", lw=0.5)
        plt.show()

    return bin_centers, avg_hist


def estimate_alpha(T=100, m0=3, m=2, runs=1000, bins=30):
    # get averaged histogram
    k, P_k = averaged_histogram(T, m0, m, runs=runs, bins=bins, plot=False)

    # avoid log(0) by filtering out zero entries
    mask = (k > 0) & (P_k > 0)
    k_fit = k[mask]
    P_k_fit = P_k[mask]

    # log-log fit
    log_k = np.log(k_fit)
    log_Pk = np.log(P_k_fit)
    slope, intercept = np.polyfit(log_k, log_Pk, 1)
    alpha = -slope

    print(f"Estimated α ≈ {alpha:.3f}")
    print("Theoretical α = 3 (BA model)")

    return alpha


def plot_ki_vs_theory(T=200, m0=5, m=2, runs=1000, nodes_to_plot=None):
    if nodes_to_plot is None:
        nodes_to_plot = np.arange(5, 9)

    # average degree trajectories
    avg_deg = {i: np.zeros(T) for i in nodes_to_plot}

    for _ in range(runs):
        _, _, deg_t = ba_model(T, m0, m)

        for node in nodes_to_plot:
            avg_deg[node] += deg_t[node]

    # average over all runs
    for node in nodes_to_plot:
        avg_deg[node] /= runs

    plt.figure(figsize=(8, 6))
    colors = plt.cm.tab10.colors

    for idx, node in enumerate(nodes_to_plot):
        t_i = max(1, node)  # avoid division by zero
        t = np.arange(t_i, T)

        k_emp = avg_deg[node][t_i:T]
        k_theory = m * np.sqrt(t / t_i)

        color = colors[idx % len(colors)]
        plt.plot(t, k_emp, label=f"Node {node} empirical", color=color)
        plt.plot(t, k_theory, "--", label=f"Node {node} theory", color=color)

    plt.xlabel("t")
    plt.ylabel("k_i(t)")
    plt.title(f"Degree growth k_i(t): empirical vs theory ({runs} runs averaged)")
    plt.grid(True)
    plt.legend()
    plt.show()
