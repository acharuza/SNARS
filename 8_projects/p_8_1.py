import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict
import numpy as np
from scipy.stats import linregress


def clustering_coefficient(G):
    C = {}
    for u in G.nodes():
        neighbors = list(G.neighbors(u))
        k = len(neighbors)
        if k < 2:
            C[u] = 0.0
            continue

        triangles = 0
        for i in range(k):
            for j in range(i + 1, k):
                a = neighbors[i]
                b = neighbors[j]
                # check if neighbors are connected
                if G.has_edge(a, b):
                    triangles += 1

        # possible edges among neighbors
        possible = k * (k - 1) / 2
        C[u] = triangles / possible
    return C


def compute_Ck(G, C):
    degrees = dict(G.degree())
    Ck = defaultdict(list)

    for node in G.nodes():
        k = degrees[node]
        Ck[k].append(C[node])

    k_vals = sorted(Ck.keys())
    Ck_vals = [sum(Ck[k]) / len(Ck[k]) for k in k_vals]

    return k_vals, Ck_vals


def plot_Ck(k_vals, Ck_vals, title="C(k) vs k"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Log-log plot
    ax1.loglog(k_vals, Ck_vals, "o", label="C(k)")

    mask = np.array(Ck_vals) > 0
    k_vals_nonzero = np.array(k_vals)[mask]
    Ck_vals_nonzero = np.array(Ck_vals)[mask]

    if len(k_vals_nonzero) >= 2:
        log_k = np.log(k_vals_nonzero)
        log_Ck = np.log(Ck_vals_nonzero)
        slope, intercept, r_value, p_value, std_err = linregress(log_k, log_Ck)

        k_fit = np.logspace(
            np.log10(min(k_vals_nonzero)), np.log10(max(k_vals_nonzero)), 100
        )
        Ck_fit = np.exp(intercept) * (k_fit**slope)
        ax1.loglog(k_fit, Ck_fit, "--", label=f"Fit: slope={slope:.2f}")

    ax1.set_xlabel("k (log)")
    ax1.set_ylabel("C(k) (log)")
    ax1.set_title(f"{title} (Log-Log)")
    ax1.grid(True, which="both", ls=":")
    ax1.legend()

    # Linear plot
    ax2.plot(k_vals, Ck_vals, "o-", label="C(k)")
    ax2.set_xlabel("k")
    ax2.set_ylabel("C(k)")
    ax2.set_title(f"{title} (Linear)")
    ax2.grid(True, ls=":")
    ax2.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    G = nx.erdos_renyi_graph(200, 0.05)
    C = clustering_coefficient(G)
    k_vals, Ck_vals = compute_Ck(G, C)
    plot_Ck(k_vals, Ck_vals)
