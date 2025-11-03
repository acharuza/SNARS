import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from power_law_analysis import import_data


def randomize_edges(G, nswap_factor=10):
    G_rewired = G.copy()
    nswap = nswap_factor * G.number_of_edges()
    nx.double_edge_swap(G_rewired, nswap=nswap, max_tries=nswap * 10)
    return G_rewired


def knn_by_degree(G):
    ann = nx.average_neighbor_degree(G)
    deg = dict(G.degree())
    rows = []
    for n in G.nodes():
        k = deg[n]
        if k == 0:
            continue
        rows.append({"node": n, "k": k, "k_nn_node": ann[n]})
    df = pd.DataFrame(rows)
    summary = (
        df.groupby("k")["k_nn_node"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "k_nn_mean", "std": "k_nn_std", "count": "n_nodes"})
    )
    return df, summary


def plot_knn_comparison(orig_summary, rewired_summary, title, ax):
    ks1 = orig_summary["k"].values
    means1 = orig_summary["k_nn_mean"].values
    stds1 = orig_summary["k_nn_std"].fillna(0).values

    ks2 = rewired_summary["k"].values
    means2 = rewired_summary["k_nn_mean"].values
    stds2 = rewired_summary["k_nn_std"].fillna(0).values

    ax.errorbar(
        ks1, means1, yerr=stds1, marker="o", linestyle="-", capsize=3, label="Original"
    )
    ax.errorbar(
        ks2,
        means2,
        yerr=stds2,
        marker="s",
        linestyle="--",
        capsize=3,
        label="Rewired",
    )
    ax.set_xlabel("Degree k")
    ax.set_ylabel("Average neighbor degree k_nn(k)")
    ax.set_title(title)
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend(fontsize=7)


# Example graphs
graphs = {
    "USAir97": import_data("inf-USAir97.mtx"),
    "Dolphins": import_data("dolphins.mtx"),
    "Bio-Yeast": import_data("bio-yeast.mtx"),
    "Zachary Karate Club": nx.karate_club_graph(),
    "Erdos-Renyi (n=100, p=0.05)": nx.erdos_renyi_graph(100, 0.05, seed=42),
    "Barabasi-Albert (n=100, m=2)": nx.barabasi_albert_graph(100, 2, seed=42),
}

fig, axs = plt.subplots(2, 4, figsize=(15, 10))
axs = axs.flatten()
results = []

for i, (name, G) in enumerate(graphs.items()):
    _, orig_summary = knn_by_degree(G)
    corr_before = round(nx.degree_pearson_correlation_coefficient(G), 4)
    G_rewired = randomize_edges(G)
    _, rewired_summary = knn_by_degree(G_rewired)
    corr_after = round(nx.degree_pearson_correlation_coefficient(G_rewired), 4)

    results.append([name, corr_before, corr_after])

    plot_knn_comparison(orig_summary, rewired_summary, name, ax=axs[i])

results_df = pd.DataFrame(
    results,
    columns=[
        "Graph",
        "Corr Before",
        "Corr After",
    ],
)

ax_table = plt.subplot2grid((2, 4), (1, 2), colspan=2)

ax_table.axis("off")
table = ax_table.table(
    cellText=results_df.values,
    colLabels=results_df.columns,
    loc="center",
    cellLoc="center",
)
table.auto_set_font_size(False)
table.set_fontsize(10)
ax_table.set_title("Correlation Coefficients")

axs[6].axis("off")
axs[7].axis("off")

plt.tight_layout()
plt.savefig("knn_degree_correlation_comparison.png", dpi=300)


# Correlation coefficient for network is the Pearson correlation coefficient between the degress of pais of nodes connected by an edge.
# Positive values: high degree nodes tend to connect to high degree nodes
# Negative values: high degree nodes tend to connect to low degree nodes
# 0 random
