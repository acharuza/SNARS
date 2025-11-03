import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.io import mmread
from scipy.stats import linregress


def import_data(filepath):
    matrix = mmread(filepath)
    G = nx.from_scipy_sparse_array(matrix)
    return G


def get_degree_sequence(G):
    degrees = [d for _, d in G.degree() if d > 0]
    return degrees


def set_ax_scale(ax, axes_scale):
    if axes_scale == "double_log":
        ax.set_xscale("log")
        ax.set_yscale("log")
    elif axes_scale == "log_x":
        ax.set_xscale("log")
    elif axes_scale == "log_y":
        ax.set_yscale("log")


def get_histogram_data(data, bins=50):
    if bins == "logarithmic":
        min_val = min(data)
        max_val = max(data)
        bins = np.logspace(np.log10(min_val), np.log10(max_val), 50)

    hist, bin_edges = np.histogram(data, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    return bin_centers, hist, bin_edges


def draw_histogram(data, ax, axes_scale="linear", bins=50, title=""):
    if bins == "logarithmic":
        min_val = min(data)
        max_val = max(data)
        bins = np.logspace(np.log10(min_val), np.log10(max_val), 50)

    ax.hist(data, bins=bins, edgecolor="black", alpha=0.7, rwidth=1)

    set_ax_scale(ax, axes_scale)

    ax.set_title(title)
    ax.set_xlabel(f"Value{' (log scale)' if ax.get_xscale() == 'log' else ''}")
    ax.set_ylabel(f"Frequency{' (log scale)' if ax.get_yscale() == 'log' else ''}")


def survival_function(data):
    sorted_data = np.sort(data)
    survival_prob = 1.0 - np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    return sorted_data, survival_prob


def draw_survival_function(data, ax=None, axes_scale="linear", x_min=None):
    # Survival function = 1 - CDF
    x_empirical, y_empirical = survival_function(data)
    ax.step(x_empirical, y_empirical, where="post", color="orange")

    set_ax_scale(ax, axes_scale)
    ax.set_title("Survival Function")
    ax.set_xlabel(f"Value{' (log scale)' if ax.get_xscale() == 'log' else ''}")
    ax.set_ylabel(
        f"Survival Probability{' (log scale)' if ax.get_yscale() == 'log' else ''}"
    )


def estimate_alpha(data, plot=True, ax=None, x_min=None):
    bin_centers, hist, _ = get_histogram_data(data, bins="logarithmic")

    mask = hist > 0 if x_min is None else (bin_centers >= x_min) & (hist > 0)
    x = bin_centers[mask]
    y = hist[mask]

    if len(x) < 2:
        raise ValueError("Not enough data points for regression after filtering.")

    log_x = np.log(x)
    log_y = np.log(y)

    slope, intercept, r_value, p_value, std_err = linregress(log_x, log_y)

    # for power law pdf: P(k) ~ k^{-alpha}, so log(P) = -alpha log(k) + const
    # slope = -alpha
    alpha = -slope

    if plot:
        ax.loglog(x, y, "o", label="Empirical PDF")
        x_fit = np.logspace(np.log10(min(x)), np.log10(max(x)), 100)
        y_fit = np.exp(intercept) * (x_fit**slope)
        ax.loglog(x_fit, y_fit, label=f"Fit: alpha={alpha:.2f}")
        ax.set_xlabel("Degree (log scale)")
        ax.set_ylabel("Frequency (log scale)")
        ax.set_title("Power Law Fit to Degree Distribution")
        ax.legend()

    return alpha


def compute_mle_estimate(data, x_min=None):
    if x_min is None:
        x_min = min(data)
    data = [d for d in data if d >= x_min]
    alpha = 1 + len(data) / np.sum(np.log(np.array(data) / x_min))
    return alpha


def plot_empirical_ccdf(data, ax):
    data = np.asarray(data)
    data = data[data > 0]
    x_sorted = np.sort(data)
    n = len(x_sorted)
    S = (n - np.arange(n)) / n  # survival = P(X ≥ x)
    ax.loglog(x_sorted, S, marker=".", linestyle="none", alpha=0.7)
    ax.set_xlabel("x")
    ax.set_ylabel("S(x) = P(X ≥ x)")
    ax.set_title("Empirical CCDF (log–log)")
    ax.grid(True, which="both", ls="--", alpha=0.4)


if __name__ == "__main__":
    G = import_data("bio-yeast.mtx")
    x_min = 2
    degrees = get_degree_sequence(G)
    fig, axs = plt.subplots(2, 3, figsize=(15, 7))
    plot_empirical_ccdf(degrees, ax=axs[0][0])
    draw_histogram(
        degrees,
        ax=axs[0][1],
        title="Degree Distribution Histogram",
    )
    draw_histogram(
        degrees,
        ax=axs[0][2],
        axes_scale="double_log",
        title="Degree Distribution Histogram\n(Double Log Scale)",
    )
    draw_histogram(
        degrees,
        ax=axs[1][0],
        axes_scale="double_log",
        bins="logarithmic",
        title="Degree Distribution Histogram\n(Log Bins & Double Log Scale)",
    )
    draw_survival_function(degrees, ax=axs[1][1], axes_scale="double_log")
    estimate_alpha(degrees, plot=True, ax=axs[1][2], x_min=x_min)
    fig.suptitle(
        "Analysis of Degree Distribution in Yeast Protein-Protein Interaction Network",
        fontsize=12,
    )
    alpha = compute_mle_estimate(degrees, x_min=x_min)
    axs[1][2].text(
        0.2,
        0.2,
        f"MLE Estimate of alpha: {alpha:.2f}.",
        transform=axs[1][2].transAxes,
        fontsize=10,
        ha="center",
    )
    axs[1][2].axis("off")
    plt.tight_layout()
    plt.savefig("degree_distribution_analysis.png", dpi=300)

    # What if x_min is unknown? Then we must estimate it along with alpha from the data.
    # One of the approaches is described in Clauset et al., "Power-law distributions in empirical data", SIAM Review 2009.
    # We first consider a set of candidate x_min values, and for each we compute the MLE estimate of alpha and the Kolmogorov-Smirnov statistic:
    # the maximum distance between the empirical CDF and the fitted power-law CDF.
    # finally, we select the x_min that minimizes the KS statistic.

    # Consequences:
    # 1. if we choose too low x_min, we include data points that do not follow power-law behavior, leading to biased estimates of alpha.
    # 2. if we choose too high x_min, we discard valuable data, increasing the variance of our estimates.
    # 3. estimating x_min from the data makes the uncertainty of estimate of alpha larger

    # Other way visual inspectiong if you plot the empirical CCDF on log–log axes, you should see a straight line for the region where the power law holds.
