import numpy as np
import matplotlib.pyplot as plt


def model_A(T=200, m0=3, m=2):
    degree = np.array([m0 - 1] * m0, dtype=float)
    deg_t = np.zeros((T, T))
    for i in range(m0):
        deg_t[i, :m0] = degree[i]

    # we add one new node at each time step
    for new_node in range(m0, T):
        # choose m unique targets uniformly at random
        targets = np.random.choice(new_node, size=m, replace=False)
        for t in targets:
            degree[t] += 1
        degree = np.append(degree, m)
        deg_t[: new_node + 1, new_node] = degree[: new_node + 1]

    return deg_t


def model_B(T=200, N=50, m=2):
    # all nodes exist from start, degrees start at 0
    degree = np.zeros(N, dtype=float)
    deg_t = np.zeros((N, T))
    for t in range(T):
        # preferential attachment probability
        probs = degree / degree.sum() if degree.sum() > 0 else np.ones(N) / N
        targets = np.random.choice(N, size=m, replace=False, p=probs)
        for t_node in targets:
            degree[t_node] += 1
        deg_t[:, t] = degree
    return deg_t


def average_deg(model_func, runs=100, **kwargs):
    sample = model_func(**kwargs)
    N, T = sample.shape
    all_deg = np.zeros((runs, N, T))
    for r in range(runs):
        all_deg[r] = model_func(**kwargs)
    deg_avg = np.mean(all_deg, axis=0)
    return deg_avg


def plot_models_avg_vs_theory(
    T=200, m0=3, m=2, N_B=50, runs=100, model="A", sample_nodes=4
):

    if model == "A":
        model_func = model_A
        kwargs = {"T": T, "m0": m0, "m": m}
    elif model == "B":
        model_func = model_B
        kwargs = {"T": T, "N": N_B, "m": m}

    sample = model_func(**kwargs)
    N, T_sim = sample.shape
    all_deg = np.zeros((runs, N, T_sim))
    for r in range(runs):
        all_deg[r] = model_func(**kwargs)
    deg_avg = np.mean(all_deg, axis=0)

    plt.figure(figsize=(8, 6))
    colors = plt.cm.tab10.colors

    if model == "A":
        # average by birth time
        for idx, node in enumerate(range(12, 16)):
            t_i = max(1, node)
            t_range = np.arange(t_i, T)
            k_emp = deg_avg[node, t_i:]
            k_theory = m * np.log(t_range / t_i) + m
            plt.plot(
                t_range,
                k_emp,
                color=colors[idx % 10],
                alpha=0.7,
                label=f"Birth {node} empirical",
            )
            plt.plot(
                t_range,
                k_theory,
                "--",
                color=colors[idx % 10],
                label=f"Birth {node} theory",
            )
        plt.title(f"Model A: Average degree per birth time over {runs} runs")

    elif model == "B":
        # overall average over all nodes
        avg_over_nodes = np.mean(deg_avg, axis=0)
        theory = 2 * np.arange(T) / N_B
        plt.plot(avg_over_nodes, label="Average empirical (all nodes)", color="blue")
        plt.plot(theory, "--", label="Theory: 2t/N", color="red")
        # sample nodes that have non-zero degree at final time
        nonzero_nodes = np.where(deg_avg[:, -1] > 0)[0]
        for idx, node in enumerate(nonzero_nodes[:sample_nodes]):
            plt.plot(
                deg_avg[node],
                alpha=0.6,
                color=colors[idx % 10],
                label=f"Node {node} empirical",
            )
        plt.title(f"Model B: Average degree over all nodes ({runs} runs)")

    plt.xlabel("Time t")
    plt.ylabel("Degree k_i(t)")
    plt.legend()
    plt.grid(True)
    plt.show()
