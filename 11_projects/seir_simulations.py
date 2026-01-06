# https://web.pdx.edu/~gjay/teaching/mth271_2020/html/09_SEIR_model.html
# i checked to better understand seir model because i wasn't sure if exposed lasts specific time or is probabilistic
# and at least there it's probabilistic so that's what im doing here too


import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random


# ----------------------------
# Utilities
# ----------------------------
def set_seed(seed=123):
    random.seed(seed)
    np.random.seed(seed)


# States
S, E, I, R = 0, 1, 2, 3


def initialize_status(nodes, initial_infected_fraction):
    status = {node: S for node in nodes}
    n_init = int(len(nodes) * initial_infected_fraction)
    infected = random.sample(nodes, n_init)
    for node in infected:
        status[node] = I
    return status


# ----------------------------
# SEIR dynamics
# ----------------------------
def simulate_seir(
    G,
    beta=0.01,  # infection probability
    sigma=0.1,  # exposed -> infected
    gamma=0.05,  # recovery
    max_steps=200,
    initial_infected_fraction=0.02,
):
    nodes = list(G.nodes())
    status = initialize_status(nodes, initial_infected_fraction)

    S_hist, E_hist, I_hist, R_hist = [], [], [], []

    for _ in range(max_steps):
        new_status = status.copy()

        for node in nodes:
            if status[node] == S:
                # infection from neighbors
                infected_neighbors = sum(
                    1 for nbr in G.neighbors(node) if status[nbr] == I
                )
                for _ in range(infected_neighbors):
                    if random.random() < beta:
                        new_status[node] = E
                        break

            elif status[node] == E:
                if random.random() < sigma:
                    new_status[node] = I

            elif status[node] == I:
                if random.random() < gamma:
                    new_status[node] = R

        status = new_status

        # record fractions
        values = list(status.values())
        S_hist.append(values.count(S) / len(nodes))
        E_hist.append(values.count(E) / len(nodes))
        I_hist.append(values.count(I) / len(nodes))
        R_hist.append(values.count(R) / len(nodes))

    return S_hist, E_hist, I_hist, R_hist


def calculate_r0(beta, s0, gamma):
    return (beta * s0) / gamma


# ----------------------------
# Run simulation
# ----------------------------
set_seed(123)

N = 1000
k_avg = 4
p = k_avg / N
m = k_avg // 2

max_steps = 150

# ER network
G_er = nx.erdos_renyi_graph(N, p)

# BA network
G_ba = nx.barabasi_albert_graph(N, m)

params = dict(
    beta=0.2,
    sigma=0.1,
    gamma=0.1,
    initial_infected_fraction=0.01,
    max_steps=max_steps,
)

s0 = 1 - params["initial_infected_fraction"]

S_er, E_er, I_er, R_er = simulate_seir(G_er, **params)
S_ba, E_ba, I_ba, R_ba = simulate_seir(G_ba, **params)

T = len(S_er)
t = np.arange(T)

colors = {
    "S": "#4C72B0",
    "E": "#DD8452",
    "I": "#C44E52",
    "R": "#55A868",
}

fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

# -------- ER --------
axes[0].plot(t, S_er, color=colors["S"], label="S")
axes[0].plot(t, E_er, color=colors["E"], label="E")
axes[0].plot(t, I_er, color=colors["I"], label="I")
axes[0].plot(t, R_er, color=colors["R"], label="R")

axes[0].fill_between(t, S_er, color=colors["S"], alpha=0.15)
axes[0].fill_between(t, E_er, color=colors["E"], alpha=0.15)
axes[0].fill_between(t, I_er, color=colors["I"], alpha=0.15)
axes[0].fill_between(t, R_er, color=colors["R"], alpha=0.15)

axes[0].set_title("SEIR on ER network")
axes[0].set_xlabel("Time")
axes[0].set_ylabel("Fraction of population")
axes[0].set_ylim(0, 1)
axes[0].grid(True, alpha=0.3)

# -------- BA --------
axes[1].plot(t, S_ba, color=colors["S"])
axes[1].plot(t, E_ba, color=colors["E"])
axes[1].plot(t, I_ba, color=colors["I"])
axes[1].plot(t, R_ba, color=colors["R"])

axes[1].fill_between(t, S_ba, color=colors["S"], alpha=0.15)
axes[1].fill_between(t, E_ba, color=colors["E"], alpha=0.15)
axes[1].fill_between(t, I_ba, color=colors["I"], alpha=0.15)
axes[1].fill_between(t, R_ba, color=colors["R"], alpha=0.15)

axes[1].set_title("SEIR on BA network")
axes[1].set_xlabel("Time")
axes[1].set_ylim(0, 1)
axes[1].grid(True, alpha=0.3)

# Shared legend
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=4)

plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.show()
