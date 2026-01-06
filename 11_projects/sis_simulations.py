import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm


def set_seed(seed_value=123):
    random.seed(seed_value)
    np.random.seed(seed_value)


def initialize_status(nodes, initial_infected_fraction):
    status = {}
    n_infected_init = int(len(nodes) * initial_infected_fraction)
    infected_nodes = random.sample(nodes, n_infected_init)
    for node in nodes:
        status[node] = 1 if node in infected_nodes else 0
    return status


def recover_infected(status, current_infected, gamma):
    for node in current_infected:
        if random.random() < gamma:
            status[node] = 0


def infect_susceptible_network(G, new_status, old_status, current_infected, beta):
    """infection can spread only to neighbors in the network"""
    for node in current_infected:
        for neighbor in G.neighbors(node):
            if old_status[neighbor] == 0:
                if random.random() < beta:
                    new_status[neighbor] = 1


def has_epidemic_ended(status):
    return all(s == 0 for s in status.values())


def simulate_sis(G, beta, gamma, max_steps=1000, initial_infected_fraction=0.05):
    # mode network is spreading to neighbors only
    # mode mean_field is spreading in mean-field manner
    nodes = list(G.nodes())
    status = initialize_status(nodes, initial_infected_fraction)

    infected_history = []

    for _ in range(max_steps):
        current_infected = [n for n, s in status.items() if s == 1]
        fraction = len(current_infected) / len(nodes)
        infected_history.append(fraction)

        if has_epidemic_ended(status):
            break

        new_status = status.copy()

        # i -> s
        recover_infected(new_status, current_infected, gamma)
        # s -> i
        infect_susceptible_network(G, new_status, status, current_infected, beta)

        status = new_status

    return infected_history


def calculate_steady_state_er(beta, gamma, avg_degree):
    # based on lecture slides
    lam = beta / gamma
    # checking if gamma / (beta * avg_degree) < 1 to avoid negative values
    if lam * avg_degree > 1:
        return 1 - (gamma / (beta * avg_degree))
    else:
        return 0


def calculate_epidemic_threshold_er(avg_degree, initial_infected_fraction=0.05):
    # based on lecture slides
    return 1 / (avg_degree * (1 - initial_infected_fraction))


def calculate_epidemic_threshold_ba(avg_degree, avg_squared_degree):
    # based on lecture slides
    # so okay lambda is small when beta is small so the infection has low probability or
    # when gamma is big so the probability of getting healthy is high so its concerning because
    # even when the virus doesnt spread much and we do everything we could to heal ill people
    # the epidemic will still burst out because of the network structure
    return avg_degree / avg_squared_degree


def get_averaged_steady_state(G, beta, gamma, runs=10):
    """runs simultations multiple times and averages the steady state values"""
    final_values = []
    for _ in range(runs):
        history = simulate_sis(G, beta, gamma, max_steps=500)

        # get steady state from the last 50 steps of this run
        if len(history) > 50:
            val = np.mean(history[-50:])
        else:
            val = history[-1]  # epidemic died out
        final_values.append(val)

    return np.mean(final_values)


# main execution
set_seed(123)

# ER NETWORK

N = 1000
k_avg = 5
p = k_avg / N
gamma = 0.1
initial_infected_fraction = 0.01

G = nx.erdos_renyi_graph(N, p)
# it may differ sligthly because of randomness
actual_k_avg = np.mean([d for _, d in G.degree()])
actual_k2_avg = np.mean([d**2 for _, d in G.degree()])

# i include epidemic threshold in lambda values to ensure it's captured
epidemic_threshold = calculate_epidemic_threshold_er(
    actual_k_avg, initial_infected_fraction
)
lambda_values = np.sort(
    np.unique(np.concatenate((np.linspace(0, 1, 20), [epidemic_threshold])))
)

network_results = []
analytical_results = []

for lam in tqdm(lambda_values, desc="Simulating SIS Model", total=len(lambda_values)):
    beta = lam * gamma

    # infect neighbors in the network
    avg_net = get_averaged_steady_state(G, beta, gamma, runs=10)
    network_results.append(avg_net)

    # analytical theory
    res_theory = calculate_steady_state_er(beta, gamma, actual_k_avg)
    analytical_results.append(res_theory)


plt.figure(figsize=(10, 6))
plt.plot(
    lambda_values,
    network_results,
    "bo-",
    label="ER network simulation \n(averaged over 10 runs)",
)
plt.plot(lambda_values, analytical_results, "r--", label="analytical theory")
plt.axvline(
    epidemic_threshold,
    color="green",
    linestyle=":",
    label=f"ER epidemic threshold $\lambda_c$={epidemic_threshold:.3f}",
)
plt.title(f"SIS Model on ER network {r"$\gamma$"}={gamma} N={N}")
plt.xlabel(r"$\lambda = \beta / \gamma$")
plt.ylabel("steady state of infected fraction")
plt.legend()
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.grid(True)
plt.savefig("sis_erdos_renyi.png", dpi=300)
plt.show()

# BA NETWORK

# generate BA network
m = k_avg // 2  # roughly same average degree as ER
G_ba = nx.barabasi_albert_graph(N, m)
degrees_ba = np.array([d for _, d in G_ba.degree()])
actual_k_avg_ba = np.mean(degrees_ba)
actual_k2_avg_ba = np.mean(
    degrees_ba**2
)  # second moment for epidemic threshold calculation

lambda_c_ba = calculate_epidemic_threshold_ba(actual_k_avg_ba, actual_k2_avg_ba)

lambda_values_ba = np.sort(
    np.unique(np.concatenate((np.linspace(0, 1, 20), [lambda_c_ba])))
)

network_results_ba = []
analytical_results_ba = []

for lam in tqdm(
    lambda_values_ba, desc="Simulating BA SIS Model", total=len(lambda_values_ba)
):
    beta = lam * gamma

    avg_net_ba = get_averaged_steady_state(G_ba, beta, gamma, runs=10)
    network_results_ba.append(avg_net_ba)

plt.figure(figsize=(10, 6))
plt.plot(
    lambda_values_ba,
    network_results_ba,
    "bo-",
    label="BA network simulation\n(averaged over 10 runs)",
)
plt.axvline(
    lambda_c_ba,
    color="green",
    linestyle=":",
    label=f"BA epidemic threshold λ_c={lambda_c_ba:.3f}",
)
plt.title(f"SIS Model on BA network (N={N}, γ={gamma})")
plt.xlabel(r"$\lambda = \beta / \gamma$")
plt.ylabel("steady state of infected fraction")
plt.legend()
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.grid(True)
plt.savefig("sis_ba_network.png", dpi=300)
plt.show()
