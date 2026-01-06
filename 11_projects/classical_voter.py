import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random


def sign(x):
    """Returns 1 if x > 0, -1 if x < 0. Returns 0 if x == 0."""
    if x > 0:
        return 1
    elif x < 0:
        return -1
    return 0


class ModifiedVoterModel:
    def __init__(
        self, G, propaganda=0.0, ideology_std=0.0, anti_fraction=0.0, anti_strength=0.0
    ):
        self.G = G
        self.nodes = list(G.nodes())
        self.propaganda = propaganda
        self.anti_fraction = anti_fraction
        self.anti_strength = anti_strength

        self.anti_nodes = set(
            random.sample(self.nodes, int(self.anti_fraction * len(self.nodes)))
        )

        self.opinions = None

        self.ideology = {n: np.random.normal(0, ideology_std) for n in self.nodes}

    def step(self):
        """Pick a random node and update opinion based on Neighbors + Propaganda + Ideology"""
        node = random.choice(self.nodes)

        peer_pressure = sum(
            self.opinions[neighbor] for neighbor in self.G.neighbors(node)
        )

        total_influence = peer_pressure + self.propaganda + self.ideology[node]

        if node in self.anti_nodes:
            total_influence = (
                -self.anti_strength * peer_pressure
                + self.propaganda
                + self.ideology[node]
            )

        new_opinion = sign(total_influence)

        self.opinions[node] = new_opinion

    def initialize_opinions(self, initial_opinions):
        self.opinions = initial_opinions.copy()

    def get_average_opinion(self):
        return np.mean(list(self.opinions.values()))


N = 500
k_avg = 6
steps = 10000

G = nx.erdos_renyi_graph(N, k_avg / N)
opinions_initial = {n: random.choice([-1, 1]) for n in G.nodes()}

# no effects (classical voter model)
model_classic = ModifiedVoterModel(G, propaganda=0.0, ideology_std=0.0)
model_classic.initialize_opinions(opinions_initial)

# propaganda only
model_propaganda = ModifiedVoterModel(G, propaganda=2.0, ideology_std=0.0)
model_propaganda.initialize_opinions(opinions_initial)

# ideology only
model_ideology = ModifiedVoterModel(G, propaganda=0.0, ideology_std=5.0)
model_ideology.initialize_opinions(opinions_initial)

# both effects
model_both = ModifiedVoterModel(G, propaganda=2.0, ideology_std=5.0)
model_both.initialize_opinions(opinions_initial)

# fraction of anti-conformists and their strength
model_anti_system = ModifiedVoterModel(
    G, propaganda=0.0, ideology_std=0.0, anti_fraction=0.2, anti_strength=2.0
)
model_anti_system.initialize_opinions(opinions_initial)

# propaganda and anti-conformists
model_anti_propaganda = ModifiedVoterModel(
    G, propaganda=2.0, ideology_std=0.0, anti_fraction=0.2, anti_strength=2.0
)
model_anti_propaganda.initialize_opinions(opinions_initial)

# simulations
history_classic = []
history_propaganda = []
history_ideology = []
history_both = []
history_anti = []
history_prop_anti = []

for _ in range(steps):
    model_classic.step()
    history_classic.append(model_classic.get_average_opinion())

    model_propaganda.step()
    history_propaganda.append(model_propaganda.get_average_opinion())

    model_ideology.step()
    history_ideology.append(model_ideology.get_average_opinion())

    model_both.step()
    history_both.append(model_both.get_average_opinion())

    model_anti_system.step()
    history_anti.append(model_anti_system.get_average_opinion())

    model_anti_propaganda.step()
    history_prop_anti.append(model_anti_propaganda.get_average_opinion())


plt.figure(figsize=(10, 6))
plt.plot(history_classic, label="classical", alpha=0.7)
plt.plot(history_propaganda, label="with propaganda", alpha=0.7)
plt.plot(history_ideology, label="with ideology", alpha=0.7)
plt.plot(history_both, label="with propaganda and ideology", alpha=0.7)
plt.plot(history_anti, label="with anti-conformists", alpha=0.7)
plt.plot(history_prop_anti, label="with propaganda and anti-conformists", alpha=0.7)

plt.title("Modified Voter Model: Effects of Propaganda and Ideology")
plt.xlabel("Time Steps")
plt.ylabel("Average Opinion (-1 to +1)")
plt.axhline(0, color="black", linestyle="--", alpha=0.3)
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim(-1, 1)
plt.savefig("modified_voter_model.png")
plt.show()
