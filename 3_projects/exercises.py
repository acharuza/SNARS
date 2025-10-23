import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import networkx as nx


def exercise1(mu=0, sigma=1):
    data_generator = lambda n: np.random.normal(mu, sigma, n)
    histogram_with_sigma_intervals(data_generator, mu, sigma)

def exercise2(p=0.3):
    data_generator = lambda n: np.random.geometric(p, n)
    check_pareto_rule(data_generator)

def exercise3(x_min, alpha):
    data_generator = lambda n: x_min * (1 + np.random.pareto(alpha - 1, n))
    mean = (alpha - 1) * x_min / (alpha - 2) if alpha > 2 else float('inf')
    sigma = (x_min**2 * (alpha - 1) / ((alpha - 3) * (alpha - 2)**2))**0.5 if alpha > 3 else float('inf')
    histogram_with_sigma_intervals(data_generator, mean, sigma)
    
def exercise4(a=1.5):
    data_generator = lambda n: np.random.zipf(a, n)
    check_pareto_rule(data_generator)

def exercise5(n=1000, m=3, p=0.01):
    ba_graph = nx.barabasi_albert_graph(n, m)
    er_graph = nx.erdos_renyi_graph(n, p)

    # calculate degrees
    ba_degrees = [deg for node, deg in ba_graph.degree()]
    er_degrees = [deg for node, deg in er_graph.degree()]

    # estimate expected degree and variance for both networks
    ba_mean = np.mean(ba_degrees)
    ba_var = np.var(ba_degrees)
    er_mean = np.mean(er_degrees)
    er_var = np.var(er_degrees)

    fig, axs = plt.subplots(2, 2, figsize=(14, 10), 
                           gridspec_kw={'height_ratios': [1, 3]})

    # BA degree histogram
    axs[0, 0].hist(ba_degrees, bins=len(np.unique(ba_degrees)), rwidth=1, color='skyblue', edgecolor='black')
    axs[0, 0].set_title("BA Network Degree Histogram")
    axs[0, 0].set_xlabel("Degree")
    axs[0, 0].set_ylabel("Number of nodes")
    textstr = f"Mean = {ba_mean:.2f}\nVariance = {ba_var:.2f}"
    axs[0, 0].text(0.95, 0.95, textstr, transform=axs[0, 0].transAxes,
                   fontsize=12, verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

    # ER degree histogram
    axs[0, 1].hist(er_degrees, bins=len(np.unique(er_degrees)), rwidth=1, color='salmon', edgecolor='black')
    axs[0, 1].set_title("ER Graph Degree Histogram")
    axs[0, 1].set_xlabel("Degree")
    axs[0, 1].set_ylabel("Number of nodes")
    textstr = f"Mean = {er_mean:.2f}\nVariance = {er_var:.2f}"
    axs[0, 1].text(0.95, 0.95, textstr, transform=axs[0, 1].transAxes,
                   fontsize=12, verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

    # BA network visualization
    if n < 200:
        pos_ba = nx.kamada_kawai_layout(ba_graph, scale=500)
    else:
        pos_ba = nx.spring_layout(ba_graph, k=2, iterations=50, seed=42)
    
    # node scaling
    ba_node_sizes = [5 + deg * 2 for deg in ba_degrees]
    
    nx.draw(ba_graph, pos=pos_ba, 
            node_size=ba_node_sizes,
            node_color='skyblue', 
            edge_color='gray', 
            width=0.3,
            alpha=0.6,
            linewidths=0,
            ax=axs[1, 0])
    axs[1, 0].set_title(f"BA Network Visualization (n={n}, m={m})")

    # ER network visualization
    if n < 200:
        pos_er = nx.kamada_kawai_layout(er_graph, scale=500)
    else:
        pos_er = nx.spring_layout(er_graph, k=2, iterations=50, seed=42)
    
    # node scaling
    er_node_sizes = [5 + deg * 2 for deg in er_degrees]
    
    nx.draw(er_graph, pos=pos_er, 
            node_size=er_node_sizes,
            node_color='salmon', 
            edge_color='gray', 
            width=0.3,
            alpha=0.6,
            linewidths=0,
            ax=axs[1, 1])
    axs[1, 1].set_title(f"ER Network Visualization (n={n}, p={p})")

    plt.tight_layout()
    plt.show()

def histogram_with_sigma_intervals(data_generator, mean, sigma):
    data = data_generator(1000)
    plt.hist(data, bins=30, density=True, alpha=0.6, color='g')
    plt.axvline(mean, color='k', linestyle='dashed', linewidth=1, label='Mean')
    
    # only sigma intervals within data range
    data_min, data_max = np.min(data), np.max(data)
    
    ticks = [mean]
    left_count = 0  # how many we've drawn
    
    for i in range(1, int(np.ceil(np.max(np.abs(data)))) + 1):
        # right side
        right_line = mean + i * sigma
        if right_line <= data_max:
            plt.axvline(right_line, color='r', linestyle='dashed', linewidth=1, alpha=0.7)
            ticks.append(right_line)
        
        # left side
        left_line = mean - i * sigma

        # one left interval, even if outside data range
        if (left_line >= data_min and left_line < mean) or (left_count == 0 and i == 1):
            plt.axvline(left_line, color='b', linestyle='dashed', linewidth=1, alpha=0.7)
            ticks.append(left_line)
            left_count += 1
        elif left_line >= data_min and left_line < mean:
            plt.axvline(left_line, color='b', linestyle='dashed', linewidth=1, alpha=0.7)
            ticks.append(left_line)
            left_count += 1
    
    plt.title('Histogram with Sigma Intervals')
    plt.xlabel('Value')
    plt.ylabel('Density')
    
    ticks = sorted(ticks)
    plt.xticks(ticks, [f'{t:.2f}' for t in ticks], rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()


def check_pareto_rule(data_generator, n=1000, simulations=500, alpha=0.05, equivalence_margin=0.05):
    """
    alpha: significance level (default 0.10 for more lenient test)
    equivalence_margin: acceptable deviation from 0.8 (default ±0.05, i.e., 0.75-0.85)
    """
    
    proportions = []

    for _ in range(simulations):
        data = data_generator(n)
        
        sorted_data = np.sort(data)[::-1]
        top_20_count = int(0.2 * n)
        top_20_sum = np.sum(sorted_data[:top_20_count])
        total_sum = np.sum(sorted_data)
            
        proportion = top_20_sum / total_sum
        proportions.append(proportion)

    mean_proportion = np.mean(proportions)
    std_proportion = np.std(proportions, ddof=1)
    se_proportion = std_proportion / np.sqrt(len(proportions))
    ci_95 = 1.96 * se_proportion
    
    # tests if proportion is within [0.8 - margin, 0.8 + margin]
    lower_bound = 0.8 - equivalence_margin
    upper_bound = 0.8 + equivalence_margin
    
    # test 1: Is mean > lower_bound? (H1: mean > lower_bound)
    t_lower = (mean_proportion - lower_bound) / se_proportion
    p_lower = 1 - stats.t.cdf(t_lower, df=len(proportions)-1)
    
    # test 2: Is mean < upper_bound? (H1: mean < upper_bound)
    t_upper = (mean_proportion - upper_bound) / se_proportion
    p_upper = stats.t.cdf(t_upper, df=len(proportions)-1)
    
    # both tests must be significant
    p_equivalence = max(p_lower, p_upper)
    is_equivalent = (p_lower < alpha) and (p_upper < alpha)

    print("Pareto Rule Analysis (TOST Equivalence Test):")
    print(f"- Based on {len(proportions)} simulations")
    print(f"- Top 20% accounts for: {mean_proportion*100:.2f}% ± {ci_95*100:.2f}% (95% CI)")
    print(f"- Standard deviation: {std_proportion*100:.2f}%")
    
    print(f"\n--- TOST Equivalence Test (α={alpha}) ---")
    print(f"Equivalence range: [{lower_bound:.2f}, {upper_bound:.2f}]")
    print(f"  Test 1 - H1: mean > {lower_bound:.2f}")
    print(f"    t = {t_lower:.4f}, p = {p_lower:.6f} {'✓' if p_lower < alpha else '✗'}")
    print(f"  Test 2 - H1: mean < {upper_bound:.2f}")
    print(f"    t = {t_upper:.4f}, p = {p_upper:.6f} {'✓' if p_upper < alpha else '✗'}")
    print(f"  Combined p-value: {p_equivalence:.6f}")
    
    if is_equivalent:
        print(f"\n✓ EQUIVALENCE DEMONSTRATED")
        print(f"  Proportion is statistically equivalent to 0.8")
        print(f"  → PARETO PRINCIPLE HOLDS")
    else:
        print(f"\n✗ EQUIVALENCE NOT DEMONSTRATED")
        print(f"  Proportion is not close enough to 0.8")
        print(f"  → PARETO PRINCIPLE DOES NOT HOLD")
    

if __name__ == "__main__":
    exercise5(n=500, m=2, p=0.01)


