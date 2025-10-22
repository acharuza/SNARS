import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def exercise1(mu=0, sigma=1):
    data_generator = lambda n: np.random.normal(mu, sigma, n)
    histogram_with_sigma_intervals(data_generator, mu, sigma)

def exercise2(p=0.3):
    data_generator = lambda n: np.random.geometric(p, n)
    check_pareto_rule(data_generator, simulations=100)

def exercise3(x_min, alpha):
    data_generator = lambda n: x_min * (1 + np.random.pareto(alpha - 1, n))
    mean = (alpha - 1) * x_min / (alpha - 2) if alpha > 2 else float('inf')
    sigma = (x_min**2 * (alpha - 1) / ((alpha - 3) * (alpha - 2)**2))**0.5 if alpha > 3 else float('inf')
    histogram_with_sigma_intervals(data_generator, mean, sigma)
    

def exercise4(a=1.5):
    data_generator = lambda n: np.random.zipf(a, n)
    check_pareto_rule(data_generator, simulations=100)

def exercise5():
    pass


def histogram_with_sigma_intervals(data_generator, mean, sigma):
    data = data_generator(1000)
    plt.hist(data, 30, density=True, alpha=0.6, color='g')
    plt.axvline(mean, color='k', linestyle='dashed', linewidth=1, label='Mean')
    
    # Only show sigma intervals that are within the data range
    data_min, data_max = np.min(data), np.max(data)
    
    ticks = [mean]
    left_count = 0  # Track how many left intervals we've drawn
    
    for i in range(1, 4):
        # Right side (mean + i*sigma)
        right_line = mean + i * sigma
        if right_line <= data_max:
            plt.axvline(right_line, color='r', linestyle='dashed', linewidth=1, alpha=0.7)
            ticks.append(right_line)
        
        # Left side (mean - i*sigma)
        left_line = mean - i * sigma
        # Force at least one left interval, even if outside data range
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
    
    # Sort ticks for better readability
    ticks = sorted(ticks)
    plt.xticks(ticks, [f'{t:.2f}' for t in ticks], rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()


def check_pareto_rule(data_generator, n=1000, simulations=100, alpha=0.05, equivalence_margin=0.05):
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
    
    # TOST (Two One-Sided Tests) for equivalence testing
    # Tests if proportion is within [0.8 - margin, 0.8 + margin]
    lower_bound = 0.8 - equivalence_margin
    upper_bound = 0.8 + equivalence_margin
    
    # Test 1: Is mean > lower_bound? (H1: mean > lower_bound)
    t_lower = (mean_proportion - lower_bound) / se_proportion
    p_lower = 1 - stats.t.cdf(t_lower, df=len(proportions)-1)
    
    # Test 2: Is mean < upper_bound? (H1: mean < upper_bound)
    t_upper = (mean_proportion - upper_bound) / se_proportion
    p_upper = stats.t.cdf(t_upper, df=len(proportions)-1)
    
    # For equivalence, both tests must be significant
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
    exercise4(a=2)


