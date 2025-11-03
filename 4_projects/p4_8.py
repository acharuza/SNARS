import requests
from bs4 import BeautifulSoup
import re
import pandas as pd
import matplotlib.pyplot as plt


def parse_erdoes_page(url):
    r = requests.get(url)
    soup = BeautifulSoup(r.text, "html.parser")
    paragraphs = soup.find_all("p")

    data = []
    for p in paragraphs:
        text = p.get_text(strip=True)
        if not text or text.startswith("NAME"):
            continue
        parts = re.split(r"\s{2,}", text)
        if len(parts) >= 4:
            name = parts[0].strip()
            year = parts[1].strip()
            subject = parts[2].strip()
            try:
                erdoes_number = int(re.findall(r"\d+", parts[3])[-1])
            except (IndexError, ValueError):
                continue
            data.append((name, year, subject, erdoes_number))

    return pd.DataFrame(data, columns=["NAME", "YEAR", "SUBJECT", "ERDÖS NUMBER"])


# URLs
url_nobel = "https://sites.google.com/oakland.edu/grossman/home/the-erdoes-number-project/some-famous-people-with-finite-erdoes-numbers/nobel-prize-winners"
url_fields = "https://sites.google.com/oakland.edu/grossman/home/the-erdoes-number-project/some-famous-people-with-finite-erdoes-numbers/fields-medal-winners"

# Parse both pages
df_nobel = parse_erdoes_page(url_nobel)
df_fields = parse_erdoes_page(url_fields)

# Plot histograms
plt.figure(figsize=(10, 5))
plt.hist(
    df_nobel["ERDÖS NUMBER"],
    bins=range(df_nobel["ERDÖS NUMBER"].min(), df_nobel["ERDÖS NUMBER"].max() + 1),
    alpha=0.6,
    label="Nobel Prize",
    edgecolor="black",
)
plt.hist(
    df_fields["ERDÖS NUMBER"],
    bins=range(df_fields["ERDÖS NUMBER"].min(), df_fields["ERDÖS NUMBER"].max() + 1),
    alpha=0.6,
    label="Fields Medal",
    edgecolor="black",
)
plt.xlabel("Erdős Number")
plt.ylabel("Count")
plt.title("Erdős Number Distribution: Nobel Prize vs Fields Medal Laureates")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("erdos_number_distribution.png", dpi=300)
