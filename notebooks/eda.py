"""
eda.py — Exploratory Data Analysis for the HDI dataset.
Generates the visualizations referenced in the project brief and saves
them to static/images so the Flask app's "Insights" page can display them.
"""

from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "hdi_dataset.csv"
OUT = BASE_DIR / "static" / "images"
OUT.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", font_scale=1.05)
PALETTE = {"Very High": "#C9A24B", "High": "#2E7D74", "Medium": "#5B7DB1", "Low": "#B4544A"}
ORDER = ["Low", "Medium", "High", "Very High"]

def classify_hdi(score: float) -> str:
    """Classify HDI score into development tier."""
    if score >= 0.800:
        return "Very High"
    elif score >= 0.700:
        return "High"
    elif score >= 0.550:
        return "Medium"
    else:
        return "Low"

df = pd.read_csv(DATA_PATH)
df["hdi_category"] = df["hdi"].apply(classify_hdi)
df["hdi_category"] = pd.Categorical(df["hdi_category"], categories=ORDER, ordered=True)

# 1. Distribution of HDI score
plt.figure(figsize=(7, 4.5))
sns.histplot(df["hdi"].dropna(), bins=25, kde=True, color="#2E7D74")
plt.title("Distribution of HDI Scores")
plt.xlabel("HDI Score")
plt.tight_layout()
plt.savefig(f"{OUT}/dist_hdi.png", dpi=140)
plt.close()

# 2. Strip plot: Electricity access by HDI category
plt.figure(figsize=(7, 4.5))
sns.stripplot(data=df, x="hdi_category", y="electricity_access", order=ORDER,
              hue="hdi_category", palette=PALETTE, legend=False, alpha=0.7, jitter=0.25)
sns.despine()
plt.title("Electricity Access by Development Tier")
plt.xlabel("HDI Category")
plt.ylabel("Electricity Access (% of population)")
plt.tight_layout()
plt.savefig(f"{OUT}/strip_electricity_access.png", dpi=140)
plt.close()

# 3. Strip plot: Internet users by HDI category
plt.figure(figsize=(7, 4.5))
sns.stripplot(data=df, x="hdi_category", y="internet_users", order=ORDER,
              hue="hdi_category", palette=PALETTE, legend=False, alpha=0.7, jitter=0.25)
plt.title("Internet Users by Development Tier")
plt.xlabel("HDI Category")
plt.ylabel("Internet Users (% of population)")
plt.tight_layout()
plt.savefig(f"{OUT}/strip_internet_users.png", dpi=140)
plt.close()

# 4. Correlation heatmap
plt.figure(figsize=(8, 7))
num_cols = [
    "electricity_access",
    "internet_users",
    "water_access",
    "health_exp_gdp",
    "female_labor_force",
    "unemployment",
    "infant_mortality",
    "fertility_rate",
    "urban_pop",
    "co2_per_capita",
    "hdi",
]
corr = df[num_cols].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="YlGnBu", square=True, cbar_kws={"shrink": .8})
plt.title("Correlation Matrix of HDI Indicators")
plt.tight_layout()
plt.savefig(f"{OUT}/heatmap_corr.png", dpi=140)
plt.close()

# 5. Scatter: Water access vs HDI
plt.figure(figsize=(7, 4.5))
sns.scatterplot(data=df, x="water_access", y="hdi", hue="hdi_category",
                 hue_order=ORDER, palette=PALETTE, alpha=0.8)
plt.title("Water Access vs HDI Score")
plt.xlabel("Water Access (% of population)")
plt.ylabel("HDI Score")
plt.legend(title="Category")
plt.tight_layout()
plt.savefig(f"{OUT}/scatter_water_hdi.png", dpi=140)
plt.close()

# 6. Scatter: Infant mortality vs HDI
plt.figure(figsize=(7, 4.5))
sns.scatterplot(data=df, x="infant_mortality", y="hdi", hue="hdi_category",
                 hue_order=ORDER, palette=PALETTE, alpha=0.8)
plt.title("Infant Mortality vs HDI Score")
plt.xlabel("Infant Mortality Rate (per 1,000 live births)")
plt.ylabel("HDI Score")
plt.legend(title="Category")
plt.tight_layout()
plt.savefig(f"{OUT}/scatter_mortality_hdi.png", dpi=140)
plt.close()

# 7. Strip plot: Fertility rate by HDI category
plt.figure(figsize=(7, 4.5))
sns.stripplot(data=df, x="hdi_category", y="fertility_rate", order=ORDER,
              hue="hdi_category", palette=PALETTE, legend=False, alpha=0.7, jitter=0.25)
plt.title("Fertility Rate by Development Tier")
plt.xlabel("HDI Category")
plt.ylabel("Fertility Rate (births per woman)")
plt.tight_layout()
plt.savefig(f"{OUT}/strip_fertility_rate.png", dpi=140)
plt.close()

# 8. Strip plot: Unemployment by HDI category
plt.figure(figsize=(7, 4.5))
sns.stripplot(data=df, x="hdi_category", y="unemployment", order=ORDER,
              hue="hdi_category", palette=PALETTE, legend=False, alpha=0.7, jitter=0.25)
plt.title("Unemployment Rate by Development Tier")
plt.xlabel("HDI Category")
plt.ylabel("Unemployment (% of labor force)")
plt.tight_layout()
plt.savefig(f"{OUT}/strip_unemployment.png", dpi=140)
plt.close()

# 9. Scatter: Health expenditure vs HDI
plt.figure(figsize=(7, 4.5))
sns.scatterplot(data=df, x="health_exp_gdp", y="hdi", hue="hdi_category",
                 hue_order=ORDER, palette=PALETTE, alpha=0.8)
plt.title("Health Expenditure vs HDI Score")
plt.xlabel("Health Expenditure (% of GDP)")
plt.ylabel("HDI Score")
plt.legend(title="Category")
plt.tight_layout()
plt.savefig(f"{OUT}/scatter_health_exp_hdi.png", dpi=140)
plt.close()

# 10. Scatter: Urban population vs HDI
plt.figure(figsize=(7, 4.5))
sns.scatterplot(data=df, x="urban_pop", y="hdi", hue="hdi_category",
                 hue_order=ORDER, palette=PALETTE, alpha=0.8)
plt.title("Urban Population vs HDI Score")
plt.xlabel("Urban Population (% of total)")
plt.ylabel("HDI Score")
plt.legend(title="Category")
plt.tight_layout()
plt.savefig(f"{OUT}/scatter_urban_pop_hdi.png", dpi=140)
plt.close()

print("EDA figures saved to", OUT)
