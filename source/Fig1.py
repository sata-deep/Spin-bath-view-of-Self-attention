import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import matplotlib as mpl
mpl.rcParams['axes.linewidth'] = 1.5

# --- 1. LOAD AND PREPARE DATA ---

# Load the collected data
try:
    df = pd.read_csv('head_analysis_results.csv')
    print("Successfully loaded 'head_analysis_results.csv'")
except FileNotFoundError:
    print("Error: 'head_analysis_results.csv' not found. Please run step1_collect_data.py first.")
    exit()

# Define our hero head
HERO_LAYER = 3
HERO_HEAD = 5

# Filter the DataFrame for our hero head
hero_df = df[(df['layer'] == HERO_LAYER) & (df['head'] == HERO_HEAD)].copy()

print(f"\nAnalyzing data for Hero Head: Layer {HERO_LAYER}, Head {HERO_HEAD}")
print(f"Number of data points (prompts): {len(hero_df)}")

# --- 2. PERFORM STATISTICAL ANALYSIS ---

# Extract the two columns we want to compare
theory_values = hero_df['delta_l_theory']
actual_values = hero_df['delta_l_actual']

# Perform linear regression
slope, intercept, r_value, p_value, std_err = stats.linregress(theory_values, actual_values)
r_squared = r_value**2

print("\n--- Statistical Results ---")
print(f"Pearson Correlation Coefficient (r): {r_value:.4f}")
print(f"Coefficient of Determination (r^2): {r_squared:.4f}")
print(f"P-value: {p_value:.2e}") # Using scientific notation for small p-values

if p_value < 0.05:
    print("The correlation is statistically significant (p < 0.05).")
else:
    print("The correlation is not statistically significant (p >= 0.05).")

# --- 3. CREATE THE SCATTER PLOT (NEW FIGURE 3) ---

print("\nGenerating scatter plot...")

# Set a professional plot style
#sns.set_theme(style="whitegrid")
plt.figure(figsize=(8, 6))

# Create the scatter plot
ax = sns.regplot(
    x=theory_values,
    y=actual_values,
    scatter_kws={'alpha': 0.7, 's': 100, 'edgecolor': 'b'}, # Style scatter points
    line_kws={'color': 'red', 'linewidth': 2.5} # Style regression line
)

# Add annotations for the statistical results
# We place the text in the corner of the plot
text_str = f'$r^2 = {r_squared:.3f}$\n$p < {0.001 if p_value < 0.001 else f"{p_value:.3f}"}$'
ax.text(0.05, 0.95, text_str, transform=ax.transAxes, fontsize=14,
        verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))

# Set labels and title
ax.set_xlabel('Theoretical Logit Difference ($\Delta L_{theory}$)', fontsize=14)
ax.set_ylabel('Actual Model Logit Difference ($\Delta L_{actual}$)', fontsize=14)
ax.set_title(f'Correlation for Head L{HERO_LAYER}H{HERO_HEAD} across 20 Prompts', fontsize=14, weight='bold')

# Ensure plot is well-laid out
plt.tight_layout()

# Save the figure
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
output_filename = 'Fig1.png'
plt.savefig(output_filename, dpi=300)
print(f"Plot saved as '{output_filename}'")
plt.show()

# --- 4. (For Discussion) Analyze Overall Head Performance ---
# This part helps confirm L11H4 is a good choice

# Calculate the correlation for every single head
correlations = df.groupby(['layer', 'head']).apply(
    lambda g: stats.linregress(g['delta_l_theory'], g['delta_l_actual']).rvalue
).rename('correlation').reset_index()

# Sort by the absolute value of the correlation to find the most predictive heads
correlations['abs_correlation'] = correlations['correlation'].abs()
sorted_correlations = correlations.sort_values(by='abs_correlation', ascending=False)

print("\n--- Top 5 Most Predictive Heads (by absolute correlation) ---")
print(sorted_correlations.head(5))
