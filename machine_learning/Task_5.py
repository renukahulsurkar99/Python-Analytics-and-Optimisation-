from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_config import DATA_PATH, QUICK_TEST, choose_threshold, capture_plots
#TASK 05: INVESTIGATION OF POTENTIAL FACTORES CONTRIBUTING TO BREACHES & PROLONGED STAYS

#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
capture_plots(__file__)
import seaborn as sns
from scipy.stats import pearsonr

# Set style
sns.set(style="whitegrid")

# Load dataset
df = pd.read_csv(DATA_PATH)

# Set seed and sample 400 patients
np.random.seed(9999)
sample = df.sample(n=400, random_state=9999)

# Convert breach to binary (ensure correct type)
# Standardise text (safety step)
sample['Breachornot'] = sample['Breachornot'].str.strip().str.lower()

# Map to binary
sample['Breach_binary'] = sample['Breachornot'].map({
    'non-breach': 0,
    'breach': 1
})

# Check mapping
print(sample['Breach_binary'].value_counts())

# Convert DayofWeek to categorical order
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday',
             'Friday', 'Saturday', 'Sunday']
sample['DayofWeek'] = pd.Categorical(sample['DayofWeek'], categories=day_order, ordered=True)

# Overall statistics
print("="*70)
print("TASK 5: OVERALL STATISTICS")
print("="*70)
print()
summary_stats = sample[['Age', 'LoS', 'noofinvestigation',
                        'nooftreatment', 'noofpatients']].describe()

print(summary_stats)

# Breach rate
print(summary_stats)

print("Breach rate:", sample['Breach_binary'].mean())

#Correlation between LoS and Breach (outcome variables)
print("="*70)
print("COORELATION BETWEEN LoS AND BREACH")
print("="*70)
print()
corr_los_breach, p_los_breach = pearsonr(
    sample['LoS'],
    sample['Breach_binary']
)

print(f"Correlation between LoS and Breach: {corr_los_breach:.3f}")
print(f"P-value: {p_los_breach:.5f}")

plt.figure()
sns.boxplot(x='Breach_binary', y='LoS', data=sample)
plt.xticks([0,1], ['Non-breach', 'Breach'])
plt.title("Length of Stay by Breach Status")
plt.ylabel("Length of Stay (minutes)")
plt.xlabel("")
plt.show()

#Factors Associated with Length of Stay
#add variables for correlation
print("="*70)
print("INVESTIGATING BREACH FACTORS ASSOCIATED WITH LENGTH OF STAY")
print("="*70)
print()

los_corr_vars = [
    'Age',
    'Day',
    'Period',
    'noofinvestigation',
    'nooftreatment',
    'noofpatients'
]

corr_results = []

for var in los_corr_vars:
    corr_coef, p_value = pearsonr(sample['LoS'], sample[var])
    corr_results.append({
        'Variable': var,
        'Correlation_with_LoS': corr_coef,
        'P_value': p_value
    })

corr_df = pd.DataFrame(corr_results)

# Add significance labels
corr_df['Significance'] = corr_df['P_value'].apply(
    lambda p: '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.1 else 'Not significant'
)

print(corr_df)
print()

plt.figure(figsize=(8,5))
sns.barplot(
    x='Correlation_with_LoS',
    y='Variable',
    data=corr_df,
    orient='h'
)
plt.axvline(0, linestyle='--')
plt.title("Correlation of Key Factors with Length of Stay")
plt.xlabel("Pearson Correlation Coefficient")
plt.ylabel("")
plt.show()
