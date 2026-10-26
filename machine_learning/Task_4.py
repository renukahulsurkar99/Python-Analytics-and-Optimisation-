from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_config import LOCAL_OUTPUT_DIR, DATA_PATH, QUICK_TEST, choose_threshold, capture_plots
#TASK 04: RANDOM SAMPLE SELECTION, MAIN FEATURES OF INDIVIDUAL VARIABLES & REALTIONSHIP BETWEEN VARIABLES

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
capture_plots(__file__)
import seaborn as sns

# Set random seed for reproducibility
RANDOM_SEED = 9999
np.random.seed(RANDOM_SEED)

# =============================================================================
# STEP 1: Load data and create sample
# =============================================================================
full_data = pd.read_csv(DATA_PATH)
print(f"Total patients in dataset: {len(full_data)}")

# Create random sample of 400 patients
sample_data = full_data.sample(n=400, random_state=RANDOM_SEED)
print(f"Sample created: {len(sample_data)} patients\n")

# =============================================================================
# STEP 2: Describe individual variables
# =============================================================================
print("="*70)
print("DESCRIPTIVE STATISTICS")
print("="*70)

# Numerical variables
print("\n1. NUMERICAL VARIABLES:")
print("-"*70)
numerical_cols = ['Age', 'Day', 'Period', 'LoS', 'noofinvestigation',
                  'nooftreatment', 'noofpatients']

stats = sample_data[numerical_cols].describe()
print(stats.round(2))

# Categorical variables
print("\n2. CATEGORICAL VARIABLES:")
print("-"*70)

# Breach status
print("\nBreach Status:")
breach_counts = sample_data['Breachornot'].value_counts()
for status, count in breach_counts.items():
    percentage = (count / len(sample_data)) * 100
    print(f"  {status}: {count} patients ({percentage:.1f}%)")

# Day of week
print("\nDay of Week:")
dow_counts = sample_data['DayofWeek'].value_counts()
for day, count in dow_counts.items():
    print(f"  {day}: {count} patients")

# Top 10 HRG codes
print("\nTop 10 Health Related Groups (HRG):")
hrg_counts = sample_data['HRG'].value_counts().head(10)
for hrg, count in hrg_counts.items():
    print(f"  {hrg}: {count} patients")

# =============================================================================
# STEP 3: Key summary insights
# =============================================================================
print("\n3. KEY SUMMARY INSIGHTS:")
print("-"*70)

breach_rate = (sample_data['Breachornot'] == 'breach').sum() / len(sample_data) * 100
avg_los = sample_data['LoS'].mean()
avg_age = sample_data['Age'].mean()
avg_investigations = sample_data['noofinvestigation'].mean()
avg_treatments = sample_data['nooftreatment'].mean()
avg_occupancy = sample_data['noofpatients'].mean()

print(f"• Breach Rate: {breach_rate:.1f}% (target is <5%)")
print(f"• Average Length of Stay: {avg_los:.0f} minutes ({avg_los/60:.1f} hours)")
print(f"• Average Patient Age: {avg_age:.1f} years")
print(f"• Average Investigations per Patient: {avg_investigations:.2f}")
print(f"• Average Treatments per Patient: {avg_treatments:.2f}")
print(f"• Average AED Occupancy on Arrival: {avg_occupancy:.1f} patients")

# =============================================================================
# STEP 4: Visualizations of individual variables
# =============================================================================
print("\nGenerating visualizations...")

# Create a figure with 9 subplots
fig = plt.figure(figsize=(18, 15))
fig.suptitle('AED Patient Sample - Variable Distributions', fontsize=16, fontweight='bold')

# 1. Age distribution
plt.subplot(3, 3, 1)
plt.hist(sample_data['Age'], bins=20, color='skyblue', edgecolor='black')
plt.axvline(avg_age, color='red', linestyle='--', linewidth=2, label=f'Mean: {avg_age:.1f}')
plt.xlabel('Age (years)')
plt.ylabel('Number of Patients')
plt.title('Age Distribution')
plt.legend()

# 2. Length of Stay distribution
plt.figure(figsize=(15, 10), dpi=120)

plt.subplot(3, 3, 2)
plt.hist(sample_data['LoS'], bins=30, color='lightcoral', edgecolor='black')
plt.axvline(240, color='red', linestyle='--', linewidth=2, label='4-hour target')

plt.xlabel('Length of Stay (minutes)', fontsize=12)
plt.ylabel('Number of Patients', fontsize=12)
plt.title('Length of Stay Distribution', fontsize=14)
plt.legend(fontsize=11)

plt.tight_layout()  # Prevents overlapping
plt.show()

# 3. Breach status
plt.figure(figsize=(18, 15), dpi=120)

plt.subplot(3, 3, 3)
breach_counts = sample_data['Breachornot'].value_counts()

colors = ['#2ecc71', '#e74c3c']

plt.bar(breach_counts.index, breach_counts.values,
        color=colors, edgecolor='black')

plt.xlabel('Breach Status', fontsize=12)
plt.ylabel('Number of Patients', fontsize=12)
plt.title('Breach Status Distribution', fontsize=14)

for i, count in enumerate(breach_counts.values):
    plt.text(i, count + max(breach_counts.values)*0.02,
             f'{count}\n({count/len(sample_data)*100:.1f}%)',
             ha='center', fontweight='bold')


plt.tight_layout()
plt.show()

# 4. Day of week
plt.figure(figsize=(15, 10), dpi=120)

plt.subplot(3, 3, 4)
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
dow_counts = sample_data['DayofWeek'].value_counts().reindex(day_order, fill_value=0)
plt.bar(range(len(dow_counts)), dow_counts.values, color='mediumpurple', edgecolor='black')
plt.xticks(range(len(dow_counts)), dow_counts.index, rotation=45, ha='right')
plt.ylabel('Number of Patients')
plt.title('Patient Arrivals by Day of Week')

# 5. Arrival time (Period)
arrival_counts = sample_data['Period'].value_counts().sort_index()

plt.figure(figsize=(18, 15), dpi=120)
plt.subplot(3, 3, 5)

plt.bar(arrival_counts.index,
        arrival_counts.values,
        color='#f39c12',
        edgecolor='black')

plt.xlabel('Hour of Day (0 = Midnight, 23 = 11 PM)', fontsize=12)
plt.ylabel('Number of Arrivals', fontsize=12)
plt.title('Arrival Time Distribution', fontsize=14)

plt.xticks(range(0, 24, 2))
plt.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()

# 6. Number of investigations
plt.figure(figsize=(15, 10), dpi=120)

plt.subplot(3, 3, 6)
max_investigations = sample_data['noofinvestigation'].max()
plt.hist(sample_data['noofinvestigation'],
         bins=range(0, max_investigations + 2),
         color='lightblue', edgecolor='black')
plt.xlabel('Number of Investigations')
plt.ylabel('Number of Patients')
plt.title('Investigations per Patient')

# 7. Number of treatments
plt.figure(figsize=(15, 10), dpi=120)

plt.subplot(3, 3, 7)
max_treatments = sample_data['nooftreatment'].max()
plt.hist(sample_data['nooftreatment'],
         bins=range(0, max_treatments + 2),
         color='lightgreen', edgecolor='black')
plt.xlabel('Number of Treatments')
plt.ylabel('Number of Patients')
plt.title('Treatments per Patient')

# 8. AED occupancy
plt.figure(figsize=(15, 10), dpi=120)

plt.subplot(3, 3, 8)
plt.hist(sample_data['noofpatients'], bins=20, color='pink', edgecolor='black')
plt.xlabel('Number of Patients Already in AED')
plt.ylabel('Number of Arrivals')
plt.title('AED Occupancy on Arrival')

plt.tight_layout()
plt.show()

# 9. Top 10 HRG codes
plt.figure(figsize=(15, 10), dpi=120)

plt.subplot(3, 3, 9)
top_hrg = sample_data['HRG'].value_counts().head(10)
plt.barh(range(len(top_hrg)), top_hrg.values, color='teal', edgecolor='black')
plt.yticks(range(len(top_hrg)), top_hrg.index, fontsize=9)
plt.xlabel('Number of Patients')
plt.title('Top 10 Health Related Groups')
plt.gca().invert_yaxis()

plt.tight_layout()
plt.savefig(LOCAL_OUTPUT_DIR / 'task4_distributions.png', dpi=300, bbox_inches='tight')
print("Saved: task4_distributions.png")

plt.show()

# =============================================================================
# STEP 5: Relationships between variables
# =============================================================================
print("\n" + "="*70)
print("RELATIONSHIPS BETWEEN VARIABLES")
print("="*70)

# Correlation analysis
print("\n1. CORRELATION WITH LENGTH OF STAY:")
print("-"*70)
correlations = sample_data[numerical_cols].corr()['LoS'].sort_values(ascending=False)
for var in numerical_cols:
    if var != 'LoS':
        corr_value = correlations[var]
        strength = "strong" if abs(corr_value) > 0.5 else "moderate" if abs(corr_value) > 0.3 else "weak"
        print(f"  {var}: {corr_value:.3f} ({strength})")

# Compare breach vs non-breach groups
print("\n2. BREACH VS NON-BREACH COMPARISON:")
print("-"*70)
breach_group = sample_data[sample_data['Breachornot'] == 'breach']
nonbreach_group = sample_data[sample_data['Breachornot'] == 'non-breach']

print(f"\n{'Variable':<20} {'Non-Breach':<15} {'Breach':<15} {'Difference':<15}")
print("-"*70)
for var in ['Age', 'LoS', 'noofinvestigation', 'nooftreatment', 'noofpatients']:
    nonbreach_mean = nonbreach_group[var].mean()
    breach_mean = breach_group[var].mean()
    diff = breach_mean - nonbreach_mean
    print(f"{var:<20} {nonbreach_mean:<15.2f} {breach_mean:<15.2f} {diff:<15.2f}")

# Breach rate by day of week
print("\n3. BREACH RATE BY DAY OF WEEK:")
print("-"*70)
for day in day_order:
    day_data = sample_data[sample_data['DayofWeek'] == day]
    if len(day_data) > 0:
        breach_rate_day = (day_data['Breachornot'] == 'breach').sum() / len(day_data) * 100
        print(f"  {day}: {breach_rate_day:.1f}%")

# Create relationship visualizations
fig2 = plt.figure(figsize=(16, 12))
fig2.suptitle('Relationships Between Variables', fontsize=16, fontweight='bold')

# 1. LoS vs Investigations (colored by breach)
plt.subplot(2, 2, 1)
colors = ['red' if x == 'breach' else 'blue' for x in sample_data['Breachornot']]
plt.scatter(sample_data['noofinvestigation'], sample_data['LoS'],
           c=colors, alpha=0.5, s=30)
plt.axhline(240, color='black', linestyle='--', alpha=0.5, label='4-hour target')
plt.xlabel('Number of Investigations')
plt.ylabel('Length of Stay (minutes)')
plt.title('Length of Stay vs Investigations\n(Red=Breach, Blue=Non-Breach)')
plt.legend()

# 2. Box plot: LoS by breach status
plt.subplot(2, 2, 2)
breach_los = sample_data[sample_data['Breachornot'] == 'breach']['LoS']
nonbreach_los = sample_data[sample_data['Breachornot'] == 'non-breach']['LoS']
box_data = [nonbreach_los, breach_los]
box = plt.boxplot(box_data, tick_labels=['Non-Breach', 'Breach'], patch_artist=True)
box['boxes'][0].set_facecolor('lightblue')
box['boxes'][1].set_facecolor('lightcoral')
plt.ylabel('Length of Stay (minutes)')
plt.title('Length of Stay by Breach Status')

# 3. Breach rate by hour of arrival
plt.subplot(2, 2, 3)
breach_by_period = []
periods = []
for period in sorted(sample_data['Period'].unique()):
    period_data = sample_data[sample_data['Period'] == period]
    if len(period_data) > 0:
        breach_rate_period = (period_data['Breachornot'] == 'breach').sum() / len(period_data) * 100
        breach_by_period.append(breach_rate_period)
        periods.append(period)

plt.bar(periods, breach_by_period, color='coral', edgecolor='black')
plt.xlabel('Hour of Day')
plt.ylabel('Breach Rate (%)')
plt.title('Breach Rate by Arrival Time')
plt.axhline(5, color='red', linestyle='--', label='Breach Target Threshold (5%)', linewidth=2)
plt.legend()

# 4. Correlation heatmap
plt.figure(figsize=(15,10))
plt.subplot(2, 2, 4)

# Create binary breach variable
sample_data['Breach_Binary'] = (sample_data['Breachornot'] == 'breach').astype(int)

# Include all relevant numerical variables
corr_vars = ['Age', 'Day', 'Period', 'LoS', 'noofinvestigation',
             'nooftreatment', 'noofpatients', 'Breach_Binary']

corr_matrix = sample_data[corr_vars].corr()

# Use seaborn for cleaner code
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
            vmin=-1, vmax=1, center=0, square=True, linewidths=0.5,
            cbar_kws={"label": "Correlation"})

plt.title('Correlation Matrix')
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.yticks(fontsize=9)
