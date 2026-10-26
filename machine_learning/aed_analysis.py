from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_config import DATA_PATH, QUICK_TEST, choose_threshold, capture_plots
# AED_ANALYSIS

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, pointbiserialr, chi2_contingency


# =========================================================
# Fixed settings
# =========================================================
CSV_PATH_DEFAULT = DATA_PATH
RANDOM_SEED = 9999
SAMPLE_SIZE = 400

# Specify day order for arrangement
DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# Specify only numerical columns
NUMERICAL_COLS = [
    "Age", "Day", "Period", "LoS",
    "noofinvestigation", "nooftreatment", "noofpatients"
]


# =========================================================
# Data loading / sampling
# =========================================================

# Only load data
def load_data(csv_path=CSV_PATH_DEFAULT):
    backup_data = pd.read_csv(csv_path)
    full_data = backup_data.copy()

    return full_data

# Create a sample of data
def make_sample(full_data, n=SAMPLE_SIZE, seed=RANDOM_SEED):
    if n > len(full_data):
        n = len(full_data)
    return full_data.sample(n=n, random_state=seed).copy()

# Load the data and sample
def load_and_sample(csv_path=CSV_PATH_DEFAULT):
    full = load_data(csv_path)
    return make_sample(full)


# =========================================================
# Helper Functions
# =========================================================
def is_numeric(series):
    return pd.api.types.is_numeric_dtype(series)


def standardize_dayofweek(series):
    return pd.Categorical(series, categories=DAY_ORDER, ordered=True)


def apply_dayofweek_order(df, var):
    if var == "DayofWeek":
        df[var] = standardize_dayofweek(df[var])
    return df


def add_breach_binary(df):
    out = df.copy()
    out["Breach_binary"] = (out["Breachornot"] == "breach").astype(int)
    return out


def significance_label(p):
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.1:
        return "*"
    return "Not significant"


# =========================================================
# Frequency and Descriptive Statistics Tables
# =========================================================
def freq_table(df, col, order=None, top_n=None, include_percent=True):
    s = df[col].dropna()
    vc = s.value_counts().head(top_n) if top_n else s.value_counts()

    if order is not None:
        vc = vc.reindex(order).dropna()

    out = vc.rename("count").to_frame()
    if include_percent:
        out["percent"] = (out["count"] / len(df) * 100).round(1)

    return out.reset_index().rename(columns={"index": col})

# Descriptive statistics: Numerical Variables
def numerical_descriptives(sample_data):
    return (
        sample_data[NUMERICAL_COLS]
        .describe()
        .T
        .rename(columns={
            "count": "N",
            "mean": "Mean",
            "std": "SD",
            "min": "Min",
            "25%": "Q1",
            "50%": "Median",
            "75%": "Q3",
            "max": "Max"
        })
        .round(2)
        .reset_index()
        .rename(columns={"index": "Variable"})
    )

# Key summary statistics
def key_summary(sample_data):
    return pd.DataFrame({
        "Metric": [
            "Average Patient Age (years)",
            "Average Investigations per Patient",
            "Average Treatments per Patient",
        ],
        "Value": [
            round(sample_data["Age"].mean(), 1),
            round(sample_data["noofinvestigation"].mean(), 2),
            round(sample_data["nooftreatment"].mean(), 2),
        ]
    })


# =========================================================
# Task 4a: Single variable tables and Distribution plots
# =========================================================

# ---------------------------------------------------------
# User Interface Functions
# ---------------------------------------------------------

# Statistics about a single variable
def single_variable_table(sample_data, var):
    df = sample_data[[var]].dropna().copy()
    df = apply_dayofweek_order(df, var)
    s = df[var]

    if is_numeric(s):
        return pd.DataFrame({
            "Metric": ["N", "Mean", "SD", "Min", "Q1", "Median", "Q3", "Max"],
            "Value": [
                len(s),
                round(s.mean(), 3),
                round(s.std(), 3),
                round(s.min(), 3),
                round(s.quantile(0.25), 3),
                round(s.median(), 3),
                round(s.quantile(0.75), 3),
                round(s.max(), 3),
            ]
        })
    # If the variable is DayofWeek, then apply order
    if var == "DayofWeek":
        return freq_table(df, var, order=DAY_ORDER, include_percent=True)

    return freq_table(df, var, include_percent=True)

# Plot a distribution or bar chart based on whether variable is numerical or categorical
def single_variable_plot(sample_data, var):
    df = sample_data[[var]].dropna().copy()
    df = apply_dayofweek_order(df, var)
    s = df[var]

    # if numerical then plot distribution
    if is_numeric(s):
        fig, ax = plt.subplots()
        ax.hist(s, bins=20, edgecolor="black")
        ax.set_title(f"Distribution of {var}")
        ax.set_xlabel(var)
        ax.set_ylabel("Count")
        fig.tight_layout()
        return fig

    fig, ax = plt.subplots()

    #if variable is DayofWeek then do valuecounts with order
    if var == "DayofWeek":
        counts = pd.Series(s).value_counts().reindex(DAY_ORDER, fill_value=0)
        ax.bar(range(len(counts)), counts.values, edgecolor="black")
        ax.set_xticks(range(len(counts)))
        ax.set_xticklabels(counts.index, rotation=45, ha="right")

    # Else: if it is categorical but not DayofWeek do value counts without a specific order
    else:
        counts = pd.Series(s).value_counts()
        ax.bar(range(len(counts)), counts.values, edgecolor="black")
        ax.set_xticks(range(len(counts)))
        ax.set_xticklabels(counts.index.astype(str), rotation=45, ha="right")

    ax.set_title(f"Distribution of {var}")
    ax.set_xlabel(var)
    ax.set_ylabel("Count")
    fig.tight_layout()
    return fig

# ---------------------------------------------------------
# Task 4a: Custom plots
# ---------------------------------------------------------

# Arrival time (Period) distribution plot
def custom_arrival_time_distribution(sample_data):
    arrival_counts = sample_data['Period'].value_counts().sort_index()

    fig, ax = plt.subplots()

    plt.bar(arrival_counts.index,
        arrival_counts.values,
        color='#f39c12',
        edgecolor='black')

    ax.set_xlabel('Hour of Day (0 = Midnight, 23 = 11 PM)', fontsize=12)
    ax.set_ylabel('Number of Arrivals', fontsize=12)
    ax.set_title('Arrival Time Distribution', fontsize=14)

    ax.set_xticks(range(0, 24, 2))

    fig.tight_layout()
    return fig

# HRG distribution plot
def custom_HRG_distribution(sample_data):
    top_hrg = sample_data['HRG'].value_counts().head(10)

    fig, ax = plt.subplots()

    ax.barh(range(len(top_hrg)), top_hrg.values, color='teal', edgecolor='black')

    ax.set_yticks(range(len(top_hrg)), top_hrg.index, fontsize=9)
    ax.set_xlabel('Number of Patients')
    ax.set_title('Health Related Groups')
    ax.invert_yaxis()

    return fig

# Age distribution plot
def custom_age_distribution(sample_data):
    avg_age = sample_data['Age'].mean()

    fig, ax = plt.subplots()

    ax.hist(sample_data['Age'], bins=20, color='skyblue', edgecolor='black')
    ax.axvline(avg_age, color='red', linestyle='--', linewidth=2, label=f'Mean: {avg_age:.1f}')
    ax.set_xlabel('Age (years)')
    ax.set_ylabel('Number of Patients')
    ax.set_title('Age Distribution')
    ax.legend()

    return fig


# Length of Stay distribution plot
def custom_LoS_distribution(sample_data):

    fig, ax = plt.subplots()

    ax.hist(sample_data['LoS'], bins=30, color='lightcoral', edgecolor='black')
    ax.axvline(240, color='red', linestyle='--', linewidth=2, label='4-hour target')

    ax.set_xlabel('Length of Stay (minutes)', fontsize=12)
    ax.set_ylabel('Number of Patients', fontsize=12)
    ax.set_title('Length of Stay Distribution', fontsize=14)
    ax.legend(fontsize=11)

    fig.tight_layout()

    return fig

# =========================================================
# Task 4b: Relationships between any two variables
# =========================================================

# ---------------------------------------------------------
# User Interface Functions
# ---------------------------------------------------------

# Function which gives correlation, means of variables etc. for each specified variable
def relationship_table(sample_data, var1, var2):
    df = sample_data[[var1, var2]].dropna().copy()
    df = apply_dayofweek_order(df, var1)
    df = apply_dayofweek_order(df, var2)

    x = df[var1]
    y = df[var2]

    if is_numeric(x) and is_numeric(y):
        corr = df[[var1, var2]].corr().iloc[0, 1]
        return pd.DataFrame({
            "Metric": ["N", "Correlation (Pearson)", f"Mean {var1}", f"Mean {var2}", f"SD {var1}", f"SD {var2}"],
            "Value": [
                len(df),
                round(corr, 3),
                round(x.mean(), 3),
                round(y.mean(), 3),
                round(x.std(), 3),
                round(y.std(), 3),
            ]
        })

    if is_numeric(x) and not is_numeric(y):
        out = df.groupby(var2)[var1].agg(N="count", mean="mean", median="median", std="std").reset_index()
        if var2 == "DayofWeek":
            out[var2] = pd.Categorical(out[var2], DAY_ORDER, ordered=True)
            out = out.sort_values(var2)
        else:
            out = out.sort_values("N", ascending=False)
        return out.round(3).reset_index(drop=True)

    if not is_numeric(x) and is_numeric(y):
        out = df.groupby(var1)[var2].agg(N="count", mean="mean", median="median", std="std").reset_index()
        if var1 == "DayofWeek":
            out[var1] = pd.Categorical(out[var1], DAY_ORDER, ordered=True)
            out = out.sort_values(var1)
        else:
            out = out.sort_values("N", ascending=False)
        return out.round(3).reset_index(drop=True)

    ct = pd.crosstab(df[var1], df[var2])

    if var1 == "DayofWeek":
        ct = ct.reindex(DAY_ORDER)
    if var2 == "DayofWeek":
        ct = ct.reindex(columns=DAY_ORDER)

    return ct.dropna(how="all").dropna(axis=1, how="all").fillna(0).astype(int)


# Plot relationship between two distinct variables
def relationship_plot(sample_data, var1, var2):
    df = sample_data[[var1, var2]].dropna().copy()
    df = apply_dayofweek_order(df, var1)
    df = apply_dayofweek_order(df, var2)

    x = df[var1]
    y = df[var2]

    if is_numeric(x) and is_numeric(y):
        fig, ax = plt.subplots()
        sns.regplot(data=df, x=var1, y=var2, scatter_kws={"alpha": 0.5}, ax=ax)
        ax.set_title(f"{var1} vs {var2}")
        fig.tight_layout()
        return fig

    if is_numeric(x) and not is_numeric(y):
        order = DAY_ORDER if var2 == "DayofWeek" else df[var2].value_counts().index.tolist()
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.boxplot(data=df, x=var2, y=var1, order=order, ax=ax)
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        return fig

    if not is_numeric(x) and is_numeric(y):
        order = DAY_ORDER if var1 == "DayofWeek" else df[var1].value_counts().index.tolist()
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.boxplot(data=df, x=var1, y=var2, order=order, ax=ax)
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        return fig

    ct = pd.crosstab(df[var1], df[var2])

    if var1 == "DayofWeek":
        ct = ct.reindex(DAY_ORDER)
    if var2 == "DayofWeek":
        ct = ct.reindex(columns=DAY_ORDER)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(ct.fillna(0).astype(int), annot=True, fmt="d", ax=ax)
    fig.tight_layout()
    return fig

# ---------------------------------------------------------
# Task 4b: Custom plots and tables
# ---------------------------------------------------------

# Custom correlation matric
def plot_correlation_matrix(sample_data):
    num = sample_data.select_dtypes(include="number")
    corr = num.corr()

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(corr, annot=True, fmt=".2f")
    ax.set_title("Correlation heatmap")
    fig.tight_layout()
    return fig

# Table of correlations between variables (descending order)
def make_correlations_table(sample_data):
    correlations = (
        sample_data[NUMERICAL_COLS]
        .corr()
        .stack()
        .reset_index()
        .rename(columns={0: "correlation"})
    )

    correlations = correlations[
        correlations["level_0"] < correlations["level_1"]
    ].sort_values("correlation", ascending=False)

    return correlations

# =========================================================
# Task 5: Relationships with Length of Stay and Breach or Not Variables
# =========================================================

# Breach rate correlations table
def los_breach_correlation_table(sample_data):
    df = add_breach_binary(sample_data)[["LoS", "Breach_binary"]].dropna()
    corr, p = pearsonr(df["LoS"], df["Breach_binary"])
    return pd.DataFrame({
        "Metric": ["N", "Correlation (Pearson)", "P-value", "Significance"],
        "Value": [len(df), round(corr, 3), round(p, 5), significance_label(p)]
    })

# Plot LoS by Breach Category
def plot_los_by_breach_binary(sample_data):
    df = add_breach_binary(sample_data)[["LoS", "Breach_binary"]].dropna()
    fig, ax = plt.subplots()
    sns.boxplot(x="Breach_binary", y="LoS", data=df, ax=ax)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Non-breach", "Breach"])
    ax.set_title("Length of Stay by Breach Status")
    ax.set_ylabel("Length of Stay (minutes)")
    ax.set_xlabel("")
    fig.tight_layout()
    return fig

# Length of Stay correlations table
def los_factor_correlations_table(sample_data):
    df = sample_data.copy()

    los_corr_vars = [
        "Age",
        "Day",
        "Period",
        "noofinvestigation",
        "nooftreatment",
        "noofpatients"
    ]

    base = df[["LoS"] + los_corr_vars].dropna()
    rows = []

    for var in los_corr_vars:
        corr, p = pearsonr(base["LoS"], base[var])
        rows.append({
            "Variable": var,
            "Correlation_with_LoS": corr,
            "P_value": p,
            "Significance": significance_label(p)
        })

    out = pd.DataFrame(rows)
    out["Correlation_with_LoS"] = out["Correlation_with_LoS"].round(3)
    out["P_value"] = out["P_value"].round(5)
    return out.sort_values("Correlation_with_LoS", ascending=False).reset_index(drop=True)

# Plot LoS Correlations
def plot_los_factor_correlations(sample_data):
    corr_df = los_factor_correlations_table(sample_data)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x="Correlation_with_LoS", y="Variable", data=corr_df, orient="h", ax=ax)
    ax.axvline(0, linestyle="--")
    ax.set_title("Correlation of Key Factors with Length of Stay")
    ax.set_xlabel("Pearson Correlation Coefficient")
    ax.set_ylabel("")
    fig.tight_layout()
    return fig

# Plot Average LoS by Age Group
def plot_avg_los_by_age_group(sample_data):
    df = sample_data[["Age", "LoS"]].dropna().copy()
    bins = [0, 20, 40, 60, 80, 120]
    labels = ["0-20", "21-40", "41-60", "61-80", "81+"]
    df["Age_group"] = pd.cut(df["Age"], bins=bins, labels=labels, include_lowest=True)
    los_by_age = df.groupby("Age_group", observed=False)["LoS"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x="Age_group", y="LoS", data=los_by_age, ax=ax)
    ax.set_title("Average Length of Stay by Age Group")
    ax.set_xlabel("Age Group (years)")
    ax.set_ylabel("Average LoS (minutes)")
    fig.tight_layout()
    return fig

# Plot LoS vs investigations and treatments bar chart
def plot_los_vs_investigations_and_treatments(sample_data):
    df = sample_data[["noofinvestigation", "nooftreatment", "LoS"]].dropna().copy()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.boxplot(x="noofinvestigation", y="LoS", data=df, ax=axes[0])
    axes[0].set_title("LoS vs Number of Investigations")
    axes[0].set_xlabel("Number of Investigations")
    axes[0].set_ylabel("LoS (minutes)")

    sns.boxplot(x="nooftreatment", y="LoS", data=df, ax=axes[1])
    axes[1].set_title("LoS vs Number of Treatments")
    axes[1].set_xlabel("Number of Treatments")
    axes[1].set_ylabel("LoS (minutes)")

    fig.tight_layout()
    return fig

# Plot LoS vs investigations bar chart only
def plot_los_vs_investigations_only(sample_data):
    df = sample_data[["noofinvestigation", "LoS"]].dropna().copy()
    fig, ax = plt.subplots()
    sns.boxplot(x="noofinvestigation", y="LoS", data=df, ax=ax)
    ax.set_title("LoS vs Number of Investigations")
    ax.set_xlabel("Number of Investigations")
    ax.set_ylabel("LoS (minutes)")
    fig.tight_layout()

    return fig

# Length of Stay vs Occupancy at AED plot
def plot_los_vs_patient_volume(sample_data):
    df = sample_data[["noofpatients", "LoS"]].dropna().copy()
    fig, ax = plt.subplots()
    sns.scatterplot(x="noofpatients", y="LoS", data=df, alpha=0.6, ax=ax)
    ax.set_title("Length of Stay vs AED Patient Volume")
    ax.set_xlabel("Patients Present on Arrival")
    ax.set_ylabel("Length of Stay (minutes)")
    fig.tight_layout()
    return fig

# Average Length of Stay by Period (Hour)
def plot_avg_los_by_period(sample_data):
    df = sample_data[["Period", "LoS"]].dropna().copy()
    los_by_period = df.groupby("Period")["LoS"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.barplot(x="Period", y="LoS", data=los_by_period, ax=ax)
    ax.set_title("Average Length of Stay by Arrival Period")
    ax.set_xlabel("Arrival Period (Hour of Day)")
    ax.set_ylabel("Average Length of Stay (minutes)")
    fig.tight_layout()
    return fig

# Numeric correlations with breach rate table
def breach_numeric_correlations_table(sample_data):
    df = add_breach_binary(sample_data).copy()
    numeric_vars = ["Age", "noofinvestigation", "nooftreatment", "noofpatients"]
    base = df[["Breach_binary"] + numeric_vars].dropna()

    rows = []
    for var in numeric_vars:
        corr, p = pointbiserialr(base["Breach_binary"], base[var])
        rows.append({
            "Variable": var,
            "Correlation_with_Breach": corr,
            "P_value": p,
            "Significance": significance_label(p)
        })

    out = pd.DataFrame(rows)
    out["Correlation_with_Breach"] = out["Correlation_with_Breach"].round(3)
    out["P_value"] = out["P_value"].round(5)
    return out.sort_values("Correlation_with_Breach", ascending=False).reset_index(drop=True)

# Numerical correlations with breach rate plot (bar chart)
def plot_breach_numeric_correlations(sample_data):
    corr_df = breach_numeric_correlations_table(sample_data)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x="Correlation_with_Breach", y="Variable", data=corr_df, orient="h", ax=ax)
    ax.axvline(0, linestyle="--")
    ax.set_title("Correlation of Key Factors with Breach")
    ax.set_xlabel("Point-biserial Correlation Coefficient")
    ax.set_ylabel("")
    fig.tight_layout()
    return fig

# Chi2 test for breach rate and other categorical variables
def breach_categorical_chi_square_table(sample_data):
    df = add_breach_binary(sample_data).copy()
    if "DayofWeek" in df.columns:
        df = apply_dayofweek_order(df, "DayofWeek")

    categorical_vars = ["DayofWeek", "Period"]
    rows = []

    for var in categorical_vars:
        if var not in df.columns:
            continue
        sub = df[[var, "Breach_binary"]].dropna()
        contingency = pd.crosstab(sub[var], sub["Breach_binary"])
        chi2, p, dof, _ = chi2_contingency(contingency)
        rows.append({
            "Variable": var,
            "Chi2": round(chi2, 2),
            "dof": int(dof),
            "P_value": round(p, 5),
            "Significance": significance_label(p)
        })

    return pd.DataFrame(rows)

# Age Distribution plot
def plot_age_distribution_by_breach(sample_data):
    df = add_breach_binary(sample_data)[["Age", "Breach_binary"]].dropna()
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(data=df, x="Age", hue="Breach_binary", bins=10, multiple="stack", ax=ax)
    ax.set_title("Patient Age Distribution by Breach Status")
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("Number of Patients")
    ax.legend(["Non-breach", "Breach"])
    fig.tight_layout()
    return fig

# Plot number of investigations by Breach category
def plot_investigations_by_breach(sample_data):
    df = add_breach_binary(sample_data)[["noofinvestigation", "Breach_binary"]].dropna()
    bins = range(0, int(df["noofinvestigation"].max()) + 2)

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(data=df, x="noofinvestigation", hue="Breach_binary", bins=bins, multiple="stack", ax=ax)
    ax.set_title("Number of Investigations by Breach Status")
    ax.set_xlabel("Number of Investigations")
    ax.set_ylabel("Number of Patients")
    fig.tight_layout()
    return fig

# Breach Rate by day
def plot_breach_rate_by_day(sample_data):
    df = add_breach_binary(sample_data).copy()
    df = apply_dayofweek_order(df, "DayofWeek")
    sub = df[["DayofWeek", "Breach_binary"]].dropna()

    breach_by_day = sub.groupby("DayofWeek", observed=False)["Breach_binary"].mean().reindex(DAY_ORDER)

    fig, ax = plt.subplots()
    ax.bar(range(len(breach_by_day)), breach_by_day.values, edgecolor="black")
    ax.set_xticks(range(len(breach_by_day)))
    ax.set_xticklabels(DAY_ORDER, rotation=45, ha="right")
    ax.set_title("Breach Rate by Day of Week")
    ax.set_ylabel("Proportion Breached")
    ax.set_xlabel("Day of Week")
    fig.tight_layout()
    return fig

# Plot breach rate by period - bar chart
def plot_breach_rate_by_period_bar(sample_data):
    df = add_breach_binary(sample_data)[["Period", "Breach_binary"]].dropna()
    breach_by_period = df.groupby("Period")["Breach_binary"].mean().reset_index().sort_values("Period")

    fig, ax = plt.subplots()
    ax.bar(breach_by_period["Period"], breach_by_period["Breach_binary"])
    ax.set_title("Breach Rate by Arrival Period")
    ax.set_xlabel("Arrival Hour")
    ax.set_ylabel("Proportion Breached")
    ax.set_xticks(range(0, 24, 2))
    fig.tight_layout()
    return fig

# Breach rate by day table
def breach_rate_by_day_table(sample_data):
    # for table (already ordered)
    df = add_breach_binary(sample_data).copy()
    df = apply_dayofweek_order(df, "DayofWeek")
    sub = df[["DayofWeek", "Breach_binary"]].dropna()
    out = sub.groupby("DayofWeek", observed=False)["Breach_binary"].mean().reindex(DAY_ORDER).reset_index()
    out["Breach Rate (%)"] = (out["Breach_binary"] * 100).round(2)
    out = out.drop(columns=["Breach_binary"])
    return out

# Breach Rate by period
def breach_rate_by_period_table(sample_data):
    df = add_breach_binary(sample_data)[["Period", "Breach_binary"]].dropna()
    out = df.groupby("Period")["Breach_binary"].mean().reset_index().sort_values("Period")
    out["Breach Rate (%)"] = (out["Breach_binary"] * 100).round(2)
    out = out.drop(columns=["Breach_binary"])
    return out.reset_index(drop=True)


# =========================================================
# Task 5: Menu for interface dropdown menu
# =========================================================
def section3_tables(sample_data):
    return {
        "LoS vs Breach correlation (Pearson + p)": los_breach_correlation_table(sample_data),
        "LoS factor correlations (Pearson + p)": los_factor_correlations_table(sample_data),
        "Breach correlations with numeric vars (point-biserial + p)": breach_numeric_correlations_table(sample_data),
        "Breach associations with categorical vars (chi-square)": breach_categorical_chi_square_table(sample_data),
        "Breach rate by Day of Week (table)": breach_rate_by_day_table(sample_data),
        "Breach rate by Period (table)": breach_rate_by_period_table(sample_data),
    }


def section3_plots(sample_data):
    return {
        "LoS by Breach (binary boxplot)": plot_los_by_breach_binary(sample_data),
        "LoS factor correlations (barplot)": plot_los_factor_correlations(sample_data),
        "Average LoS by Age Group": plot_avg_los_by_age_group(sample_data),
        "LoS vs Investigations and Treatments (boxplots)": plot_los_vs_investigations_and_treatments(sample_data),
        "LoS vs Patient Volume (scatter)": plot_los_vs_patient_volume(sample_data),
        "Average LoS by Period": plot_avg_los_by_period(sample_data),
        "Breach correlations with numeric vars (barplot)": plot_breach_numeric_correlations(sample_data),
        "Age distribution by Breach (stacked)": plot_age_distribution_by_breach(sample_data),
        "Investigations by Breach (stacked)": plot_investigations_by_breach(sample_data),
        "Breach rate by Day of Week (ordered bar)": plot_breach_rate_by_day(sample_data),
        "Breach rate by Period (bar)": plot_breach_rate_by_period_bar(sample_data),
    }
