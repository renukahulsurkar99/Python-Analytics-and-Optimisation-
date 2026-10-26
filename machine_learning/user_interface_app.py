from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_config import DATA_PATH, QUICK_TEST, choose_threshold, capture_plots
#TASK 07 & 08: USER INTERFACE & DATA MANAGEMENT

# run the streamlit app by executing this command in the terminal
# streamlit run user_interface_app.py

# Import packages
import streamlit as st
from streamlit_option_menu import option_menu
import os, json, uuid
from datetime import datetime, timezone
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, pointbiserialr, chi2_contingency

# Import other files
from machine_learning import aed_analysis as aed
from optimisation.finalopt import solve_task2



# Correct day order to rearrange
DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]

# Menu for the optimisation dropdown menu
MENU = {
    "1: Show the overall cost": "overall_cost",
    "2: Show the number of hours assigned to an operator": "op_hours",
    "3: Show the schedule on a particular day": "day_schedule",
    "4: Show the schedule of an operator": "op_schedule",
    "5: Show the overall schedule": "overall_schedule",
}

LOG_PATH = str(Path(__file__).resolve().parents[1] / "local_logs" / "audit_log.jsonl")

st.set_page_config(page_title="AED Dashboard", layout="wide")


def show_dataframe(data, **kwargs):
    """Render mixed numeric/text summary columns without Arrow fallback errors."""
    if isinstance(data, pd.DataFrame):
        data = data.copy()
        for column in data.columns:
            if data[column].dtype == object:
                types = {type(value) for value in data[column].dropna()}
                if len(types) > 1:
                    data[column] = data[column].map(lambda value: "" if pd.isna(value) else str(value))
    return st.dataframe(data, **kwargs)


# mMking a stable session ID
def get_session_id():
    """Stable ID for this user's Streamlit session (per browser tab)."""
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    return st.session_state.session_id

# Forming the timestamp for and session id etc for the log
def log_action(action: str, details: dict | None = None):

    event = {
        "ts_utc": datetime.now(timezone.utc).isoformat(),
        "session_id": get_session_id(),
        "action": action,
        "details": details or {},
    }

    os.makedirs(os.path.dirname(LOG_PATH) or ".", exist_ok=True)

    # Append the log item at the end of the JSON file
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")

# Helper function to read last few logs
def read_last_logs(n: int = 200) -> list[dict]:
    if not os.path.exists(LOG_PATH):
        return []
    with open(LOG_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()[-n:]
    out = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return out


@st.cache_data
# Ensure the day order is as it is supposed to in DAY_ORDER
def ensure_day_order(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or "Day" not in df.columns:
        return df
    out = df.copy()
    out["Day"] = pd.Categorical(out["Day"], categories=DAY_ORDER, ordered=True)
    return out

# Function for creating Task 1: Optimisation menu
def task1_menu_panel(res: dict):
    st.subheader("Menu")

    choice_label = st.selectbox("Choose an option:", list(MENU.keys()))
    choice = MENU[choice_label]

    # Option 1: overall cost
    if choice == "overall_cost":
        cost = res.get("total_cost", None)
        st.metric("Overall cost (£)", "—" if cost is None else f"{cost:,.2f}")

    # Option 2: operator total hours
    elif choice == "op_hours":
        weekly_df = res.get("weekly_df")
        if weekly_df is None or weekly_df.empty:
            st.warning("No weekly totals available.")
            return

        operators = weekly_df["Name"].tolist()
        name = st.selectbox("Enter the name of the operator:", operators)

        hours = float(weekly_df.loc[weekly_df["Name"] == name, "Weekly_Hours"].iloc[0])
        st.success(f"The number of hours assigned to {name} is {hours:.0f} hours")

    # Option 3: schedule for a day
    elif choice == "day_schedule":
        schedule_df = ensure_day_order(res.get("schedule_df"))
        if schedule_df is None or schedule_df.empty:
            st.warning("No schedule table available.")
            return

        day = st.selectbox("Choose a day:", DAY_ORDER)
        day_sched = (
            schedule_df[schedule_df["Day"] == day][["Name", "Hours"]]
            .sort_values(["Hours", "Name"], ascending=[False, True])
        )

        if day_sched.empty:
            st.info(f"No assignments found for {day}.")
        else:
            st.write(f"Schedule for **{day}**")
            show_dataframe(day_sched, width="stretch", hide_index=True)

    # Option 4: schedule for an operator
    elif choice == "op_schedule":
        schedule_df = ensure_day_order(res.get("schedule_df"))
        if schedule_df is None or schedule_df.empty:
            st.warning("No schedule table available.")
            return

        operators = sorted(schedule_df["Name"].unique().tolist())
        name = st.selectbox("Choose an operator:", operators)

        op_sched = (
            schedule_df[schedule_df["Name"] == name][["Day", "Hours"]]
            .sort_values("Day")
        )

        if op_sched.empty:
            st.info(f"No assignments found for {name}.")
        else:
            st.write(f"Schedule for **{name}**")
            show_dataframe(op_sched, width="stretch", hide_index=True)

    # Option 5: overall schedule matrix
    elif choice == "overall_schedule":
        pivot_df = res.get("pivot_df")
        if pivot_df is None or pivot_df.empty:
            st.warning("No schedule matrix available.")
            return

        # Ensure day columns ordered (critical)
        pivot_df = pivot_df.reindex(columns=DAY_ORDER)

        st.write("Overall schedule matrix (Hours)")
        show_dataframe(pivot_df, width="stretch")

        # Downloads (nice touch)
        st.download_button(
            "Download schedule matrix (CSV)",
            data=pivot_df.to_csv().encode("utf-8"),
            file_name="task1_schedule_matrix.csv",
            mime="text/csv",
        )

# Loading and sampling dataset, compatible with function in Task 4 file
def load_sample():
    return aed.load_and_sample()

# Helper function to render plots and tables when selecting Tables or Plots from the menu
def render_tables_or_plots(tables, plots):
    view = st.radio("Show", ["Tables", "Plots"], horizontal=True)

    if view == "Tables":
        name = st.selectbox("Choose a table", list(tables.keys()))
        show_dataframe(tables[name], width="stretch")
    else:
        name = st.selectbox("Choose a plot", list(plots.keys()))
        fig = plots[name]
        st.pyplot(fig, clear_figure=True)

# Function which converts log into a table
def read_log_as_dataframe() -> pd.DataFrame:
    if not os.path.exists(LOG_PATH):
        return pd.DataFrame()

    records = []
    with open(LOG_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)


    if "details" in df.columns:
        details_df = pd.json_normalize(df["details"])
        details_df.columns = [f"details.{c}" for c in details_df.columns]
        df = pd.concat([df.drop(columns=["details"]), details_df], axis=1)

    return df






# ===========================
# MAIN FUNCTION INTERFACE
# ===========================

def main():

    # Link to bootstrap icons, which are imported
    st.markdown("""
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.css">
    """, unsafe_allow_html=True)

    # Creating a sidebar menu
    with st.sidebar:
        st.title("Accident & Emergency Department Summary Dashboard")
        main_section = option_menu(
            menu_title=None,
            options=["Tasks 1-3: Optimisation", "Task 4a: Descriptive Statistics (Single Variables)", "Task 4b: Relationships Between Variables", "Task 5: Factors influencing Length of Stay and Breach", "Raw Data Access"],
            icons=["pie-chart", "bar-chart", "graph-up", "bar-chart-steps", "database-fill-gear"],
            default_index=0,
        )

    # Loading full and sample data, which we are gonna work with
    # Scheduling is independent of the patient dataset.
    if main_section != "Tasks 1-3: Optimisation":
        try:
            full_data = aed.load_data()
            sample_data = aed.make_sample(full_data)
        except FileNotFoundError:
            st.error("The local AED dataset is missing. See data/README.md for setup.")
            st.stop()


    # =========================================
    # Tasks 1-3 (Optimisation)
    # =========================================

    if main_section == "Tasks 1-3: Optimisation":
        st.header("Skills-constrained scheduling")

        if "task1_results" not in st.session_state:
            st.session_state["task1_results"] = None

        if st.button("Solve scheduling optimisation", type="primary"):
            with st.spinner("Solving..."):
                st.session_state["task1_results"] = solve_task2()   # <-- real call

        # always assigning res, regardless of button click
        res = st.session_state["task1_results"]

        # Warning mesage if no res
        if res is None:
            st.warning("No results yet. Click **Solve scheduling optimisation**.")
            st.stop()

        task1_menu_panel(res)

    # =========================================
    # Task 4a: Descriptive Statistics (Single Variables)
    # =========================================

    if main_section == "Task 4a: Descriptive Statistics (Single Variables)":
        st.markdown("""<h1>
          <i class="bi bi-bar-chart"></i>
          Task 4a: Descriptive statistics
            </h1>""", unsafe_allow_html=True)

        st.info("This section shows the descriptive statistics of the dataset. The central tendencies, dispersion metrics and other statistics of specific variables are reported here")
        # Expander for the subsections descriptions
        with st.expander("Show sub-section descriptions"):
            st.markdown("""
                **Sub-sections:**
            - **Dashboard**: Key descriptive statistics
            - **Single Variables**: Custom single variables statistics and distributions
            """)

        # Adding some spaces in between the objects
        st.markdown("")
        st.markdown("")
        st.markdown("")


        # # Makes the tabs larger and stand more out
        st.markdown("""
        <style>
        /* Make tabs text bigger */
        button[data-baseweb="tab"] {
            font-size: 24px;
            padding: 10px 156px;
        }

        /* Optional: make active tab stand out more */
        button[data-baseweb="tab"][aria-selected="true"] {
            font-weight: 600;
        }
        </style>
        """, unsafe_allow_html=True)

        # Defining tabs for this page
        tab_dashboard, tab_single = st.tabs(["Dashboard", "Single variables"])

        with tab_dashboard:
            st.markdown("")
            st.markdown("")


            st.subheader("Breach Analysis")


            st.markdown("")
            st.markdown("")
            st.markdown("")

            # Creates three columns so that metrics displayed can be side by side
            col1, col2, col3 = st.columns(3)

            # First metric
            with col1:
                st.metric(
                label="Breach Rate",
                value=f"{round((sample_data["Breachornot"] == "breach").mean() * 100, 1):.2f} %"
                )
            # Second metric
            with col2:
                st.metric(
                label=f" Average Length of Stay (hours)",
                value=f"{round(sample_data["LoS"].mean() / 60, 1)}"
                )
            # Third metric
            with col3:
                st.metric(
                label=f" Average Length of Stay (minutes)",
                value=f"{round(sample_data["LoS"].mean(), 1)}"
                )

            if round((sample_data["Breachornot"] == "breach").mean() * 100, 1) <= 5:

                # success message
                st.success("Breach Target satisfied")
            else:
                st.warning("Breach Target higher than 5%!")


            st.markdown("")
            st.markdown("")
            st.divider()

            st.markdown("")
            st.markdown("")

            st.subheader("Occupancy Statistics")

            st.markdown("")
            st.markdown("")
            st.markdown("")

            # Three statistics
            col4, col5, col6 = st.columns(3)


            with col4:
                st.metric(
                label=f" Busiest Day of Week with {sample_data["DayofWeek"].value_counts().max()} patients",
                value=f"{sample_data["DayofWeek"].value_counts().idxmax()}"
                )
            with col5:
                st.metric(
                label=f" Busiest Hour with {sample_data["Period"].value_counts().max()} patients in total",
                value=f"{sample_data["Period"].value_counts().idxmax()}:00 - {sample_data["Period"].value_counts().idxmax() +1}:00"
                )
            with col6:
                st.metric(
                label="Average AED Occupancy on Arrival",
                value=f"{round(sample_data["noofpatients"].mean(), 1)} patients"
                )


            st.markdown("")
            st.markdown("")

            st.divider()

            st.markdown("")
            st.markdown("")

            # Printing key summary table
            tbl = aed.key_summary(sample_data)

            st.markdown("")

            st.subheader("Key Statistics")
            show_dataframe(tbl, width="stretch")

            st.markdown("")
            st.markdown("")

            st.divider()

            # Specify centred heading
            st.markdown(
            "<h2 style='text-align: center;'>Key Plots</h2>",
            unsafe_allow_html=True
            )

            # Create a small line for division of sections
            st.markdown("""
            <hr style="
                width: 5%;
                margin-left: 46%;
                margin-top: 1rem;
                margin-bottom: 1.5rem;
                border: 1px solid #ccc;
            ">
            """, unsafe_allow_html=True)


            st.markdown("")


            # Different distribution plots tabs
            col7, col8 = st.columns(2, gap="large")

            with col7:
                st.markdown("### Arrival Time Distribution")
                fig = aed.custom_arrival_time_distribution(sample_data)
                st.pyplot(fig, clear_figure=True, width="stretch")

                with st.expander("Show description"):
                    st.markdown("""
                    Most patients start coming at around 9AM and peak hours continue until 8PM with some fluctuations from hour to hour.
                    """)

            with col8:
                st.markdown("### Age Distribution")
                fig = aed.custom_age_distribution(sample_data)
                st.pyplot(fig, clear_figure=True, width="stretch")

                # Expander for hidden description of the plot
                with st.expander("Show description"):
                    st.markdown(f"""
                   Although mean age is {round(sample_data['Age'].mean(),1)} the distribution is skewed with higher number of younger individuals attending the hospital
                    """)

            col9, col10 = st.columns(2, gap="large")
            with col9:
                st.markdown("### HRG Group Frequency")
                fig = aed.custom_HRG_distribution(sample_data)
                st.pyplot(fig, clear_figure=True, width="stretch")

                with st.expander("Show description"):
                    st.markdown(f"""
                    Most patients  ({sample_data["HRG"].value_counts().max()/ aed.SAMPLE_SIZE*100}%) come from an HRG of {sample_data["HRG"].value_counts().idxmax()}. On the other hand merely {sample_data["HRG"].value_counts().min()/ 400*100}% of patient coming to the hospital belonged to {sample_data["HRG"].value_counts().idxmin()} group.
                    """)

            with col10:
                st.markdown("### Length of Stay Distribution")
                fig = aed.custom_LoS_distribution(sample_data)
                st.pyplot(fig, clear_figure=True, width="stretch")

                with st.expander("Show description"):
                    st.markdown(f"""
                    The distribution indicates that only few cases breach the limit of 4 hours length of stay. However, many patients reach close to the limit boundary.
                    """)

        # Tab for single variable distributions
        with tab_single:
            st.subheader("Overview")
            cols = [col for col in sample_data.columns if col != "ID"]
            var = st.selectbox("Select a variable", cols, index=0)

            tbl = aed.single_variable_table(sample_data, var)
            fig = aed.single_variable_plot(sample_data, var)

            st.subheader("Table")
            show_dataframe(tbl, width="stretch")

            st.subheader("Distribution")
            st.pyplot(fig, clear_figure=True)





    # =========================================
    # Task 4b:  Relationships between variables
    # =========================================
    elif main_section == "Task 4b: Relationships Between Variables":
        st.markdown("""<h1>
          <i class="bi bi-graph-up"></i>
          Task 4b: Relationships Between Variables
            </h1>""", unsafe_allow_html=True)

        st.info("This section shows the relationships between all variables. Correlations and charts are used to visualise the relationships.")

        with st.expander("Show sub-section descriptions"):
            st.markdown("""
                **Sub-sections:**
            - **Dashboard**: Summary metrics and visualisations of key relationships found
            - **Custom Relationship plots**: In this sub-section you can view the statistics and visualisations for your chosen variables

            """)
        st.markdown("")
        st.markdown("")
        st.markdown("")

        st.markdown("""
        <style>
        /* Make tabs text bigger */
        button[data-baseweb="tab"] {
            font-size: 24px;
            padding: 10px 127px;
        }

        /* Make active tab stand out more */
        button[data-baseweb="tab"][aria-selected="true"] {
            font-weight: 600;
        }
        </style>
        """, unsafe_allow_html=True)

        tab_dashboard, tab_relationship_plots = st.tabs(["Dashboard", "Custom Relationship Plots"])

        with tab_dashboard:
            st.markdown(
            "<h2 style='text-align: center;'>Correlation Matrix</h2>",
            unsafe_allow_html=True
            )

            st.markdown("""
            <hr style="
                width: 5%;
                margin-left: 46%;
                margin-top: 1rem;
                margin-bottom: 1.5rem;
                border: 1px solid #ccc;
            ">
            """, unsafe_allow_html=True)



            col1, col2, col3 = st.columns([0.9 ,3, 1])  # left column narrower for actual central alignment

            # Correlation matrix
            with col2:
                corr_mat = aed.plot_correlation_matrix(sample_data)

                st.pyplot(corr_mat, clear_figure=True, width="content")

            tbl = aed.make_correlations_table(sample_data)

            #figure = aed.

            st.markdown("")

            st.subheader("Highest correlated variables")
            show_dataframe(tbl, width="stretch")

        # Custom relationship plots
        with tab_relationship_plots:
            st.subheader("Overview")

            cols = [col for col in sample_data.columns if col != "ID"]
            c1, c2 = st.columns(2)

            with c1:
                var1 = st.selectbox("Select Variable 1", cols, index=0)
            with c2:
                default_index = 1 if len(cols) > 1 else 0
                var2 = st.selectbox("Select Variable 2", cols, index=default_index)

            if var1 == var2:
                st.warning("Please select two different variables.")
                return

            tbl = aed.relationship_table(sample_data, var1, var2)
            fig = aed.relationship_plot(sample_data, var1, var2)

            st.subheader("Table")
            show_dataframe(tbl, width="stretch")

            st.subheader("Plot")
            st.pyplot(fig, clear_figure=True)

    # =========================================
    # Task 5: Factors Influencing Length of Stay and Breach
    # =========================================
    elif main_section == "Task 5: Factors influencing Length of Stay and Breach":
        st.markdown("""<h1>
          <i class="bi bi-clipboard-data"></i>
          Task 5: Factors influencing Length of Stay and Breach
            </h1>""", unsafe_allow_html=True)

        # Section description
        st.info("This section shows the relationships between all variables and our variables of interest Length of Stay and Breach. Correlations and Visualisations of the relationships will be provided in this section in order to explore the most significant factors influencing breaches of hospital stay limit.")

        with st.expander("Show sub-section descriptions"):
            st.markdown("""
                **Sub-sections:**
            - **Dashboard**: Key findings of relationships
            - **Statistical Tests**: Shows correlation tables with each variable and their calculated p-values
            - **Plots and Tables**: Allows you to select the relevant visualisations of factors influencing Length of Stay and Breach Rate along with tables of correlations and mean comparisons
            """)
        st.markdown("")
        st.markdown("")
        st.markdown("")



        st.markdown("""
        <style>
        /* Make tabs text bigger */
        button[data-baseweb="tab"] {
            font-size: 24px;
            padding: 10px 79px;
        }

        /* Make active tab stand out more */
        button[data-baseweb="tab"][aria-selected="true"] {
            font-weight: 600;
        }
        </style>
        """, unsafe_allow_html=True)

        tab_dash, tab_inference, tab_plots_tables = st.tabs(["Dashboard", "Statistical Tests", "Plots and tables"])



        with tab_dash:

            st.markdown("")
            st.markdown("")


            st.subheader("Key Metrics")


            st.markdown("")
            st.markdown("")
            st.markdown("")

            # Key metrics side by side
            col1, col2, col3 = st.columns(3)

            with col1:

                corr, p = pearsonr(sample_data["LoS"], sample_data["noofinvestigation"])

                st.metric(
                label="Most significant correlation with LoS",
                value=f"{round(corr, 3)}"
                )
            with col2:
                df_temp = aed.add_breach_binary(sample_data).copy()
                corr, p = pointbiserialr(df_temp["Breach_binary"], df_temp["Age"])
                st.metric(
                label=f" Most significant correlation with Breach",
                value=f"{round(corr, 3)}"
                )
            with col3:

                breach_group = sample_data[sample_data['Breachornot'] == 'breach']
                st.metric(
                label=f" Mean Length of Stay (min) when Breach",
                value=f"{round(breach_group["LoS"].mean(), 2)}"
                )

            st.divider()
            st.markdown("")

            st.subheader("Key Factors Related to Length of Stay")
            plot_corr = aed.plot_los_factor_correlations(sample_data)

            st.pyplot(plot_corr, clear_figure=True)

            st.markdown("")
            st.divider()

            st.markdown("")

            # Key Factors plots
            st.subheader("Key Factors Related to Breach")
            plot_corr_breach = aed.plot_breach_numeric_correlations(sample_data)

            st.pyplot(plot_corr_breach, clear_figure=True)

            st.markdown("")
            st.markdown("")
            st.divider()

            st.subheader("Breach Rate by Day of Week")

            # Braech rate percentage by day table
            tbl_breach_day = aed.breach_rate_by_day_table(sample_data)

            show_dataframe(tbl_breach_day, width="stretch")

            st.markdown("")
            st.markdown("")

            st.divider()

            st.markdown(
            "<h2 style='text-align: center;'>Key Relationships Plots</h2>",
            unsafe_allow_html=True
            )

            st.markdown("""
            <hr style="
                width: 5%;
                margin-left: 46%;
                margin-top: 1rem;
                margin-bottom: 1.5rem;
                border: 1px solid #ccc;
            ">
            """, unsafe_allow_html=True)


            st.markdown("")

            col7, col8 = st.columns(2, gap="large")

            with col7:
                st.markdown("### Length of Stay by Breach")
                fig = aed.plot_los_by_breach_binary(sample_data)
                st.pyplot(fig, clear_figure=True, width="stretch")

                with st.expander("Show description"):
                    breach_group = sample_data[sample_data['Breachornot'] == 'breach']
                    non_breach_group = sample_data[sample_data['Breachornot'] == 'non-breach']




                    st.markdown(f"""
                    The average LoS for people who Breached time limit is {round(breach_group["LoS"].mean(), 2)} and for those that did not {round(non_breach_group["LoS"].mean(), 2)}, which gives difference of {round(breach_group["LoS"].mean(), 2)-round(non_breach_group["LoS"].mean(), 2)}.
                    """)

            with col8:
                st.markdown("### Length of Stay and Number of Investigations")
                fig = aed.plot_los_vs_investigations_only(sample_data)
                st.pyplot(fig, clear_figure=True, width="stretch")

                with st.expander("Show description"):
                    st.markdown(f"""
                   Although it might seem small, there is a noticeable increase in the mean length of stay with the number of investigations performed. As shown on previous plots, this is one of the most significant correlations.
                    """)

            col9, col10 = st.columns(2, gap="large")
            with col9:
                st.markdown("### Average Length of Stay by Age")
                fig = aed.plot_avg_los_by_age_group(sample_data)
                st.pyplot(fig, clear_figure=True, width="stretch")

                with st.expander("Show description"):
                    st.markdown(f"""
                    The mean Length of Stay shows a gradual upward trend with rising Age. Therefore it can be inferred that on average older people (especially 80+ years old) stay longer in the emergency department than younger individuals.
                    """)


        # Inferential statistics section
        with tab_inference:

            table_cat = aed.breach_categorical_chi_square_table(sample_data)
            table_num = aed.breach_numeric_correlations_table(sample_data)
            table_num_LoS = aed.los_factor_correlations_table(sample_data)


            st.subheader("Categorical Variables with Breach (Chi-square)")

            show_dataframe(table_cat, width="stretch")

            st.subheader("Numerical Variables with Breach (Pearson Correlation)")

            show_dataframe(table_num, width="stretch")

            st.subheader("Numerical Variables with Length of Stay (Pearson Correlation)")

            show_dataframe(table_num_LoS, width="stretch")


        # Plot custom tables and plots
        with tab_plots_tables:
            tables = aed.section3_tables(sample_data)
            plots = aed.section3_plots(sample_data)

            view = st.radio("Show", ["Tables", "Plots"], horizontal=True)

            if view == "Tables":
                name = st.selectbox("Choose a table", list(tables.keys()))
                show_dataframe(tables[name], width="stretch")
            else:
                name = st.selectbox("Choose a plot", list(plots.keys()))
                st.pyplot(plots[name], clear_figure=True)

    # =========================================
    # Task 7: Accessing and modifying raw data
    # =========================================
    elif main_section == "Raw Data Access":



        st.set_page_config(page_title="Raw Data Display", layout="wide")

        # ============================
        # Configuration
        # ============================
        ID_COL = "ID"  # set to your id column name; if missing it will be created

        def ensure_unique_id(df: pd.DataFrame, id_col: str = ID_COL) -> pd.DataFrame:
            df = df.copy()
            if id_col not in df.columns:
                df.insert(0, id_col, range(1, len(df) + 1))
            if df[id_col].duplicated().any():
                df[id_col] = pd.factorize(df[id_col])[0] + 1
            return df

        # ============================
        # Loading Data
        # ============================

        # Ensuring there is no df in session state before, if not, assigning it to our dataset
        if ("df" not in st.session_state) or (st.session_state.df is None) or (len(st.session_state.df) == 0):
            st.session_state.df = ensure_unique_id(full_data, ID_COL)

        # Saving session state of df in a df variable for easier manipulation
        df = st.session_state.df

        # ============================
        # Default filter setting for session state
        # ============================

        # Function defining default setting for filtering
        def default_filters_for(df_: pd.DataFrame) -> dict:
            return {
                "search_cols": [c for c in df_.columns if c != ID_COL][:5], # First five columns are selected as a default
                "search_query": "",
                "filter_col": "(none)", # filter value will be none at the beginning
                "min_val": None,
                "max_val": None,
                "visible_cols": df_.columns.tolist(),
                "cat_filters": {},
                }
                # most defaults are none as we will add to them with filtering

        # Setting the default session state of filters
        st.session_state.setdefault("filters", default_filters_for(df))


        # ============================
        # Checking filters in session state
        # ============================


        # Stores all columns
        all_cols = list(df.columns)

        # Stores numerical columns
        numeric_cols = [
            c for c in df.columns
               if pd.api.types.is_numeric_dtype(df[c]) and c != ID_COL
            ]

        # Stores categorical columns, which are of all types except ID and numerical
        categorical_cols = [
             c for c in df.columns
            if c != ID_COL and (
                pd.api.types.is_object_dtype(df[c]) or
                pd.api.types.is_string_dtype(df[c]) or
                isinstance(df[c].dtype, pd.CategoricalDtype) or
                pd.api.types.is_bool_dtype(df[c])
                )
            ]


        # Checking for change in data frame and whether columns options for filter settings are valid
        # Access the current filters in session state and store it
        current_filters = st.session_state.filters

        # Search if the columns for string search are still in the dataframe (checks all columns in all_cols)
        # if they are not there anymore it defaults to the first 5
        current_filters["search_cols"] = [c for c in current_filters.get("search_cols", []) if c in all_cols and c != ID_COL] \
                               or [c for c in all_cols if c != ID_COL][:5]

        # Searches if numerical columns in session state are still in the dataset
        # Defaults all numerical filters to none if not
        if current_filters.get("filter_col") not in (["(none)"] + numeric_cols):
             current_filters["filter_col"] = "(none)"
             current_filters["min_val"] = None
             current_filters["max_val"] = None

        # Checks if filtered visible columns in session state are still in the dataset
        current_filters["visible_cols"] = [c for c in current_filters.get("visible_cols", []) if c in all_cols] or all_cols

        # Ensures ID column is always visible and cannot be filtered out
        if ID_COL in all_cols and ID_COL not in current_filters["visible_cols"]:
            current_filters["visible_cols"] = [ID_COL] + current_filters["visible_cols"]

        # Access the categorical variables in feature state (stored as dictionary key value pairs)
        # and ensure that the column is still in the dataset
        cat_filters = current_filters.get("cat_filters", {})
        cat_filters = {col: vals for col, vals in cat_filters.items() if col in categorical_cols}
        current_filters["cat_filters"] = cat_filters

        # Assigns the checked and corrected filters to the session state
        st.session_state.filters = current_filters





        # ============================
        # Actual user interface
        # ============================
        st.title("Data View and Manipulation")

        tab_view, tab_patient, tab_add, tab_edit, tab_delete, tab_logs = st.tabs(["View", "Patient Search", "Add", "Edit", "Delete", "Logs"])


        with tab_view:
            with st.container(border=True):
                st.subheader("Search By String")

                #-------------------------------
                # Search by a string option
                #-------------------------------

                r1c1, r1c2 = st.columns([2.2, 2.2])

                # Select which column
                with r1c1:
                    st.multiselect(
                        "Search columns",
                        options=all_cols,
                        default=st.session_state.filters["search_cols"],
                        key="ui_search_cols",
                    )

                # Input a text that you want to search for (in selected columns)
                with r1c2:
                    st.text_input(
                        "Search (contains)",
                        value=st.session_state.filters["search_query"],
                        key="ui_search_query",
                    )

                st.divider()

                #-------------------------------
                # Hide or display columns (choose columns to display)
                #-------------------------------
                st.markdown("### Columns to display")
                st.multiselect(
                    "Displayed columns",
                    options=all_cols,
                    default=st.session_state.filters["visible_cols"],
                    key="ui_visible_cols",
                )

                st.divider()

                #-------------------------------
                # Filter a numerical variable by selecting range
                #-------------------------------
                st.markdown("### Numeric filter")

                r2c1, r2c2 = st.columns([2.2, 2.2])

                # Select a numerical variable
                with r2c1:
                    ui_filter_col = st.selectbox(
                        "Min/Max column",
                        ["(none)"] + numeric_cols,
                        index=(["(none)"] + numeric_cols).index(st.session_state.filters["filter_col"]),
                        key="ui_filter_col_live",
                    )

                # if maximum value and minimum value are equal then return nothing
                ui_min_val = ui_max_val = None


                with r2c2:
                    if ui_filter_col != "(none)":
                        s = df[ui_filter_col].dropna()

                        # if the selected numerical variable column is non-zero length
                        # then identify max and min values
                        if len(s):
                            mn, mx = float(s.min()), float(s.max())


                            prev_min = st.session_state.filters["min_val"]
                            prev_max = st.session_state.filters["max_val"]

                            # if there isn't the minimum value or maximum value in session state
                            # then slider will assume default min max values
                            if prev_min is None or prev_max is None:
                                slider_default = (mn, mx)

                            # if there is minimum or maximum value in the session state
                            # then it will choose larger from the minimum value and lower from the maximum value
                            else:
                                slider_default = (float(max(mn, prev_min)), float(min(mx, prev_max)))

                                # if minimum value is larger than maximum value (the values of maximum and minimum would switch)
                                # then set the values to default values
                                if slider_default[0] > slider_default[1]:
                                    slider_default = (mn, mx)

                            # Defining the actual numerical slider with values based on the previous conditions
                            ui_min_val, ui_max_val = st.slider(
                                f"{ui_filter_col} range",
                                min_value=mn,
                                max_value=mx,
                                value=slider_default,
                                key="ui_range_slider_live",
                            )
                        else:
                            st.info("Selected numeric column has no values.")
                    else:
                        st.caption("Select a numeric column to enable the range slider.")

                st.divider()

                #-------------------------------
                # Filter by a categorical variable level
                #-------------------------------
                st.markdown("### Categorical filters")

                st.caption("Choose one or more categorical columns, then select one or more allowed categories per column.")

                selected_cat_cols = st.multiselect(
                    "Categorical columns to filter",
                    options=categorical_cols,
                    default=list(st.session_state.filters.get("cat_filters", {}).keys()),
                    key="ui_cat_cols",
                )

                # Each category and its selected levels will be stored as a dictionary key

                # Initialise the current categorical filters dictionary
                current_cat_filters = {}

                # Define variable label for NAs
                MISSING_TOKEN = "(missing)"

                # For each selected column identify its unique values
                for col in selected_cat_cols:

                    # Store the selected column
                    ser = df[col]

                    # Are there any missing values?
                    has_missing = ser.isna().any()

                    # Identify unique values (levels) in the categorical variable
                    uniques = ser.dropna().unique().tolist()

                    # Sort the unique values
                    uniques_sorted = sorted(uniques, key=lambda x: str(x))

                    # Initialise options list
                    options = []

                    # if there are any missing values, assign the category level "(missing)"
                    if has_missing:
                        options.append(MISSING_TOKEN)

                    # Add all sorted unique values
                    options += uniques_sorted

                    # Default selection from previously applied filters (if any)
                    prev_vals = st.session_state.filters.get("cat_filters", {}).get(col, [])

                    # From unique values in session state only keep those values that are still
                    # in the dataset (if the dataset changed, could be different
                    prev_vals = [v for v in prev_vals if (v == MISSING_TOKEN) or (v in uniques_sorted)]

                    # Define the actual multiselect feature with options identified
                    # and with default values from the session state
                    sel = st.multiselect(
                        f"Allowed values for '{col}'",
                        options=options,
                        default=prev_vals,
                        key=f"ui_cat_vals__{col}",
                    )

                    # Only if something selected in the multiselect, assign it to the current category filters
                    if sel:
                        current_cat_filters[col] = sel

                st.divider()

                #-------------------------------
                # Define the Apply and Clear buttons
                #-------------------------------
                c_apply, c_clear, c_hint = st.columns([1, 1, 4])
                with c_apply:
                    apply_clicked = st.button("Apply filters", type="primary")
                with c_clear:
                    clear_clicked = st.button("Clear filters")
                with c_hint:
                    st.caption("Apply commits and filters the table.")

            #-------------------------------
            # Actions for Apply and Clear buttons
            #-------------------------------

            # What happens when the Apply button is clicked
            if apply_clicked:

                # Visible columns: must not be empty, must include ID_COL for edit/delete
                visible_cols = st.session_state.get("ui_visible_cols", st.session_state.filters["visible_cols"])

                # if there is nothing in visible columns (all deselected) then display at least ID
                # if even ID is not there then display all columns
                if not visible_cols:
                    visible_cols = [ID_COL] if ID_COL in all_cols else all_cols

                # if ID isn't in the visible columns selection it will still be displayed
                if ID_COL in all_cols and ID_COL not in visible_cols:
                    visible_cols = [ID_COL] + visible_cols

                # Collect categorical filters from current session state
                cat_filters_to_save = {}
                for col in st.session_state.get("ui_cat_cols", []):
                    vals = st.session_state.get(f"ui_cat_vals__{col}", [])
                    if vals:
                        cat_filters_to_save[col] = vals

                # Set the session state filters
                st.session_state.filters = {
                    "search_cols": st.session_state.get("ui_search_cols", st.session_state.filters["search_cols"]),
                    "search_query": st.session_state.get("ui_search_query", st.session_state.filters["search_query"]),
                    "filter_col": st.session_state.get("ui_filter_col_live", "(none)"),
                    "min_val": ui_min_val,
                    "max_val": ui_max_val,
                    "visible_cols": visible_cols,
                    "cat_filters": cat_filters_to_save,
                }

                log_action("APPLY_FILTERS", {
                    "search_query": st.session_state.filters.get("search_query", ""),
                    "search_cols": st.session_state.filters.get("search_cols", []),
                    "filter_col": st.session_state.filters.get("filter_col"),
                    "min_val": st.session_state.filters.get("min_val"),
                    "max_val": st.session_state.filters.get("max_val"),
                    "visible_cols": st.session_state.filters.get("visible_cols", []),
                    "cat_filters": st.session_state.filters.get("cat_filters", {}),
                })

                st.rerun()

            # What happens when the clear button is clicked
            if clear_clicked:

                # Assign current session state filters to default values
                st.session_state.filters = default_filters_for(df)

                # define which features to clear in session state
                keys_to_clear = [
                    "ui_search_cols", "ui_search_query", "ui_visible_cols",
                    "ui_filter_col_live", "ui_range_slider_live",
                    "ui_cat_cols",
                ]

                # Clear also all unique values of categories stored
                for k in list(st.session_state.keys()):
                    if str(k).startswith("ui_cat_vals__"):
                        keys_to_clear.append(k)

                # Remove the keys (selected features etc.) from the session state
                for k in keys_to_clear:
                    st.session_state.pop(k, None)

                st.rerun()

            # ============================
            # Filtering the actual dataset shown in table
            # ============================

            # Assign the filters session state to a variable
            f = st.session_state.filters

            # Create a copy of the dataset, which will be displayed
            filtered = df.copy()

            # Search by a string
            if f["search_query"].strip() and f["search_cols"]:

                # Remove white spaces and turn the entered string to lowercase
                q = f["search_query"].strip().lower()

                # The filtered dataset rows returned are only those with matching the string in selected columns
                # Also make sure the type is string
                filtered = filtered[
                    filtered[f["search_cols"]]
                    .astype(str)
                    .apply(lambda col: col.str.lower().str.contains(q, na=False)) # search each cell of each pre-defined column for the string
                    .any(axis=1)
                ]


            # Apply the numerical maximum and minimum values for the selected numerical variable column
            if f["filter_col"] != "(none)" and f["min_val"] is not None and f["max_val"] is not None:
                filtered = filtered[filtered[f["filter_col"]].between(f["min_val"], f["max_val"], inclusive="both")]



            # Apply categorical variable filters

            # Get the selected categorical columns and categorical levels from session state
            cat_filters = f.get("cat_filters", {})

            # check if the selected column is in filtered dataset
            # and check if there are any allowed variable levels (categories)
            for col, allowed in cat_filters.items():
                if col not in filtered.columns or not allowed:
                    continue

                # Make set from allowed (all possible) category levels
                allowed_set = set(allowed)

                # is there a "(missing)" category selected
                include_missing = MISSING_TOKEN in allowed_set

                # all allowed variables except the missing category
                allowed_values = [v for v in allowed if v != MISSING_TOKEN]

                # Create a mask for all rows with allowed values
                mask = filtered[col].isin(allowed_values)

                # If (missing) category selected include the missing values in mask
                if include_missing:
                    mask = mask | filtered[col].isna()

                # Filter the dataset
                filtered = filtered[mask]


            # =========================
            # METRICS
            # =========================

            # Display how many rows and columns are there
            m1, m2, m3 = st.columns(3)
            m1.metric("Total records", len(df))
            m2.metric("Filtered records", len(filtered))
            m3.metric("Columns", len(filtered.columns))

            st.divider()

            st.subheader("Records (filtered)")

            # Display the dataframe with only columns defined in visible columns filter
            display_cols = [c for c in f["visible_cols"] if c in filtered.columns]
            show_dataframe(filtered[display_cols], width="stretch", hide_index=True)

            f = st.session_state.filters
            display_cols = [c for c in f["visible_cols"] if c in filtered.columns]

            # Download filtered data
            csv_bytes = filtered[display_cols].to_csv(index=False).encode("utf-8")

            st.download_button(
                label="Download Filtered Data (CSV)",
                data=csv_bytes,
                file_name="filtered_data.csv",
                mime="text/csv",
                key="download_filtered_csv",
            )


        # =========================
        # Tab for displaying patient details
        # =========================
        with tab_patient:
            st.subheader("Patient details (select ID)")

            df_all = st.session_state.df

            if ID_COL not in df_all.columns:
                st.error(f"ID column '{ID_COL}' not found in the dataset.")
            else:
                id_options = df_all[ID_COL].tolist()

                # If no ID options found then throw an error
                if not id_options:
                    st.info("No rows to select with these filters applied")

                # Otherwise create a selectbox
                else:
                    selected_id = st.selectbox(
                        f"Select {ID_COL}",
                        options=id_options,
                        key="row_details_selected_id",
                    )

                    # Extract the row
                    row = df_all.loc[df_all[ID_COL] == selected_id]

                    # If row is empty then throw a warning
                    if row.empty:
                        st.warning("This ID does not exist.")
                    else:

                        st.markdown("### Patient Record")
                        row_kv = row.iloc[0].to_frame(name="value")
                        show_dataframe(row_kv, width="stretch")

            log_action("VIEW_PATIENT_DETAILS", {"selected_id": selected_id})

        with tab_add:
            st.subheader("Add a record")

            with st.form("add_form", clear_on_submit=True):

                # Initialise a new row
                new_row = {}

                for col in df.columns:

                    # continue if the column is ID
                    if col == ID_COL:
                        continue

                    # if boolean type put a checkbox there
                    if pd.api.types.is_bool_dtype(df[col]):
                        new_row[col] = st.checkbox(col, value=False)

                    # if numerical variable, then there is numerical input
                    elif pd.api.types.is_numeric_dtype(df[col]):
                        new_row[col] = st.number_input(col, value=0.0)


                    # Else accept textual input
                    else:
                        new_row[col] = st.text_input(col, "")

                # "Add record" button
                add_btn = st.form_submit_button("Add record", type="primary")

            # Action when the Add button is clicked
            if add_btn:

                # copy the dataset from session state
                full = st.session_state.df.copy()

                # Assign new ID number which is one above the current maximum number
                next_id = int(full[ID_COL].max()) + 1 if len(full) else 1
                new_row[ID_COL] = next_id

                # Add new row at the end of the dataset
                full = pd.concat([full, pd.DataFrame([new_row])], ignore_index=True)

                # store the new dataset in session state
                st.session_state.df = ensure_unique_id(full, ID_COL)
                st.success("Record added.")

                log_action("ADD_RECORD", {"new_id": next_id, "fields": {k: str(v) for k, v in new_row.items() if k != ID_COL}})

                st.rerun()

        with tab_edit:
            st.subheader("Edit records inline (filtered view)")

            # if ID column is not in the filtered dataset return an error
            if ID_COL not in filtered.columns:
                st.error(f"Missing ID column '{ID_COL}'. Set ID_COL to your key or let it be created.")

            # Otherwise store columns to display and create a data editor with them
            else:
                display_cols = [c for c in f["visible_cols"] if c in filtered.columns]
                if ID_COL not in display_cols:
                    display_cols = [ID_COL] + display_cols

                # Define a data editor object
                edited = st.data_editor(
                    filtered[display_cols].copy(),
                    disabled=[ID_COL],
                    width="stretch",
                    hide_index=True,
                    num_rows="fixed",
                    key="editor_on_page",
                )

                # When Apply edits is pressed apply changes
                if st.button("Apply edits", type="primary"):

                    # create a copy of dataset and set index to ID column
                    full = st.session_state.df.copy().set_index(ID_COL)

                    # set index to the edited rows in data editor dataset as well to ID
                    edited_idx = edited.set_index(ID_COL)

                    # which rows are in the full dataset and the edited
                    common_ids = full.index.intersection(edited_idx.index)

                    # which columns are in the full dataset and the edited
                    common_cols = [c for c in edited_idx.columns if c in full.columns and c != ID_COL]

                    # Update the dataset with edited rows from the edited data editor and reset index
                    full.loc[common_ids, common_cols] = edited_idx.loc[common_ids, common_cols]
                    st.session_state.df = full.reset_index()
                    st.success("Edits applied.")

                    log_action("APPLY_EDITS", {
                        "edited_rows": int(len(edited_idx)),
                        "edited_cols": common_cols,
                        "ids_updated_count": int(len(common_ids)),
                    })
                    st.rerun()


        with tab_delete:
            st.subheader("Delete records")

            # Error if ID column missing
            if ID_COL not in filtered.columns:
                st.error(f"Missing ID column '{ID_COL}'. Set ID_COL to your key or let it be created.")

            # otherwise display a multiselect with IDs
            else:
                delete_ids = st.multiselect(
                    f"Select {ID_COL} values to delete",
                    options=filtered[ID_COL].tolist(),
                    key="delete_ids",
                )

                cdel1, cdel2 = st.columns([1, 3])

                # Delete selected button
                with cdel1:
                    do_delete = st.button("Delete selected", type="secondary")

                # Warning about immediate deletion
                with cdel2:
                    st.warning("This removes rows immediately from session data.")

                # Action executed by Delete selected button
                if do_delete and delete_ids:

                    # Copy dataset
                    full = st.session_state.df.copy()

                    # Length of dataset before
                    before = len(full)

                    # Actual deletion
                    full = full[~full[ID_COL].isin(delete_ids)].copy()
                    st.session_state.df = ensure_unique_id(full, ID_COL)

                    # Display how many records deleted
                    st.success(f"Deleted {before - len(full)} record(s).")

                    log_action("DELETE_RECORDS", {
                        "count": len(delete_ids),
                        "ids": delete_ids[:200],
                    })
                    st.rerun()

        with tab_logs:

            st.subheader("Interface Log")

            n = st.number_input("Show last N entries", min_value=10, max_value=2000, value=200, step=10)
            logs = read_last_logs(int(n))

            if not logs:
                st.info("No logs yet.")
            else:
                show_dataframe(pd.DataFrame(logs), width="stretch", hide_index=True)


            # Convert to a table
            log_df = read_log_as_dataframe()

            if log_df.empty:
                st.warning("Log file does not exist yet or is empty.")
            else:
                csv_bytes = log_df.to_csv(index=False).encode("utf-8")

                # Download as csv file
                st.download_button(
                    label="Download Interface Log as CSV",
                    data=csv_bytes,
                    file_name="interface_log.csv",
                    mime="text/csv",
                    key="download_audit_log_csv",
                )

if __name__ == "__main__":
    main()
