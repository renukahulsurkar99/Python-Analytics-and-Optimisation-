import streamlit as st
import pandas as pd
from finalopt import solve_task2


DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]

MENU = {
    "1: Show the overall cost": "overall_cost",
    "2: Show the number of hours assigned to an operator": "op_hours",
    "3: Show the schedule on a particular day": "day_schedule",
    "4: Show the schedule of an operator": "op_schedule",
    "5: Show the overall schedule": "overall_schedule",
    "6: Show the stats/plots of a variable": "stats_plots",
}

def ensure_day_order(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure Day is an ordered categorical for correct sorting."""
    if df is None or df.empty or "Day" not in df.columns:
        return df
    out = df.copy()
    out["Day"] = pd.Categorical(out["Day"], categories=DAY_ORDER, ordered=True)
    return out


def task1_menu_panel(res: dict):
    st.subheader("Menu")

    choice_label = st.selectbox("Choose an option:", list(MENU.keys()))
    choice = MENU[choice_label]

    # --- Option 1: overall cost ---
    if choice == "overall_cost":
        cost = res.get("total_cost", None)
        st.metric("Overall cost (£)", "—" if cost is None else f"{cost:,.2f}")

    # --- Option 2: operator total hours ---
    elif choice == "op_hours":
        weekly_df = res.get("weekly_df")
        if weekly_df is None or weekly_df.empty:
            st.warning("No weekly totals available.")
            return

        operators = weekly_df["Name"].tolist()
        name = st.selectbox("Enter the name of the operator:", operators)

        hours = float(weekly_df.loc[weekly_df["Name"] == name, "Weekly_Hours"].iloc[0])
        st.success(f"The number of hours assigned to {name} is {hours:.0f} hours")

    # --- Option 3: schedule for a day ---
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
            st.dataframe(day_sched, width="stretch", hide_index=True)

    # --- Option 4: schedule for an operator ---
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
            st.dataframe(op_sched, width="stretch", hide_index=True)

    # --- Option 5: overall schedule matrix ---
    elif choice == "overall_schedule":
        pivot_df = res.get("pivot_df")
        if pivot_df is None or pivot_df.empty:
            st.warning("No schedule matrix available.")
            return

        # Ensure day columns ordered (critical)
        pivot_df = pivot_df.reindex(columns=DAY_ORDER)

        st.write("Overall schedule matrix (Hours)")
        st.dataframe(pivot_df, width="stretch")

        # Downloads (nice touch)
        st.download_button(
            "Download schedule matrix (CSV)",
            data=pivot_df.to_csv().encode("utf-8"),
            file_name="task1_schedule_matrix.csv",
            mime="text/csv",
        )

    # --- Option 6: stats/plots placeholder ---
    elif choice == "stats_plots":
        st.info("Stats/plots: tell me which variable you want (e.g., Hours, Cost) and I’ll add charts.")

def main():
    st.set_page_config(page_title="Skills-constrained scheduling", layout="wide")
    st.title("Skills-constrained scheduling")

    if "task1_results" not in st.session_state:
        st.session_state["task1_results"] = None

    if st.button("Solve scheduling optimisation", type="primary"):
        with st.spinner("Solving..."):
            st.session_state["task1_results"] = solve_task2()   # <-- real call

    #ALWAYS assign res, regardless of button click
    res = st.session_state["task1_results"]

    if res is None:
        st.warning("No results yet. Click **Solve scheduling optimisation**.")
        st.stop()

    task1_menu_panel(res)

if __name__ == "__main__":
    main()
