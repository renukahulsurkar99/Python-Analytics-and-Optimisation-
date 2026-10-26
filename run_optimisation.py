"""Solve and export the four self-contained coursework scheduling scenarios."""

from contextlib import redirect_stdout
import json
from pathlib import Path
import runpy

import pandas as pd


def main():
    root = Path(__file__).resolve().parent
    output = root / "outputs" / "optimisation"
    logs = root / "local_outputs" / "logs"
    output.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    summaries = []
    for filename in ["task1_cost.py", "task2_minimax.py", "task2_range.py", "task3_skills.py"]:
        with (logs / (Path(filename).stem + ".log")).open("w") as log, redirect_stdout(log):
            values = runpy.run_path(str(root / "optimisation" / filename))
        model = values["model1"]
        if model.Status != 2:
            raise RuntimeError(f"{filename}: solver did not return an optimal schedule.")
        rows = [
            {"operator": name, "day": day, "hours": int(round(values["h"][name, day].X)),
             "hourly_cost": values["wages"][name]}
            for name in values["names"] for day in values["days"]
        ]
        schedule = pd.DataFrame(rows)
        schedule["cost"] = schedule["hours"] * schedule["hourly_cost"]
        if not schedule.groupby("day")["hours"].sum().eq(14).all():
            raise RuntimeError("Daily staffing hours do not reconcile.")
        for row in rows:
            if row["hours"] > values["Availability"][row["operator"], row["day"]]:
                raise RuntimeError("Schedule exceeds operator availability.")
        weekly = schedule.groupby("operator")["hours"].sum()
        minimums = [7 if name in ["R.Perez", "C.Santos"] else 8 for name in weekly.index]
        if not (weekly.to_numpy() >= minimums).all():
            raise RuntimeError("A weekly commitment is not met.")
        if "programming_skill" in values:
            for skills in [values["programming_skill"], values["troubleshooting_skill"]]:
                coverage = (schedule["hours"] * schedule["operator"].map(skills)).groupby(schedule["day"]).sum()
                if not coverage.ge(6).all():
                    raise RuntimeError("Daily skill coverage is not met.")
        schedule.to_csv(output / (Path(filename).stem + "_schedule.csv"), index=False)
        summaries.append({
            "scenario": Path(filename).stem,
            "optimal": True,
            "weekly_total_hours": int(schedule["hours"].sum()),
            "total_cost": int(schedule["cost"].sum()),
            "maximum_operator_hours": int(weekly.max()),
            "minimum_operator_hours": int(weekly.min()),
            "weekly_hours_range": int(weekly.max() - weekly.min()),
        })
    result = pd.DataFrame(summaries)
    result.to_csv(output / "scenario_summary.csv", index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
