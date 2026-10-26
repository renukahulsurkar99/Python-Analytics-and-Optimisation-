import gurobipy as gp
import pandas as pd
from gurobipy import GRB

def solve_task2():

    model1 = gp.Model("Optimisation model") #this creates our model for us

    #next would be to create two libraries/ dictionaries of indexes which are time and the names

    names = ["E.Khan", "Y.Chen", "A.Taylor", "R.Zidane", "R.Perez", "C.Santos"]
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    #then create a dictionary of the availability of everyone, included in the case study

    Availability = { ("E.Khan", "Monday"): 6 , ("E.Khan", "Tuesday"): 0, ("E.Khan", "Wednesday"): 6, ("E.Khan", "Thursday"): 0, ("E.Khan", "Friday"): 6,
                ("Y.Chen", "Monday"): 0 , ("Y.Chen", "Tuesday"): 6, ("Y.Chen", "Wednesday"): 0, ("Y.Chen", "Thursday"): 6, ("Y.Chen", "Friday"): 0,
                ("A.Taylor", "Monday"): 4 , ("A.Taylor", "Tuesday"): 8, ("A.Taylor", "Wednesday"): 4, ("A.Taylor", "Thursday"): 0, ("A.Taylor", "Friday"): 4,
                ("R.Zidane", "Monday"): 5 , ("R.Zidane", "Tuesday"): 5, ("R.Zidane", "Wednesday"): 5, ("R.Zidane", "Thursday"): 0, ("R.Zidane", "Friday"): 5,
                ("R.Perez", "Monday"): 3 , ("R.Perez", "Tuesday"): 0, ("R.Perez", "Wednesday"): 3, ("R.Perez", "Thursday"): 8, ("R.Perez", "Friday"): 0,
                ("C.Santos", "Monday"): 0 , ("C.Santos", "Tuesday"): 0, ("C.Santos", "Wednesday"): 0, ("C.Santos", "Thursday"): 6, ("C.Santos", "Friday"): 2
    }
    #lets also create a dictionary with everyone's wages to compute our objective function

    wages = { "E.Khan" : 25, "Y.Chen": 26, "A.Taylor": 24, "R.Zidane": 23, "R.Perez": 28, "C.Santos": 30}


    #now lets set up our decision variables

    h = model1.addVars(names, days, lb=0.0, vtype = GRB.INTEGER, name= "hours worked")
    y = model1.addVars(names, days, vtype= GRB.BINARY, name = "present")

    Min_hours_EKhan = 8
    model1.addConstr(gp.quicksum((h["E.Khan", d]for d in days)) >= Min_hours_EKhan, name= "Bachelor's minimum committment")
    Min_hours_YChen = 8
    model1.addConstr(gp.quicksum((h["Y.Chen", d]for d in days)) >= Min_hours_YChen, name= "Bachelor's minimum committment")
    Min_hours_ATaylor = 8
    model1.addConstr(gp.quicksum((h["A.Taylor", d]for d in days)) >= Min_hours_ATaylor, name= "Bachelor's minimum committment")
    Min_hours_RZidane = 8
    model1.addConstr(gp.quicksum((h["R.Zidane", d]for d in days)) >= Min_hours_RZidane, name= "Bachelor's minimum committment")
    Min_hours_Rperez = 7
    model1.addConstr(gp.quicksum((h["R.Perez", d]for d in days)) >= Min_hours_Rperez, name= "Bachelor's minimum committment")
    Min_hours_Csantos = 7
    model1.addConstr(gp.quicksum((h["C.Santos", d]for d in days)) >= Min_hours_Csantos, name= "Bachelor's minimum committment")

    #include constraint about the plant only running for 14 hours on a given day from 8-12
    model1.addConstr(gp.quicksum(h[w, "Monday"] for w in names) == 14,name="Monday_Capacity")
    model1.addConstr(gp.quicksum(h[w, "Tuesday"] for w in names) == 14,name="Tuesday_Capacity")
    model1.addConstr(gp.quicksum(h[w, "Wednesday"] for w in names) == 14,name="Wednesday_Capacity")
    model1.addConstr(gp.quicksum(h[w, "Thursday"] for w in names) == 14,name="Thursday_Capacity")
    model1.addConstr(gp.quicksum(h[w, "Friday"] for w in names) == 14,name="Friday_Capacity")


    Hmax = model1.addVar(lb=0.0, vtype=GRB.INTEGER, name="Hmax")

    model1.addConstrs(
        (gp.quicksum(h[n, d] for d in days) <= Hmax for n in names),
        name="max_weekly_hours_link"
    )
    #new constraint considering availability
    model1.addConstrs(
        (h[n, d] <= Availability[n, d] for n in names for d in days),
        name="availability"
    )

    #including skill based constraint, one user has got either programming or troubleshooting, someone has got both.
    programming_skill = {"E.Khan": 1, "Y.Chen": 1, "A.Taylor": 0, "R.Zidane": 0, "R.Perez": 1, "C.Santos": 1}
    troubleshooting_skill = {"E.Khan": 0, "Y.Chen": 0, "A.Taylor": 1, "R.Zidane": 1, "R.Perez": 0, "C.Santos": 1}

    model1.addConstrs((gp.quicksum(programming_skill[n]*h[n,d] for n in names) >= 6 for d in days), name= "programming present")
    model1.addConstrs((gp.quicksum(troubleshooting_skill[n]*h[n,d] for n in names) >= 6 for d in days), name= "programming present")
    #how to approach scenario 2
    Hmax = model1.addVar(lb=0.0, vtype=GRB.INTEGER, name="Hmax")
    # --- Weekly totals per person ---
    H = {n: gp.quicksum(h[n, d] for d in days) for n in names}

    # --- Range variables ---
    Hmax = model1.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="Hmax_weekly")
    Hmin = model1.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="Hmin_weekly")

    # Link weekly totals to min/max
    model1.addConstrs((H[n] <= Hmax for n in names), name="link_Hmax")
    model1.addConstrs((H[n] >= Hmin for n in names), name="link_Hmin")

    # Define range
    R = model1.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="Range")
    model1.addConstr(R == Hmax - Hmin, name="define_range")

    # ========== Stage 1: minimise range ==========
    model1.setObjective(R, GRB.MINIMIZE)
    model1.optimize()
    if model1.Status != GRB.OPTIMAL:
        return {
            "status_code": model1.Status,
            "total_cost": None,
            "schedule_df": None,
            "pivot_df": None,
            "weekly_df": None,
        }
    R_star = R.X
    print("Best range R* =", R_star)

    # ========== Stage 2: minimise cost subject to range optimal ==========
    model1.addConstr(R <= R_star + 1e-6, name="fix_R_star")

    cost_expr = gp.quicksum(wages[n] * h[n, d] for n in names for d in days)
    model1.setObjective(cost_expr, GRB.MINIMIZE)
    model1.optimize()

    status_code = model1.Status


    if status_code != GRB.OPTIMAL:
        return {
            "status_code": status_code,
            "total_cost": None,
            "schedule_df": None,
            "pivot_df": None,
            "weekly_df": None
        }

    total_cost = float(model1.ObjVal)

    # Build schedule DataFrame
    rows = []
    for n in names:
        for d in days:
            hrs = float(h[n, d].X)
            if hrs != 0:
                rows.append({
                    "Name": n,
                    "Day": d,
                    "Hours": hrs,
                    "Wage_per_hr": wages[n],
                    "Cost": hrs * wages[n]
                })

    schedule_df = pd.DataFrame(rows).sort_values(["Day", "Name"])
    pivot_df = schedule_df.pivot_table(index="Name", columns="Day", values="Hours", aggfunc="sum", fill_value=0)
    weekly_df = (
        schedule_df.groupby("Name", as_index=False)
        .agg(Weekly_Hours=("Hours", "sum"), Weekly_Cost=("Cost", "sum"))
        .sort_values("Weekly_Cost", ascending=False)
    )

    return {
        "status_code": status_code,
        "total_cost": total_cost,
        "schedule_df": schedule_df,
        "pivot_df": pivot_df,
        "weekly_df": weekly_df
    }
