
import gurobipy as gp
from gurobipy import GRB
import pandas as pd

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

#now we have to add in our constraints
model1.addConstrs((h[n,d] <= y[n,d] * Availability[n,d] for d in days for n in names), name = "Avaliability")

#next constraint is the fact that bachelor's students have different committments and masters students also have different time committmnets
#bachelors have minimum weekly committment of 8 hours
#masters students have minimum weekly committment of 7 hours

Min_hours_EKhan = 8
min_weekly = {"E.Khan": 8, "Y.Chen": 8,"A.Taylor": 8,"R.Zidane": 8, "R.Perez": 7, "C.Santos": 7 }

model1.addConstrs((gp.quicksum(h[n, d]for d in days) >= min_weekly[n] for n in names), name= "minimum weekly committment")

#include constraint about the plant only running for 14 hours on a given day from 8-12
model1.addConstr(gp.quicksum(h[w, "Monday"] for w in names) == 14,name="Monday_Capacity")
model1.addConstr(gp.quicksum(h[w, "Tuesday"] for w in names) == 14,name="Tuesday_Capacity")
model1.addConstr(gp.quicksum(h[w, "Wednesday"] for w in names) == 14,name="Wednesday_Capacity")
model1.addConstr(gp.quicksum(h[w, "Thursday"] for w in names) == 14,name="Thursday_Capacity")
model1.addConstr(gp.quicksum(h[w, "Friday"] for w in names) == 14,name="Friday_Capacity")

#add a constraint that checks the differences between the students below a certain number
#this is the availability constraint
model1.addConstrs(
     (h[n, d] <= Availability[n, d] for n in names for d in days),
        name="availability"
    )
#now to define our objective function in terms of wages, because its a minimisation problem total cost = wages * hours worked for a particular person

model1.setObjective(gp.quicksum(wages[n] * h[n,d] for d in days for n in names), sense= GRB.MINIMIZE)

model1.optimize()
status_code = model1.Status


day = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
for d in range(len(day)):
    for n in names:
        if h[n, day[d]].X > 0:
            print(n, "works", h[n, day[d]].X, "hours on", day[d])

for n in names:
        total_hours = sum(h[n, d].X for d in days)
        print(f"{n} works {total_hours} hours in total")
