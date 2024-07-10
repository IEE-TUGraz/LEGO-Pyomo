# Transport problem example (based on https://github.com/Pyomo/PyomoGallery/blob/master/transport/transport.ipynb)
import pyomo.environ as pyo

model = pyo.ConcreteModel()

# Sets
model.i = pyo.Set(initialize=['seattle', 'san-diego'], doc='Canning plans')
model.j = pyo.Set(initialize=['new-york', 'chicago', 'topeka'], doc='Markets')

# Parameters
model.a = pyo.Param(model.i, initialize={'seattle': 350, 'san-diego': 600}, doc='Capacity of plant i in cases')
model.b = pyo.Param(model.j, initialize={'new-york': 325, 'chicago': 300, 'topeka': 275},
                    doc='Demand at market j in cases')

dtab = {
    ('seattle', 'new-york'): 2.5,
    ('seattle', 'chicago'): 1.7,
    ('seattle', 'topeka'): 1.8,
    ('san-diego', 'new-york'): 2.5,
    ('san-diego', 'chicago'): 1.8,
    ('san-diego', 'topeka'): 1.4,
}
model.d = pyo.Param(model.i, model.j, initialize=dtab, doc='Distance in thousands of miles')

model.f = pyo.Param(initialize=90, doc='Freight in dollars per case per thousand miles')


def c_init(model, i, j):
    return model.f * model.d[i, j] / 1000
model.c = pyo.Param(model.i, model.j, initialize=c_init, doc='Transport cost in thousands of dollar per case')

# Variables
model.x = pyo.Var(model.i, model.j, bounds=(0.0, None), doc='Shipment quantities in case')


# Constraints
def supply_rule(model, i):
    return sum(model.x[i, j] for j in model.j) <= model.a[i]
model.supply = pyo.Constraint(model.i, rule=supply_rule, doc='Observe supply limit at plant i')


def demand_rule(model, j):
    return sum(model.x[i, j] for i in model.i) >= model.b[j]
model.demand = pyo.Constraint(model.j, rule=demand_rule, doc='Satisfy demand at market j')


# Objective
def objective_rule(model):
    return sum(model.c[i, j] * model.x[i, j] for i in model.i for j in model.j)
model.objective = pyo.Objective(rule=objective_rule, sense=pyo.maximize, doc='Define objective function')


def pyomo_postprocess(options=None, instance=None, results=None):
    model.x.display()


# This is an optional code path that allows the script to be run outside of
# pyomo command-line.  For example:  python transport.py
if __name__ == '__main__':
    # This emulates what the pyomo command-line tools does
    from pyomo.opt import SolverFactory

    opt = SolverFactory("gurobi")
    results = opt.solve(model)
    # sends results to stdout
    results.write()
    print("\nDisplaying Solution\n" + '-' * 60)
    pyomo_postprocess(None, model, results)

print("Done")
