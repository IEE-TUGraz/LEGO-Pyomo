import os
import sqlite3

import pandas as pd
import pyomo.core.base.set
import pyomo.environ as pyo

from InOutModule.printer import Printer

printer = Printer.getInstance()


def model_to_sqlite(model: pyo.base.Model, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    cnx = sqlite3.connect(filename)

    for o in model.component_objects():
        match type(o):
            case pyomo.core.base.set.OrderedScalarSet:
                df = pd.DataFrame(o.data())
            case pyomo.core.base.var.IndexedVar | pyomo.core.base.param.IndexedParam | pyomo.core.base.param.ScalarParam:
                indices = [str(i) for i in o.index_set().subsets()]
                df = pd.DataFrame(pd.Series(o.extract_values()), columns=['values'])
                if len(indices) == len(df.index.names):
                    if len(indices) > 1:
                        df = df.reset_index().rename(columns={f"level_{i}": b for i, b in enumerate(indices)})
                    else:
                        df = df.reset_index().rename(columns={"index": indices[0]})
                    df = df.set_index(indices)
            case pyomo.core.base.objective.ScalarObjective:
                df = pd.DataFrame([pyo.value(o)], columns=['values'])
            case pyomo.core.base.constraint.ConstraintList:  # Those will not be saved by decision
                continue
            case _:
                printer.error(f"Pyomo-Type {type(o)} not implemented, {o.name} will not be saved to SQLite")
                continue
        df.to_sql(o.name, cnx, if_exists='replace')
        cnx.commit()
    cnx.close()
    pass
