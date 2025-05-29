from pyomo.environ import *
from pyomo.opt import SolverFactory

def solve_row_with_pyomo(row, solver_name="ipopt"):
    # Constants
    kappa_u = 1e-28
    kappa_e = 1e-28
    alpha = 0.5
    beta = 0.5
    latency_deadline = 10000
    B_cloud = 100

    # Build model
    model = ConcreteModel()
    model.offload = Var(domain=Binary)
    model.cache = Var(domain=Binary)

    # Latency
    def comm_latency(m):
        return m.offload * ((row["Task_Size"] / row["Bandwidth"]) +
                            (1 - m.cache) * (row["Task_Size"] / B_cloud))

    def comp_latency(m):
        return m.offload * (row["Task_Size"] * 1000 / row["CPU_Load"]) + \
               (1 - m.offload) * (row["Task_Size"] * 1000 / row["Local_CPU"])

    model.latency = Expression(rule=lambda m: comm_latency(m) + comp_latency(m))

    # Energy (choose linear or nonlinear based on solver)
    if solver_name in ["glpk", "gurobi", "cbc"]:
        # Approximate energy as fixed linear values (e.g., average energy)
        avg_comp_energy = 0.0001
        model.energy = Expression(rule=lambda m: m.offload * (row["Tx_Power"] * (row["Task_Size"] / row["Bandwidth"]) + avg_comp_energy) +
                                                (1 - m.offload) * avg_comp_energy)
    else:
        def tx_energy(m):
            return m.offload * row["Tx_Power"] * (row["Task_Size"] / row["Bandwidth"])

        def comp_energy(m):
            return m.offload * kappa_e * row["Task_Size"] * 1000 * row["CPU_Load"]**2 + \
                   (1 - m.offload) * kappa_u * row["Task_Size"] * 1000 * row["Local_CPU"]**2

        model.energy = Expression(rule=lambda m: tx_energy(m) + comp_energy(m))

    # Objective
    model.obj = Objective(expr=alpha * model.latency + beta * model.energy, sense=minimize)

    # Constraints
    model.latency_con = Constraint(expr=model.latency <= latency_deadline)
    model.energy_con = Constraint(expr=model.energy <= row["Residual_Energy"])

    # Solve
    solver = SolverFactory(solver_name)
    try:
        result = solver.solve(model)
        feasible = int(result.solver.termination_condition == TerminationCondition.optimal)
    except:
        return {"error": f"Solver {solver_name} failed or not installed."}

    # Return results
    return {
        "offload": round(value(model.offload)),
        "cache": round(value(model.cache)),
        "latency": value(model.latency),
        "energy": value(model.energy),
        "feasible": feasible
    }


# Quick test
if __name__ == "__main__":
    row = {
        "Bandwidth": 61,
        "CPU_Load": 4.943,
        "Task_Size": 33,
        "Mobility": 0.27,
        "Cache_Occupancy": 0.34,
        "Residual_Energy": 100.0,
        "Local_CPU": 1.709,
        "Tx_Power": 0.5
    }
    result = solve_row_with_pyomo(row, solver_name="ipopt")
    print(result)
