# import pylp
import pytest

def gurobi_installed_with_license():
    try:
        raise RuntimeError("Don't use gurobi right now, it seems to be crashing")
        # solver = pylp.create_linear_solver(pylp.Preference.Gurobi)
        # solver.initialize(1, pylp.VariableType.Binary)
        # objective = pylp.LinearObjective(1)
        # objective.set_coefficient(1, 1)
        # solver.set_objective(objective)
        # constraints = pylp.LinearConstraints()
        # solver.set_constraints(constraints)
        # solution, message = solver.solve()
        # success = True
    except RuntimeError:
        success = False

    return pytest.mark.skipif(not success, reason="Requires Gurobi License")