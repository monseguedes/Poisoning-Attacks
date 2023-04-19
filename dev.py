# -*- coding: utf-8 -*-

"""Description of this file"""

import numpy as np
import gurobipy
import pyomo.environ as pyo
import pyomo.kernel as pmo


class Mul:

    def __init__(self, model):
        self.model = model
        self.dct = dict()

    def __call__(self, a, b):
        u, v = (a, b) if a.index < b.index else (b, a)
        key = (u.index, v.index)
        if key in self.dct:
            return self.dct[key]
        x = self.model.addVar(name=f"{u.VarName}&{v.VarName}", lb=-float('inf'))
        self.model.addConstr(x == u * v)
        self.dct[key] = x
        return x


class GRB:

    def __init__(self):
        self.env = gurobipy.Env(params={"LogToConsole": False})
        self.model = gurobipy.Model(env=self.env)
        self.model.params.NonConvex = 2
        mul = Mul(self.model)
        self.x = self.model.addVar(name="x", lb=-1, ub=10)
        self.y = self.model.addVar(name="y", lb=-1, ub=10)
        self.z = self.model.addVar(name="z", lb=-1, ub=10)
        self.model.update()

        self.p = 10

        self.model.addConstr(dev(self, mul, 2))
        self.model.addConstr(dev(self, mul, 3))

        self.model.update()

        self.model.setObjective(obj(self))

    def display(self):
        self.model.display()

    def solve(self):
        self.model.optimize()

    def get_objective_value(self):
        return self.model.ObjVal

    def get_solution(self):
        return np.array([
            self.x.X,
            self.y.X,
            self.z.X,
        ])


class PYO(pmo.block):

    def __init__(self, solver):
        super().__init__()
        self.solver = solver
        self.x = pmo.variable(lb=-1, ub=10)
        self.y = pmo.variable(lb=-1, ub=10)
        self.z = pmo.variable(lb=-1, ub=10)

        self.p = pmo.parameter(10)

        if solver == "gurobi":
            dct = dict()
            self.vars = pmo.variable_list()
            self.cons = pmo.constraint_list()

            def mul(a, b):
                u, v = (a, b) if id(a) < id(b) else (b, a)
                key = (id(u), id(v))
                if key in dct:
                    return dct[key]
                x = pmo.variable()
                self.vars.append(x)
                self.cons.append(pmo.constraint(x == u * v))
                dct[key] = x
                return x

        else:
            def mul(x, y):
                return x * y

        self.con1 = pmo.constraint(dev(self, mul, 2))
        self.con2 = pmo.constraint(dev(self, mul, 3))

        self.objective = pmo.objective(obj(self))

    def solve(self):
        if self.solver == "ipopt":
            opt = pyo.SolverFactory("ipopt")
        else:
            opt = pyo.SolverFactory("gurobi", solver_io="python")
            opt.options["NonConvex"] = 2
        opt.solve(self, load_solutions=True, tee=False)

    def get_objective_value(self):
        return pyo.value(self.objective)

    def get_solution(self):
        return np.array([
            self.x.value, self.y.value, self.z.value
        ])


def obj(solver):
    return solver.x + solver.y + solver.z


def dev(solver, mul, coef):
    return coef * mul(solver.x, solver.y) * solver.z - 2 * solver.y - solver.p >= 0
    # return coef * solver.x * solver.y * solver.z - 2 * solver.y - solver.p >= 0

# def dev(solver, mul, coef):
#     return coef * mul(solver.x, solver.y) - 2 * solver.z - solver.p >= 0

# def dev(solver, mul, coef):
#     return (coef - 1) ** 2 * solver.x + coef * solver.y  + solver.z - 4 >= 0


def main():
    """Run the main routine of this script"""
    # solver1 = GRB()
    # # solver1.display()
    # solver1.solve()
    # print(solver1.get_objective_value())
    # print(solver1.get_solution())

    for solver in ["ipopt", "gurobi"]:
        print(f"{solver=}")
        print()
        solver2 = PYO(solver)
        # solver2 = PYO("ipopt")
        solver2.solve()
        print('first problem')
        print(solver2.get_objective_value())
        print(solver2.get_solution())
        print()

        print('second problem')
        solver2.p.value = 8
        solver2.solve()
        print(solver2.get_objective_value())
        print(solver2.get_solution())
        print()

        print('third problem')
        solver2.x.fix(2)
        solver2.solve()
        print(solver2.get_objective_value())
        print(solver2.get_solution())
        print()


if __name__ == "__main__":
    main()
