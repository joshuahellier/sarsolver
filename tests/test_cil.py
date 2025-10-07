import numpy as np
from cil.optimisation.algorithms import PDHG, CGLS
from cil.optimisation.functions import ZeroFunction, L2NormSquared
from pytest import approx


def test_direct(little_sar_operator, little_ground_truth):
    y = little_sar_operator.range.allocate()
    little_sar_operator.direct(little_ground_truth, y)


def test_adjoint(little_sar_operator, little_synthetic_measurement, little_backproj):
    y = little_sar_operator.domain.allocate()
    backproj = little_sar_operator.adjoint(little_synthetic_measurement, y)
    assert backproj.array == approx(little_backproj.array, rel=1.0E-3)


def test_cgls(little_sar_operator, little_synthetic_measurement):
    solver = CGLS(initial=little_sar_operator.domain.allocate(),
                  operator=little_sar_operator,
                  data=little_synthetic_measurement)
    solver.run(3)


def test_pdhg(little_sar_operator, little_soln, little_synthetic_measurement):
    f = L2NormSquared(b=little_synthetic_measurement)
    g = ZeroFunction()
    solver = PDHG(initial=little_sar_operator.domain.allocate(), operator=little_sar_operator, f=f, g=g)
    solver.run(3)
    print(f"Canned array: {little_soln.array}")
    print(f"Canned array: {solver.x.array}")
    print(f"Canned array: {little_soln.array - solver.x.array}")
    print(f"Error is {np.linalg.norm((little_soln.array - solver.x.array)).reshape([-1])}, rel is "
          f"{np.linalg.norm((little_soln.array - solver.x.array)).reshape([-1]) /
             np.linalg.norm(little_soln.array).reshape([-1])}")
    print()
    assert solver.x.array == approx(little_soln.array, rel=1.0E-3)
