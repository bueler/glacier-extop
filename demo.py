# demo derived from
#   https://www.firedrakeproject.org/external_operators.html

from firedrake import *
import numpy as np


class TranslationOperator(AbstractExternalOperator):
    def __init__(self, *operands, function_space, **kwargs):
        AbstractExternalOperator.__init__(
            self, *operands, function_space=function_space, **kwargs
        )

    @assemble_method(0, (0,))
    def assemble_N(self, *args, **kwargs):
        """Evaluate N"""
        u, f = self.ufl_operands
        N = assemble(u - f)
        return N

    @assemble_method((1, 0), (0, 1))
    def assemble_Jacobian(self, *args, **kwargs):
        """Evaluate the Jacobian of N"""
        dNdu = Function(self.function_space()).assign(1)

        # Construct the Jacobian matrix
        integral_types = set(["cell"])
        assembly_opts = kwargs.get("assembly_opts")
        J = self._matrix_builder((), assembly_opts, integral_types)
        with dNdu.dat.vec as vec:
            J.petscmat.setDiagonal(vec)
        return J

    @assemble_method((1, 0), (0, None))
    def assemble_Jacobian_action(self, *args, **kwargs):
        """Evaluate the action of the Jacobian"""
        w = self.argument_slots()[-1]
        return w

mesh = UnitSquareMesh(20, 20)
V = FunctionSpace(mesh, "CG", 1)
x, y = SpatialCoordinate(mesh)
f = Function(V).interpolate((1 + 2 * pi ** 2) * sin(pi * x) * sin(pi * y))

# Assemble N on its own and check it.
u = Function(V).interpolate(cos(x * y))
N = TranslationOperator(u, f, function_space=V)
assembled_N = assemble(N)
assert np.allclose(assembled_N.dat.data_ro, u.dat.data_ro[:] - f.dat.data_ro[:])

for j in range(2):
    u = Function(V, name="u")
    v = TestFunction(V)
    bcs = DirichletBC(V, 0, "on_boundary")

    N = TranslationOperator(u, f, function_space=V)
    F = (inner(grad(u), grad(v)) + inner(N, v)) * dx

    sp = {"snes_converged_reason": None,
          #"snes_view": None,
          "snes_monitor": None}
    if j == 0:
        print("using assembled Jacobian and direct linear solver ...")
    else:
        print("using matrix-free Jacobian and plain CG ...")
        mf = {"mat_type": "matfree",
              "ksp_type": "cg",
              "ksp_converged_reason": None,
              "pc_type": "none"}
        sp.update(mf)
    solve(F == 0, u, bcs=bcs, solver_parameters=sp)
    
    # The exact solution is sin(pi x) sin(pi y).
    u_exact = sin(pi * x) * sin(pi * y)
    print(f"L2 error: {errornorm(u_exact, u):.3e}")
    uerr = Function(V, name="u_error").interpolate(u_exact - u)

    VTKFile(f"foo{j}.pvd").write(u, uerr)
