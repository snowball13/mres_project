"""Simulate a periodic current along a western boundary, with significantly
larger velocities along the western edge than the rest of the region

The original test description can be found in: N. Fabbroni, 2009,
Numerical Simulation of Passive tracers dispersion in the sea,
Ph.D. dissertation, University of Bologna
http://amsdottorato.unibo.it/1733/1/Fabbroni_Nicoletta_Tesi.pdf
"""

from firedrake import *

# We need a mesh - lets use a nxn element square, n=200
n = 200
mesh = UnitSquareMesh(n, n)

# Define some other constants
A = 100
eps = 0.05
a = 10000
b = 10000
r = 1e-6
beta = 2e-11

# We also need a function space V over which to solve the problem
# We will use pw linear functions between elements
V = FunctionSpace(mesh, "Lagrange", 1)

# Test and "trial" functions
phi = TestFunction(V)
psi = TrialFunction(V)

# Declare and define f (the RHS of the eqn), a function over V
# When using interpolate(), we just pass the analytic expression.
f = Function(V)
f.interpolate(Expression("-sin(pi*x[1])"))

# Define the integrands of the weak form (bilinear and linear forms resp.)
c = (-eps * dot(grad(phi), grad(psi)) + phi * grad(psi)[0]) * dx
L = f * phi * dx

# Redefine psi to hold the solution
psi = Function(V)

# We can now solve the equation...
bc = DirichletBC(V, 0, 'on_boundary')
solve(c == L, psi, bcs=[bc])

# Output to file
File("stommel.pvd").write(psi)

# Plot stream function psi
try:
    import matplotlib.pyplot as plt
except:
    warning("Matplotlib not imported")

try:
    plot(psi, contour=True)
    plt.show()
except Exception as e:
    warning("Cannot plot, error msg: '%s'" % e.message)
