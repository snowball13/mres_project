""" Python script to run firedrake for a simple example of the Helmoltz eqn:
    -grad^2(u) + u = f on Omega, with BC grad(u).n = 0 on Gamma (boundry)

In weak form, this is (for suitable function space V):
integral(grad(u).grad(v) + uv) = integral(fv)
where v is any test function in V.

Here, we consider in 2D with f = (1.0 + 8.0pi^2)cos(2pix)cos(2piy)

Note the analytic solution is u(x,y) = cos(2pix)cos(2piy)
"""


from firedrake import *

# We need a mesh - lets use a 10x10 element square
mesh = UnitSquareMesh(10, 10)

# We also need a function space V over which to solve the problem
# We will use pw linear functions between elements
V = FunctionSpace(mesh, "CG", 1)

# Test and "trial" functions
v = TestFunction(V)
u = TrialFunction(V)

# Declare and define f (the RHS of the eqn), a function over V
# When using interpolate(), we just pass the analytic expression.
f = Function(V)
f.interpolate(Expression("(1+8*pi*pi)*cos(2*pi*x[0])*cos(2*pi*x[1])"))

# Define the integrands of the weak form (bilinear and linear forms resp.)
a = (dot(grad(v), grad(u)) + u * v) * dx
L = f * v * dx

# We can now solve the equation...

# Redefine u to hold the solution
u = Function(V)

# Using PETSc emplying the CG alg (as Helmoltz eqn is symmetric) we solve
solve(a == L, u, solver_parameters={'ksp_type': 'cg'})

# Output to file
File("helmoltz.pvd").write(u)

# Plot using built-in firedrake plotting
try:
    import matplotlib.pyplot as plt
except:
    warning("Matplotlib not imported")

try:
    plot(u, contour=True)
    plt.show()
except Exception as e:
    warning("Cannot plot, error msg: '%s'" % e.message)

# We can also check the L2 error
f.interpolate(Expression("cos(2*pi*x[0])*cos(2*pi*x[1])"))
print sqrt(assemble(dot(u - f, u - f) * dx))
