# Python script to run firedrake to solve Burger's eqn:
# u_t + u.grad(u) - nu*grad^2(u) = 0 on Omega, nu constant
# n.grad(u) = 0 on Gamma (boundry)

# We seek a solution over a function (vector-valued) space V.

# Dot with test function v in V and integrate by parts:
# integral(v.u_t + v.(u.grad(u)) + nu*(grad(v).grad(u))) = 0

# Distretise in time (use backward Euler)
# integral(v.(u^(n+1) - u^(n))/Deltat + v.(u^(n+1).grad(u^(n+1)))
#                   + nu*(grad(v).grad(u^(n+1)))) = 0



from firedrake import *

# We need a mesh - lets use a nxn element square, with n = 30 (so Deltax = 1/30)
n = 30
mesh = UnitSquareMesh(n, n)

# For our vector space V, we choose deg 2 Lagrange polynomials (and pw linear
# space for output)
V = VectorFunctionSpace(mesh, "CG", 2)
V_out = VectorFunctionSpace(mesh, "CG", 1)

# Define test function, and solution functions for current and next time steps
# Note that as the eqn is nonlinear, we do not define trial functions.
v = TestFunction(V)
u_t = Function(V, name="Velocity")
u_tnext = Function(V, name="VelocityNext")

# Specify ICs
ic = project(Expression(["sin(pi*x[0])", 0]), V)

# We set the current and next u to the initial conditions
u_t.assign(ic)
u_tnext.assign(ic)

# Set nu, the constant scalar viscocity (some small value)
nu = 0.0001

# Specify a time step Deltat, to use. We choose it so the the Courant number
# c is 1, to guarantee stability and good temporal resolution.
timestep = 1.0/n

# Define the residual of the equation.
F = (
    inner((u_tnext - u_t)/timestep, v) +
    inner(dot(u_tnext, nabla_grad(u_tnext)), v) +
    nu * inner(grad(u_tnext), grad(v))
    ) * dx

# Output file for visualisation
outfile = File("burgers.pvd")

# Begin by writing the current u_tnext (the IC)
outfile.write(project(u_tnext, V_out, name="Velocity"))

# We now loop over timesteps solving at each and writing to file
t = 0.0
end = 0.5
while t <= end:
    solve(F == 0, u_tnext)
    u_t.assign(u_tnext)
    t += timestep
    outfile.write(project(u_tnext, V_out, name="Velocity"))
