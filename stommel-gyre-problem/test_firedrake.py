# firedrake_import = False
# try:
#     from firedrake import *
#     firedrake_import = True
# except:
#     pass
from firedrake import *
from parcels import FiredrakeGrid, ParticleSet, ScipyParticle, JITParticle, Variable
from parcels import AdvectionRK4, AdvectionEE, AdvectionRK45
import numpy as np
import pytest
from datetime import timedelta as delta

ptype = {'scipy': ScipyParticle, 'jit': JITParticle}
method = {'RK4': AdvectionRK4, 'EE': AdvectionEE, 'RK45': AdvectionRK45}

def stommel_grid(xdim=200, ydim=200):
    """Simulate a periodic current along a western boundary, with significantly
    larger velocities along the western edge than the rest of the region

    The original test description can be found in: N. Fabbroni, 2009,
    Numerical Simulation of Passive tracers dispersion in the sea,
    Ph.D. dissertation, University of Bologna
    http://amsdottorato.unibo.it/1733/1/Fabbroni_Nicoletta_Tesi.pdf
    """

    # We begin by using firedrake to solve for the stream function and
    # velocities on our mesh

    eps = 0.05
    a = 1
    b = 1
    mesh = UnitSquareMesh(xdim, ydim)

    # We need a function space V over which to solve the problem
    # We will use pw linear functions between elements
    V = FunctionSpace(mesh, "CG", 2)

    # Test and "trial" functions
    phi = TestFunction(V)
    psi = TrialFunction(V)

    # Declare and define f (the RHS of the eqn), a function over V
    # When using interpolate(), we just pass the analytic expression.
    f = Function(V)
    f.interpolate(Expression("-sin(pi*x[1])/10000"))

    # Define the integrands of the weak form (bilinear and linear forms resp.)
    c = (-eps * dot(grad(phi), grad(psi)) + phi * grad(psi)[0] / a) * dx
    L = f * phi * dx

    # Redefine psi to hold the solution
    psi = Function(V, name="psi")

    # We can now solve the equation...
    bc = DirichletBC(V, 0, 'on_boundary')
    solve(c == L, psi, bcs=[bc])

    # Now, we compute the velocity field, u=(u,v)
    # where u=-psi_y, v=psi_x (partial derivatives)
    # We treat this as a simple finite element problem

    # Define the vector space over which to solve
    V = VectorFunctionSpace(mesh, "CG", 1)

    # Test and trial functions
    u = TrialFunction(V)
    v = TestFunction(V)

    # Define the RHS of our problem. This is gradperp(psi).
    gradperp = lambda u: as_vector((-u.dx(1), u.dx(0)))
    f = Function(V)
    f.interpolate(gradperp(psi))

    # Bilinear and linear forms
    c = (dot(v, u)) * dx
    L = (dot(v, f)) * dx

    # Solve
    u = Function(V)
    solve(c == L, u)

    # We now create the PARCELS grid, and use point evaluation to gain values
    # for the stream function and velocities at points in the domain

    # Coordinates of the test grid (on A-grid in deg)
    lon = np.linspace(0, a, xdim, dtype=np.float32)
    lat = np.linspace(0, b, ydim, dtype=np.float32)
    return FiredrakeGrid(u, lon, lat, fields=[psi])

def UpdateP(particle, grid, time, dt):
    particle.p = grid.psi.eval(time, particle.lon, particle.lat)

def stommel_example(npart=1, mode='scipy', method=AdvectionRK4):

    grid = stommel_grid()

    # Determine particle class according to mode
    ParticleClass = JITParticle if mode == 'jit' else ScipyParticle

    class MyParticle(ParticleClass):
        p = Variable('p', dtype=np.float32, initial=0.)
        p_start = Variable('p_start', dtype=np.float32, initial=0.)

    pset = ParticleSet.from_line(grid, size=npart, pclass=MyParticle,
                                 start=(0.01, 0.5), finish=(0.02, 0.5))
    for particle in pset:
        particle.p_start = grid.psi.eval(0., particle.lon, particle.lat)

    # Execute for 50 days, with 5min timesteps and hourly output
    runtime = delta(days=0.50)
    dt = delta(minutes=0.5)
    interval = delta(hours=0.12)
    pset.execute(method + pset.Kernel(UpdateP), runtime=runtime, dt=dt,
                 interval=interval, show_movie=False)

    return pset


@pytest.mark.parametrize('mode', ['scipy'])
def test_stommel_grid(mode):
    psetRK4 = stommel_example(1, mode=mode, method=method['RK4'])
    assert np.allclose([p.lon for p in psetRK4], rtol=1e-3)
    err_adv = np.array([abs(p.p_start - p.p) for p in psetRK4])
    assert(err_adv <= 1.e-1).all()
    err_smpl = np.array([abs(p.p - psetRK4.grid.psi.eval(0., p.lon, p.lat)) for p in psetRK4])
    assert(err_smpl <= 1.e-1).all()
