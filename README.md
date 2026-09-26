# Biceps - (B) Ice pStokes Solver
Biceps is a prognostic two-dimensional full-Stokes ice-sheet solver. Furthermore, it comes bundled with its own FEM libraries implemented using the numerical linear algebra library [Eigen3](https://eigen.tuxfamily.org/index.php?title=Main_Page). It is mainly intended for investigating numerical stability of viscous free-surface flows.

# Build instruction
In the following instructions, commands requiring elevated privileges are prepended with a '#', and commands that can be run as regular user with a '$'.
## Linux (Ubuntu/Debian)

### C++
This project has rather few dependencies; a minimal C++ installation requires only a working C++ compiler and tool chain (e.g., [gcc](https://gcc.gnu.org/)), [CMake](https://cmake.org/), [boost](https://www.boost.org/), and [Eigen3](https://eigen.tuxfamily.org/index.php?title=Main_Page):

```console
# apt install gcc build-essential cmake libboost-dev libeigen3-dev
```

```console
$ cmake .. -DCMAKE_BUILD_TYPE=Release
```

### Python

Building the python interface requires [EigenPy](https://github.com/stack-of-tasks/eigenpy) and the python module of [boost](https://www.boost.org/). [EigenPy](https://github.com/stack-of-tasks/eigenpy) in turn depends on [NumPy](https://numpy.org/) and [SciPy](https://scipy.org/):

```console
# apt install python3-numpy python3-scipy libboost-python-dev
```

After installing dependencies for [EigenPy](https://github.com/stack-of-tasks/eigenpy), grab the latest release from [here](https://github.com/stack-of-tasks/eigenpy/archive/refs/tags/v3.10.3.tar.gz) and extract it:

```console
$ tar -xvzf v3.10.3.tar.gz && cd eigenpy-3.10.3
```

Then configure and compile (on all available cores):

```console
$ mkdir -p .build && cd .build && cmake .. -DCMAKE_BUILD_TYPE=release && make -j$(cat /proc/cpuinfo | grep "core id" | sort | uniq | wc -l)
```

Finally, to install run:

```console
# make install
```

To build Biceps with Python enabled, configure CMake:

```console
$ cmake .. -DCMAKE_BUILD_TYPE=Release -DENABLE_PYTHON=ON
```

### Documentation

Generating documentation requires [Doxygen](https://www.doxygen.nl/) and [Graphviz](https://graphviz.org/):

```console
# apt install doxygen graphviz
```

To build Biceps documentation, configure CMake:
```console
$ cmake .. -DCMAKE_BUILD_TYPE=Release -DENABLE_DOCS=ON
```

### Unit testing
Unit testing is performed using the boost unit test module, which can be installed by running:

```console
# apt install libboost-test-dev
```

To build Biceps with testing enabled, configure CMake:
```console
$ cmake .. -DCMAKE_BUILD_TYPE=Release -DENABLE_TESTS=ON
```


### Example build

The steps for installing this repository is:

1. Clone repository:
```console
$ git clone https://github.com/andrelofgrenSU/biceps.git && cd biceps
```

2. Configure and compile (on all available cores):
```console
$ mkdir -p .build && cd .build && cmake .. -DCMAKE_BUILD_TYPE=release -DENABLE_PYTHON=ON -DENABLE_TESTS=ON -DENABLE_DOCS=ON -DUSE_LONG_DOUBLE=OFF && make -j$(cat /proc/cpuinfo | grep "core id" | sort | uniq | wc -l)
```

3. Build documentation (optional):
```console
$ make docs
```

4. Run tests (optional):
```console
$ make test
```

5. Install:
```console
# make install
```

# Theory

## The pStokes equations

### Strong formulation
The velocity- and pressure distribution, denoted $\mathbf{u}$ and $p$ respectively, inside the ice sheet is governed by the Stokes equation (a simplification of the Navier-Stokes equation, valid only for viscous dominated flows), which consists of the momentum balance and an incompressibility condition:

```math
\begin{aligned}
    \nabla \cdot (2 \eta(\mathbf{u}) \dot{\varepsilon}(\mathbf{u})) - \nabla p + \mathbf{f} &= \mathbf{0}, \quad \mathbf{x} \in \Omega, \\
    \nabla \cdot \mathbf{u} &= 0, \quad \mathbf{x} \in \Omega.
\end{aligned}
```

Here $\dot{\varepsilon}(\mathbf{u}) = \frac{1}{2} \left ( \nabla \mathbf{u} + \nabla{\mathbf{u}}^T \right )$ is the strain-rate tensor, $\mathbf{f}$ is the volumetric external body force acting on each fluid element (gravity in case of ice sheets and glaciers). Furthermore, ice is a shear thinning fluid with the viscosity function $\eta$ following a power-law rheology known as Glen's flow law

```math
\eta(\mathbf{u}) = A^{\frac{1}{n}} \left (\dot{\varepsilon}^2_e (\mathbf{u}) + \dot{\varepsilon}^2_0 \right)^{\frac{1-n}{2n}}.
```

Here $A$ is the so-called rate factor or ice softness parameter, $n \approx 3$ is the glen exponent, and $\dot{\varepsilon}_e$ is the effective strain rate

```math
\dot{\varepsilon}^2_e = \frac{1}{2} \left (\text{tr} \left (\dot{\varepsilon}^2 \right ) - \text{tr} \left (\dot{\varepsilon} \right )^2 \right).
```

In addition, a small regularization term $\varepsilon_0$ is included to prevent infinite viscosity at zero effective strain rate.

This type of power-law viscosity coupled Stokes equation is referred to as the pStokes equation.

### Boundary conditions
For wellposedness of the pStokes equation, boundary conditions needs to be specified on all parts of the boundary. Specifically, for grounded ice sheets the following are typically imposed:

1. Stress free condition on the ice-atmosphere interface $\Gamma_s$: $\sigma \hat{\mathbf{n}} = \mathbf{0}$
2. No-slip on the bedrock $\Gamma_b$: $\mathbf{u} \cdot \hat{\mathbf{n}} = 0$
3. Impenetrability on the lateral boundaries $\Gamma_E$ and $\Gamma_W$: $\mathbf{u} \cdot \hat{\mathbf{n}} = 0$

These boundary conditions can, however, be used interchangeably on all boundary parts.

### Weak formulation
The pStokes equations are solved using the finite element method (FEM), which discretizes the weak formulation. To state the weak form, the momentum equation and the incompressibility condition are multiplied by test functions $\mathbf{v} \in \mathcal{V}$ and $q \in \mathcal{Q}$, respectively, and then integrating over the domain $\Omega$. This results in the following weak formulation:

Find $\mathbf{u} \in \mathcal{U}$ and $p \in \mathcal{Q}$, such that

```math
\left (\dot{\varepsilon}(\mathbf{v}), 2 \eta(\mathbf{u}) \dot{\varepsilon}(\mathbf{u}) \right )_{\Omega} - (\nabla \cdot \mathbf{v}, p)_{\Omega} - (q, \nabla \cdot \mathbf{u})_{\Omega} = (\mathbf{v}, \mathbf{f})_{\Omega}
```

for all $\mathbf{v} \in \mathcal{V}$ and all $q \in \mathcal{Q}$. Here $\mathcal{U}, \mathcal{V}$ and $\mathcal{Q}$ are appropriate Sobolev spaces, in particular the discretized trial spaces $\mathcal{U}_h \subset \mathcal{U}$ and $\mathcal{Q}_h \subset \mathcal{Q}$ should be chosen so that they satisfy a discrete *inf-sup* stability condition. A common choice is so-called Taylor-Hood element, using quadratic basis functions to construct $\mathcal{U}_h$ and linear basis for $\mathcal{Q}_h$.

### Nonlinear iterations
To resolve the nonlinearity a Picard iteration scheme is employed where $2 \eta(\mathbf{u}) \dot{\varepsilon}(\mathbf{u}) \approx 2 \eta(\mathbf{u}_0) \dot{\varepsilon}(\mathbf{u})$, with $\mathbf{u}_0$ being some known approximation of $\mathbf{u}$. The following problem is then solved to obtain an improved guess $\mathbf{u}^{m+1}$ from the known guess $\mathbf{u}^m$:

Find $\mathbf{u}^{m+1} \in \mathcal{U}$ and $p^{m+1} \in \mathcal{Q}$, such that

```math
    \left (\dot{\varepsilon}(\mathbf{v}), 2 \eta(\mathbf{u}^m) \dot{\varepsilon}(\mathbf{u}^{m+1}) \right )_{\Omega} - (q, \nabla \cdot \mathbf{u}^{m+1})_{\Omega} - (\nabla \cdot \mathbf{v}, p^{m+1})_{\Omega} = (\mathbf{v}, \mathbf{f})_{\Omega}
```

for all $\mathbf{v} \in \mathcal{V}$ and all $q \in \mathcal{Q}$. This is then iterated upon until a user defined step tolerance $\epsilon_s$ is reached:

```math
    \lVert \mathbf{u}^{m+1} - \mathbf{u}^{m} \rVert_{L^2(\Omega)} < \epsilon_s \rVert \mathbf{u}^{m+1}\lVert_{L^2(\Omega)}
```

## The free-surface equation

### Strong formulation
The interface between the ice and atmosphere is modeled as freely moving boundary, this interface moves either due to ice particles being transported by the ice flow across the boundary, or due snow accumulating or ablating on top of it. Tracking the free-surface height $h(x, t)$, its evolution until time $T$ is described by the so-called free-surface equation

```math
\begin{equation}
    \frac{\partial h}{\partial t} + u^s_x(h(x, t)) \frac{\partial h}{\partial x} = u^s_z(h(x, t)) + a_s(x, t), \quad (x, t) \in \Gamma^{\perp}_s \times [0, T],
\end{equation}
```

where $u^s_x$ and $u^s_z$ are the respective horizontal and vertical ice surface velocities, $a_s$ is the surface mass balance, and $\Gamma_s^{\perp}$ is the projection of the surface onto the horizontal line $z = 0$.

### Weak formulation
Similarly, in this case the weak formulation is derived by multiplying by a test function $w$ and integrating over $\Gamma_s^{\perp}$, resulting in the following problem:

Find $h \in \mathcal{Z}$ such that

```math
    \left (w, \frac{\partial h}{\partial t} \right)_{\Gamma_s^{\perp}} + \left (w, u_x^s \frac{\partial h}{\partial x} \right)_{\Gamma_s^{\perp}} = \left (w, u^s_z + a_s\right )_{\Gamma_s^{\perp}}
```

for all $w \in \mathcal{Z}$, where $\mathcal{Z}$ is an appropriate Sobolev space.

### Time discretization
This equation is numerically integrated in time by replacing the time derivative with a forward difference scheme, and either evaluating $\frac{\partial h}{\partial x}$ at the current time step $k$ or the next time step $k+1$. The weak formulation for both cases can be written as:

Find $h^{k+1} \in \mathcal{Z}$ such that
```math
    \left (w, \frac{h^{k+1} - h^k}{\Delta t} \right)_{\Gamma_s^{\perp}} + \left (w, u_x^s \frac{\partial h^{k + \gamma}}{\partial x} \right)_{\Gamma_s^{\perp}} = \left (w, u^s_z + a_s\right )_{\Gamma_s^{\perp}}
```
for all $w \in \mathcal{Z}$, where setting $\gamma = 0$ and $\gamma = 1$ results in an explicit- and semi-implicit scheme, respectively.

After rearranging terms, the weak formulation for each case is:

**Explicit**

Find $h^{k+1} \in \mathcal{Z}$ such that

```math
    \left (w,  h^{k+1}\right)_{\Gamma_s^{\perp}} = \left (w, h^k\right)_{\Gamma_s^{\perp}} - \Delta t \left (w, u^s_x \frac{\partial h^k}{\partial x} \right)_{\Gamma_s^{\perp}} + \Delta t (w, u^s_z + a_s )_{\Gamma_s^{\perp}}
```

for all $w \in \mathcal{Z}$.

**Semi implicit**

Find $h^{k+1} \in \mathcal{Z}$ such that

```math
    \left (w,  h^{k+1}\right)_{\Gamma_s^{\perp}} + \Delta t \left (w, u^s_x \frac{\partial h^{k+1}}{\partial x} \right)_{\Gamma_s^{\perp}} = \left (w, h^k\right)_{\Gamma_s^{\perp}} + \Delta t \left (w, u^s_z + a_s \right)_{\Gamma_s^{\perp}}
```

for all $w \in \mathcal{Z}$.

### Free surface stabilization
The above time discretization are both examples of explicit (w.r.t. surface velocities) time stepping schemes, and are therefore subject to stability constraints on the time-step size. However, solving for the unknown velocities at the next time step in a fully implicit manner requires an iterative algorithm to resolve the nonlinearity. A more computationally appealing approach is to combine the stability of an implicit scheme with the low computational cost of an explicit scheme. A method to this end is the so-called free-surface stabilization algorithm (FSSA), which numerical studies have shown to increase stable time-step size up to an order of magnitude.

For the purpose of deriving this method, note that a fully implicit scheme corresponds to replacing the domain of integration in the weak formulation of the pStokes equation with $\Omega^{k+1}$. 

For the fully implicit scheme, the weak form reads:

Find $\mathbf{u} \in \mathcal{U}$ and $p \in \mathcal{Q}$, such that

```math
\left (\dot{\varepsilon}(\mathbf{v}), 2 \eta(\mathbf{u}) \dot{\varepsilon}(\mathbf{u}) \right )_{\Omega^{k+1}} - (\nabla \cdot \mathbf{v}, p)_{\Omega^{k+1}} - (q, \nabla \cdot \mathbf{u})_{\Omega^{k+1}} = (\mathbf{v}, \mathbf{f})_{\Omega^{k+1}}
```

for all $\mathbf{v} \in \mathcal{V}$ and all $q \in \mathcal{Q}$. 

Next all integrals on the left-hand side are approximated as 

```math
(\cdot, \cdot)_{\Omega^{k+1}} \approx (\cdot, \cdot)_{\Omega^{k}}.
```

Now only the right-hand side is integrated over $\Omega^{k+1}$, which is still unknown, but can be estimated by a Taylor expansion

```math
(\mathbf{v}, \mathbf{f})_{\Omega^{k+1}} \approx (\mathbf{v}, \mathbf{f})_{\Omega^k} + \theta \Delta t (\mathbf{v}, (\mathbf{u}_b \cdot \hat{\mathbf{n}}) \mathbf{f} )_{\Gamma_s^k}
```
The boundary velocity $\mathbf{u}_b = \mathbf{u} + a_s \mathbf{e}_z$ is simply the sum of the surface velocity and the vertical accumulation rate. Inserting this into the above and moving the term involving $\mathbf{u}$ to the left-hand side gives the free-surface stabilized weak formulation of the pStokes equations:

Find $\mathbf{u} \in \mathcal{U}$ and $p \in \mathcal{Q}$, such that

```math
\left (\dot{\varepsilon}(\mathbf{v}), 2 \eta(\mathbf{u}) \dot{\varepsilon}(\mathbf{u}) \right )_{\Omega^k} - (\nabla \cdot \mathbf{v}, p)_{\Omega^k} - (q, \nabla \cdot \mathbf{u})_{\Omega^k} - \theta \Delta t (\mathbf{v}, (\mathbf{u} \cdot \hat{\mathbf{n}}) \mathbf{f} )_{\Gamma_k^s} = (\mathbf{v}, \mathbf{f})_{\Omega^k} + \theta \Delta t (\mathbf{v}, (a_s \mathbf{e}_z \cdot \hat{\mathbf{n}}) \mathbf{f} )_{\Gamma_s^k}
```

for all $\mathbf{v} \in \mathcal{V}$ and all $q \in \mathcal{Q}$. 

It is seen that two extra terms are included that adjusts velocities based on the movement of the surface, essentially making the pStokes equation aware of the evolving domain. The added term on the left-hand side accounts for the movement due to the ice flow, and the term on the right-hand side the movement due to accumulation or ablation. In addition an implicitness parameter $\theta \in \mathbb{R}_+$ has also been introduced, where setting $\theta = 0$ results in an explicit solver and $\theta = 1$ a (quasi) implicit solver. In the code the FSSA parameter in the FSSA assembly routine corresponds to $\theta \Delta t$.

# Demos
The two main modules in this project are the pStokesProblem and the FreeSurfaceProblem, used for setting up and solving the pStokes equation and the free-surface equation, respectively. These modules are deliberately made independent, and it is up to the discretion of the user to couple the two. Full working examples on how this is done in, for both the C++ and Python interface, is provided in the demos below. The demos can be also found under /demos, where CMake files are provided for compiling the C++ code.

## C++

Running this demo requires [Matplot++](https://matplotlib.org), a C++ plotting library with an [API](https://en.wikipedia.org/wiki/API) similar to the popular Python library [Matplotlib](https://matplotlib.org/). A tarball for Matplot++ is available from [here](https://github.com/alandefreitas/matplotplusplus/archive/refs/tags/v1.2.2.tar.gz). To install it, first extract it

```console
$ tar -xvzf v1.2.2.tar.gz && cd matplotplusplus-1.2.2
```
then configure
```console
$ mkdir -p .build && cd .build && cmake .. -DCMAKE_BUILD_TYPE=release -DBUILD_SHARED_LIBS=ON -DMATPLOTPP_BUILD_EXAMPLES=OFF -DMATPLOTPP_BUILD_TESTS=OFF
```
if successful, build it (using all available cores for fast compilation)
```console
$ make -j$(cat /proc/cpuinfo | grep "core id" | sort | uniq | wc -l)
```
and if compilation succeeded, finally install it
```console
# make install 
```
Lastly, install the default [gnuplot](https://gnuplot.info) backend
```console
# apt install gnuplot
```
To run the demo below first compile and install it:
```console
$ mkdir -p .build && cd .build && cmake .. -DCMAKE_BUILD_TYPE=release && make && make install && cd ..
```
And then run the executable:
```console
$ bin/biceps_demo
```

```C++
#include <enums.hpp>
#include <pstokes_fem.hpp>
#include <free_surface_fem.hpp>
#include <boost/format.hpp>
#include <matplot/matplot.h>

#define GRAVITY 9.8
#define ICE_DENSITY 910

// Define domain and grid parameters
FloatType x0 = 0.0;  // Left end
FloatType x1 = 100.0;  // Right end
FloatType L = x1 - x0;  // Length of the domain
FloatType H = 1.0;  // Mean height of the domain
FloatType z0 = 0.1;  // Amplitude of surface undulation

FloatType A = 100.0;  // Ice softness parameter
FloatType n_i = 3.0;  // Glen exponent
FloatType eps_reg_2 = 1e-10;  // Regularization parameter
int fssa_version = FSSA_NONE;  // No FSSA stabilization
FloatType fssa_param = 0;  // Stabilization parameter in FSSA 

int nx = 50;  // Number of elements in x-direction
int nz = 5;  // Number of elements in z-direction
int nt = 100;  // Number of time steps
FloatType dt = 35.0;  // Time step size
int deg_u = 2;  // Polynomial degree for velocity field
int deg_p = 1;  // Polynomial degree for pressure field
int deg_h = 1;  // Polynomial degree for height field
int gauss_precision = 5;  // Number of Gauss points in each direction per element
int max_iter = 100;  // Maximum number of iterations for solver
FloatType stol = 1e-6;  // Convergence tolerance for solver
int cell_type = MESH2D::TRIANGLE_LEFT;  // 2D mesh cell type

FloatType zb_expr(FloatType x)
{
    return 0.0;
}

FloatType zs_expr(FloatType x)
{
    return H + z0*COS_FUNC(PI_CONST*x/L);
}

FloatType force_x(FloatType x, FloatType z)
{
    return 0.0;
}

FloatType force_z(FloatType x, FloatType z)
{
    return -1e-3*ICE_DENSITY*GRAVITY;
}

int main(int argc, char *argv[])
{
    // Create structured meshes for velocity, pressure, and height fields
    StructuredMesh u_mesh_2d(nx, nz, deg_u, cell_type);
    StructuredMesh p_mesh_2d(nx, nz, deg_p, cell_type);

    // Extrude the meshes in the x and z directions
    u_mesh_2d.extrude_x(x0, x1);
    u_mesh_2d.extrude_z(zb_expr, zs_expr);
    p_mesh_2d.extrude_x(x0, x1);
    p_mesh_2d.extrude_z(zb_expr, zs_expr);

    // Extract degrees of freedom for surface nodes
    std::vector<int> sdofs_u = u_mesh_2d.extract_dof_inds(MESH2D::SURFACE_ID);
    std::vector<int> sdofs_h = u_mesh_2d.extract_vertex_dof_inds(MESH2D::SURFACE_ID);

    // Extract surface coordinates
    Eigen::MatrixX<FloatType> spmat_u = u_mesh_2d.pmat(sdofs_u, Eigen::all);
    Eigen::MatrixX<FloatType> spmat_h = u_mesh_2d.pmat(sdofs_h, Eigen::all);
    Eigen::VectorX<FloatType> xs_vec = spmat_h(Eigen::all, 0);
    Eigen::VectorX<FloatType> zs_vec = spmat_h(Eigen::all, 1);

    // Project mesh to z=0
    spmat_u(Eigen::all, 1).array() = 0.0;
    spmat_h(Eigen::all, 1).array() = 0.0;

    // Create 1D meshes for velocity and height on the surface
    IntervalMesh u_mesh_1d = IntervalMesh(spmat_u, deg_u);
    IntervalMesh h_mesh_1d = IntervalMesh(spmat_h, deg_h);

    // Define ids for Dirichlet boundary condition
    int ux_boundary_id = (
        MESH2D::NORTH_WEST_ID |
        MESH2D::WEST_ID |
        MESH2D::BED_ID |
        MESH2D::EAST_ID |
        MESH2D::NORTH_EAST_ID
    );
    int uz_boundary_id = MESH2D::BED_ID;

    // Initialize FEM functions for height, velocity, and accumulation
    FEMFunction1D ux_func = FEMFunction1D(u_mesh_1d);
    FEMFunction1D uz_func = FEMFunction1D(u_mesh_1d);
    FEMFunction1D h0_func = FEMFunction1D(h_mesh_1d);
    FEMFunction1D ac_func = FEMFunction1D(h_mesh_1d);

    // For L2 norm calculation
    h0_func.assemble_mass_matrix();

    // Initialize the pStokes and Free Surface Problem
    pStokesProblem psp(u_mesh_2d, p_mesh_2d);
    FreeSurfaceProblem fsp(h_mesh_1d, u_mesh_1d);

    // Plot initial surface profile
    matplot::plot(xs_vec, zs_vec)->line_width(2.0);
    matplot::hold(true);
    for (int k = 0; k < nt; k++) {
        psp.solve_nonlinear_system_picard(
            A, n_i, eps_reg_2, fssa_version, fssa_param, force_x, force_z,
            ux_boundary_id, uz_boundary_id, max_iter, stol, gauss_precision
        );
        // Clear lhs matrix and rhs vector.
        psp.reset_system();

        // Extract velocity field solutions
        Eigen::VectorX<FloatType> ux_vec = psp.velocity_x().vals;
        Eigen::VectorX<FloatType> uz_vec = psp.velocity_z().vals;
        // Set free surface velocity
        ux_func.vals = ux_vec(sdofs_u);
        uz_func.vals = uz_vec(sdofs_u);
        // Set initial height
        h0_func.vals = zs_vec;

        // Print surface energy and domain area
        std::cout << boost::format{"||E|| = %.16f"} %h0_func.L2_norm() << std::endl;
        std::cout << boost::format{"A = %.16f"} %u_mesh_2d.area() << std::endl;

        // Solve the free surface problem using explicit time stepping
        fsp.assemble_lhs_explicit(gauss_precision);
        fsp.commit_lhs();
        fsp.assemble_rhs_explicit(
            h0_func, ux_func, uz_func, ac_func, dt, gauss_precision
        );
        fsp.solve_linear_system();
        // Clear lhs matrix and rhs vector
        fsp.reset_system();

        // Update surface elevation
        zs_vec = fsp.zs_vec;

        // Update mesh with new surface elevation
        u_mesh_2d.extrude_z(zs_vec);
        p_mesh_2d.extrude_z(zs_vec);

    }

    // Plot final surface
    matplot::plot(xs_vec, zs_vec)->line_width(2.0);
    matplot::show();

    return 0;
}
```

## Python

This demo requires [Matplotlib](https://matplotlib.org):

```console
# apt install python3-matplotlibi
```

To run:

```console
$ python biceps_demo.py
```

```python
import numpy as np
import matplotlib.pyplot as plt
import biceps as bp

GRAVITY = 9.8
ICE_DENSITY = 910


# Define the expressions for the bottom and surface elevations
def zb_expr(x): return 0.0  # Flat bottom surface
def zs_expr(x): return H + z0*np.cos(np.pi*x/L)  # Undulating surface elevation


# Define external force functions
def force_x(x, z): return 0.0  # No horizontal force
def force_z(x, z): return -1e-3*GRAVITY*ICE_DENSITY  # Gravity force in the z-direction


# Define domain and grid parameters
x0 = 0.0  # Left end
x1 = 100.0  # Right end
L = x1 - x0  # Length of the domain
H = 1  # Mean height of the domain
z0 = 0.1  # Amplitude of surface undulation

A = 100.0  # Ice softness parameter
n_i = 3.0  # glen exponent
eps_reg_2 = 1e-10  # Regularization parameter
fssa_version = bp.FSSA_VERSION.FSSA_NONE  # No FSSA stabilization
fssa_param = 0  # Stabilization parameter in FSSA

nx = 50  # Number of elements in x-direction
nz = 5  # Number of elements in z-direction
nt = 100  # Number of time steps
dt = 35  # Time step size

deg_u = 2  # Polynomial degree for velocity field
deg_p = 1  # Polynomial degree for pressure field
deg_h = 1  # Polynomial degree for height field

max_iter = 100  # Maximum number of iterations for solver
stol = 1e-6  # Convergence tolerance for solver
cell_type = bp.CELL_TYPE_2D.TRIANGLE_LEFT  # 2D mesh cell type

# Create structured meshes for velocity, pressure, and height fields
u_mesh_2d = bp.StructuredMesh(nx, nz, deg_u, cell_type)
p_mesh_2d = bp.StructuredMesh(nx, nz, deg_p, cell_type)

# Extrude the meshes in the x and z directions
u_mesh_2d.extrude_x(x0, x1)
u_mesh_2d.extrude_z(zb_expr, zs_expr)
p_mesh_2d.extrude_x(x0, x1)
p_mesh_2d.extrude_z(zb_expr, zs_expr)

# Extract degrees of freedom for surface nodes
sdofs_u = u_mesh_2d.extract_dof_inds(bp.DOMAIN_IDS_2D.SURFACE_ID)
sdofs_h = u_mesh_2d.extract_vertex_dof_inds(bp.DOMAIN_IDS_2D.SURFACE_ID)

# Extract surface coordinates
spmat_u = u_mesh_2d.pmat[sdofs_u, :].copy()
spmat_h = u_mesh_2d.pmat[sdofs_h, :].copy()
xs_vec = spmat_h[:, 0].copy()
zs_vec = spmat_h[:, 1].copy()
# Project mesh to z=0
spmat_u[:, 1] = 0
spmat_h[:, 1] = 0

# Create 1D meshes for velocity and height on the surface
u_mesh_1d = bp.IntervalMesh(spmat_u, deg_u)
h_mesh_1d = bp.IntervalMesh(spmat_h, deg_h)

# Initialize FEM functions for height, velocity, and accumulation
ux_func = bp.FEMFunction1D(u_mesh_1d)
uz_func = bp.FEMFunction1D(u_mesh_1d)
h0_func = bp.FEMFunction1D(h_mesh_1d)
ac_func = bp.FEMFunction1D(h_mesh_1d)

# For L2 norm calculation
h0_func.assemble_mass_matrix()

# Initialize the pStokes
psp = bp.pStokesProblem(
    A, n_i, eps_reg_2, force_x, force_z, u_mesh_2d, p_mesh_2d
)

# Set Dirichlet BC masks
# Horizontal velocity component
psp.ux_dirichlet_bc_mask = (
    bp.DOMAIN_IDS_2D.NORTH_WEST_ID |
    bp.DOMAIN_IDS_2D.WEST_ID |
    bp.DOMAIN_IDS_2D.BED_ID |
    bp.DOMAIN_IDS_2D.EAST_ID |
    bp.DOMAIN_IDS_2D.NORTH_EAST_ID
)
# Vertical velocity component
psp.uz_dirichlet_bc_mask = bp.DOMAIN_IDS_2D.BED_ID

# Initialize the free-surface problem
fsp = bp.FreeSurfaceProblem(h_mesh_1d, u_mesh_1d)

# Plot initial surface profile
plt.plot(xs_vec, zs_vec)

# Time-stepping loop
for k in range(nt):
    # Assemble and solve the nonlinear pStokes problem
    psp.solve_nonlinear_system()
    # Clear lhs matrix and rhs vector
    psp.reset_system()

    # Extract velocity field solutions
    ux_vec = psp.velocity_x().vals
    uz_vec = psp.velocity_z().vals
    # Set free surface velocity
    ux_func.vals = ux_vec[sdofs_u]
    uz_func.vals = uz_vec[sdofs_u]
    # Set initial height
    h0_func.vals = zs_vec.copy()

    # Print surface energy and domain area
    print(f"||E|| = {h0_func.L2_norm(): .16f}")
    print(f"A = {u_mesh_2d.area(): .16f}")

    # Solve the free surface problem using explicit time stepping
    fsp.assemble_lhs_explicit()
    fsp.commit_lhs()
    fsp.assemble_rhs_explicit(
        h0_func, ux_func, uz_func, ac_func, dt
    )
    fsp.solve_linear_system()
    # Clear lhs matrix and rhs vector
    fsp.reset_system()

    # Update surface elevation
    zs_vec = fsp.zs_vec.copy()

    # Update mesh with new surface elevation
    u_mesh_2d.extrude_z(zs_vec)
    p_mesh_2d.extrude_z(zs_vec)

# Plot final surface
plt.plot(xs_vec, zs_vec)
plt.show()
```

## Julia

This demo requires Julia packages [PyCall](https://github.com/JuliaPy/PyCall.jl) and [Plots](https://docs.juliaplots.org/stable/):

```julia
import Pkg
Pkg.add("PyCall")
Pkg.add("Plots")
```

To run:

```console
$ julia biceps_demo.jl
```

```julia
using PyCall
using Plots
import Printf.@printf
bp = pyimport("biceps")

GRAVITY = 9.8  # Gravitational acceleration
ICE_DENSITY = 910  # Ice mass density

x0 = 0.0  # Left end
x1 = 100.0  # Right end
L = x1 - x0  # Length of the domain
H = 1  # Mean height of the domain
z0 = 0.1  # Amplitude of surface undulation

py"""
import numpy as np

# Define the expressions for the bottom and surface elevations
def zb_expr(x): return 0.0  # Flat bottom surface
def zs_expr(x): return $H + $z0*np.cos(np.pi*x/$L)  # Undulating surface elevation


# Define external force functions
def force_x(x, z): return 0.0  # No horizontal force
def force_z(x, z): return -1e-3*$GRAVITY*$ICE_DENSITY  # Gravity force in the z-direction
"""

A = 100.0  # Ice softness parameter
n_i = 3.0  # glen exponent
eps_reg_2 = 1e-10  # Regularization parameter
fssa_version = bp.FSSA_VERSION.FSSA_NONE  # No FSSA stabilization
fssa_param = 0  # Stabilization parameter in FSSA

nx = 50  # Number of elements in x-direction
nz = 5  # Number of elements in z-direction
nt = 100  # Number of time steps
dt = 35  # Time step size

deg_u = 2  # Polynomial degree for velocity field
deg_p = 1  # Polynomial degree for pressure field
deg_h = 1  # Polynomial degree for height field

max_iter = 100  # Maximum number of iterations for solver
stol = 1e-6  # Convergence tolerance for solver
cell_type = bp.CELL_TYPE_2D.TRIANGLE_LEFT  # 2D mesh cell type

# Create structured meshes for velocity, pressure, and height fields
u_mesh_2d = bp.StructuredMesh(nx, nz, deg_u, cell_type)
p_mesh_2d = bp.StructuredMesh(nx, nz, deg_p, cell_type)

# Extrude the meshes in the x and z directions
u_mesh_2d.extrude_x(x0, x1)
u_mesh_2d.extrude_z(py"zb_expr", py"zs_expr")
p_mesh_2d.extrude_x(x0, x1)
p_mesh_2d.extrude_z(py"zb_expr", py"zs_expr")

# Extract degrees of freedom for surface nodes
sdofs_u = u_mesh_2d.extract_dof_inds(bp.DOMAIN_IDS_2D.SURFACE_ID)
sdofs_h = u_mesh_2d.extract_vertex_dof_inds(bp.DOMAIN_IDS_2D.SURFACE_ID)
# Compensate for julia being 1-indexed
sdofs_u = 1 .+ sdofs_u
sdofs_h = 1 .+ sdofs_h

# Extract surface coordinates
spmat_u = copy(u_mesh_2d.pmat[sdofs_u, :])
spmat_h = copy(u_mesh_2d.pmat[sdofs_h, :])
xs_vec = copy(spmat_h[:, 1])
zs_vec = copy(spmat_h[:, 2])
# Project mesh to z=0
spmat_u[:, 2] .= 0
spmat_h[:, 2] .= 0

# Create 1D meshes for velocity and height on the surface
u_mesh_1d = bp.IntervalMesh(spmat_u, deg_u)
h_mesh_1d = bp.IntervalMesh(spmat_h, deg_h)

# Initialize FEM functions for height, velocity, and accumulation
ux_func = bp.FEMFunction1D(u_mesh_1d)
uz_func = bp.FEMFunction1D(u_mesh_1d)
h0_func = bp.FEMFunction1D(h_mesh_1d)
ac_func = bp.FEMFunction1D(h_mesh_1d)

# For L2 norm calculation
h0_func.assemble_mass_matrix()

# Initialize the pStokes
psp = bp.pStokesProblem(
    A, n_i, eps_reg_2, py"force_x", py"force_z", u_mesh_2d, p_mesh_2d
)

# Set Dirichlet BC masks
# Horizontal velocity component
psp.ux_dirichlet_bc_mask = (
    bp.DOMAIN_IDS_2D.NORTH_WEST_ID |
    bp.DOMAIN_IDS_2D.WEST_ID |
    bp.DOMAIN_IDS_2D.BED_ID |
    bp.DOMAIN_IDS_2D.EAST_ID |
    bp.DOMAIN_IDS_2D.NORTH_EAST_ID
)
# Vertical velocity component
psp.uz_dirichlet_bc_mask = bp.DOMAIN_IDS_2D.BED_ID

# Initialize the free-surface problem
fsp = bp.FreeSurfaceProblem(h_mesh_1d, u_mesh_1d)

# Plot initial surface profile
p = plot()
plot!(xs_vec, zs_vec, lw=2)
# Time-stepping loop
for k=1:nt
    # Assemble and solve the nonlinear pStokes problem
    psp.solve_nonlinear_system()
    # Clear lhs matrix and rhs vector
    psp.reset_system()

    # Extract velocity field solutions
    ux_vec = psp.velocity_x().vals
    uz_vec = psp.velocity_z().vals
    # Set free surface velocity
    ux_func.vals = ux_vec[sdofs_u]
    uz_func.vals = uz_vec[sdofs_u]
    # Set initial height
    h0_func.vals = copy(zs_vec)

    # Print surface energy and domain area
    @printf("||E|| = %.16f\n", h0_func.L2_norm())
    @printf("A = %.16f\n", u_mesh_2d.area())

    # Solve the free surface problem using explicit time stepping
    fsp.assemble_lhs_explicit()
    fsp.commit_lhs()
    fsp.assemble_rhs_explicit(
        h0_func, ux_func, uz_func, ac_func, dt
    )
    fsp.solve_linear_system()
    # Clear lhs matrix and rhs vector
    fsp.reset_system()

    # Update surface elevation
    global zs_vec = copy(fsp.zs_vec)

    # Update mesh with new surface elevation
    u_mesh_2d.extrude_z(zs_vec)
    p_mesh_2d.extrude_z(zs_vec)
end

# Plot final surface
plot!(xs_vec, zs_vec, lw=2)
gui()
# Pause program to display plot
readline()
```
