
 2D LAPLACE'S EQUATION IN A UNIT SQUARE:
 FINITE DIFFERENCES vs. PHYSICS-INFORMED NEURAL NETWORK (PINN)
===============================================================================

1. OVERVIEW
-----------
This script solves the 2-D Laplace equation for the electrostatic potential
V(x, y) inside the unit square [0,1] x [0,1] in three ways, and compares them:

  1. Analytic (exact) solution        - the reference "ground truth"
  2. Finite differences (FD)          - classic Jacobi iteration on a grid
  3. Physics-Informed Neural Network  - a small PyTorch network trained so
                                        that it satisfies the PDE and the
                                        boundary conditions

It reports errors and run times for each method, plots the solutions and
error surfaces in 3-D, and finally tests how well the trained PINN
generalises to a much finer grid it never saw during training.


2. THE PHYSICS PROBLEM
----------------------
PDE (interior of the box):

      d2V/dx2 + d2V/dy2 = 0              for 0 < x < 1, 0 < y < 1

Boundary conditions (Dirichlet):

      V(x, 1) = sin(pi x)                top edge  ("the lid")
      V(x, 0) = 0                        bottom edge (grounded)
      V(0, y) = 0                        left edge   (grounded)
      V(1, y) = 0                        right edge  (grounded)

Exact solution (separation of variables):

      V(x, y) = sin(pi x) * sinh(pi y) / sinh(pi)


3. REQUIREMENTS
---------------
  - Python 3.8+
  - numpy
  - matplotlib   (uses mpl_toolkits.mplot3d for 3-D surfaces)
  - torch        (PyTorch, CPU is sufficient)

Install with: pip install numpy matplotlib torch


4. HOW TO RUN
-------------
Save the code as e.g. laplace_fd_vs_pinn.py and run: python laplace_fd_vs_pinn.py

or paste it into a Jupyter notebook / Google Colab cell. Four figure windows
open in sequence (close each one to continue when running as a script).
Expect the PINN training (20,000 iterations) to take from tens of seconds to
a few minutes, and the 200x200 FD solve to take noticeably longer than the
20x20 one (Jacobi converges slowly on fine grids).


5. CODE STRUCTURE
-----------------
grid(N)
    Builds an (N+2) x (N+2) uniform mesh on [0,1]^2 (N interior points per
    direction plus the two boundary points). Arrays are indexed V[j, i] with
    row j <-> y and column i <-> x ("xy" meshgrid indexing).

V_exact(N)
    Evaluates the analytic solution on grid(N).

finite_diff(N)
    Jacobi iteration: every interior point is repeatedly replaced by the
    average of its four neighbours (the 5-point stencil for the Laplacian).
    Starts from zero, sets the top boundary to sin(pi x), and stops when the
    largest change between iterations is below tol = 1e-8 (or after
    100,000 iterations). Returns (V_fd, elapsed_time_in_seconds).

surface(ax, N, Z, title, ...)
    Helper that draws a field Z(x, y) as a 3-D surface on grid(N).

PINN data preparation
    - Interior collocation points: N_GRID x N_GRID points strictly inside
      (0,1)^2, where the PDE residual is enforced.
    - Boundary points: N_GRID evenly spaced points on each of the four
      edges (corners included) with their prescribed values V_boundary.

class PINN(nn.Module)
    Fully connected network (x, y) -> V.
    Architecture: `depth` hidden layers of `width` neurons with tanh
    activations, linear output. Xavier-normal weight init, zero biases.
    tanh is used because the loss needs smooth second derivatives
    (ReLU has zero second derivative almost everywhere).

laplacian(f, xy)
    Computes V_xx + V_yy of any function f at the points xy using
    torch.autograd (automatic differentiation, create_graph=True so the
    result can itself be back-propagated through).

pinn_loss(model)
    Returns two terms:
      loss_pde = mean( (V_xx + V_yy)^2 )        over interior points
      loss_bc  = mean( (V_pred - V_boundary)^2 ) over boundary points

Training loop
    Total loss = lamda * loss_pde + loss_bc, minimised with Adam.
    Loss history (total, PDE, BC) is stored and plotted on a log scale.


6. KEY PARAMETERS
-----------------
  N_GRID      = 20       interior points per direction (grid is 22 x 22)
  tol         = 1e-8     FD convergence tolerance (inside finite_diff)
  input       = 2        network inputs (x, y)
  output      = 1        network output V
  width       = 32       neurons per hidden layer
  depth       = 4        number of hidden layers
  max_epochs  = 20000    training iterations
  lr          = 1e-4     Adam learning rate
  lamda       = 1e-1     weight of the PDE term relative to the BC term
  Nn          = 200      fine grid used to test generalisation (202 x 202)
  torch.manual_seed(0)   for reproducible PINN results


7. OUTPUT
---------
Console:
  - FD max / mean error vs. exact solution and solve time (ms)
  - PINN loss every 5,000 iterations (total, PDE, BC) and training time
  - FD vs. PINN max / mean error and evaluation time on the 22x22 grid
  - PINN max / mean error on the fine grid and its evaluation time

Figures:
  1. FD solution | exact solution | |FD - exact|
  2. PINN training curves (total, PDE and BC loss, log scale)
  3. Top row:    FD, PINN, exact solutions on the 22x22 grid
     Bottom row: |FD - exact|, |PINN - exact|, |PINN - FD|
     (shared z-scales so the panels are directly comparable)
  4. Fine-grid test (Nn = 200):
     Top row:    PINN, FD, exact solutions
     Bottom row: |PINN - exact|, |FD - exact|, |PINN - FD|


8. WHAT THE COMPARISON SHOWS
----------------------------
  - FD is accurate and fast on a small grid, but its answer only exists at
    the grid nodes; a finer grid means re-solving from scratch, and Jacobi
    iteration gets much slower as the grid grows.
  - The PINN is mesh-free: once trained, it can be evaluated at any (x, y)
    almost instantly, including on a 200x200 grid it never saw. The price
    is a comparatively long training phase and, usually, a lower accuracy
    than FD unless the network and training are tuned further.
  - The largest PINN errors typically appear near the top corners, where
    the boundary data and the PDE are hardest to satisfy simultaneously.


9. NOTES AND POSSIBLE IMPROVEMENTS
----------------------------------
  - finite_diff(N) is called twice for each grid (once to get V, once to
    get the time), so each FD solve runs twice. Calling it once and
    unpacking, e.g.  V_fd, t_fd = finite_diff(N_GRID),  halves the cost.
  - The comment "shape (100, 1)" next to the boundary tensor `e` is
    outdated: its shape is (N_GRID, 1), i.e. (20, 1) with the defaults.
  - The variable name `input` shadows Python's built-in input(); a name
    like `n_in` would be safer.
  - The PINN timing on the 22x22 grid measures inference only; training
    time is reported separately.
  - Ideas to improve PINN accuracy: more / randomly resampled collocation
    points, a larger learning rate with a scheduler, an L-BFGS fine-tuning
    stage after Adam, tuning `lamda`, or hard-coding the boundary
    conditions into the network output.
  - Gauss-Seidel or SOR would converge much faster than Jacobi for FD.
