# SCP for Trajectory Optimization

Sequential convex programming (SCP) enables the real-time generation of
dynamically feasible trajectories for robotic, aerospace, and other
systems. Under the hood, the algorithm relies on optimal control and convex
optimization theory.

This project contains the code for several SCP algorithms, and several examples
of using these algorithms for trajectory generation. The examples can be found
in the `examples/` directory, and include:

- Quadrotor path planning with obtacle avoidance
- Freeflyer flight inside of a space station environment with obtacles
- SpaceX Starship landing "flip" maneuver
- (Bonus) Variable-mass rocket landing using lossless convexification. This
  does not rely on SCP, but rather on pure convex optimization.

More documentation is on its way :books:
