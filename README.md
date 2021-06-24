# SCP for Trajectory Optimization

This repository contains sequential convex programming (SCP) algorithms for
real-time generation of dynamically feasible trajectories of aerospace,
robotic, and other systems. Under the hood, the algorithms rely on optimal
control and convex numerical optimization theory.

Four algorithms are implemented, and can be found in the `solvers/` directory:
- Lossless convexification ([LCvx](https://doi.org/10.2514/1.27553))
- Successive convexification ([SCvx](https://arxiv.org/abs/1804.06539))
- Guaranteed Sequential Trajectory Optimization ([GuSTO](http://asl.stanford.edu/wp-content/papercite-data/pdf/Bonalli.Cauligi.Bylard.Pavone.ICRA19.pdf))
- Penalized trust region ([PTR](https://arxiv.org/abs/1811.10803))

Several example applications show how the algorithms can be used. These can all
be found in the `examples/` director, and include:

1. [Double integrator with friction](examples/src/double_integrator)
2. [Mars rocket landing](examples/src/rocket_landing)
3. [SpaceX Starship landing "flip" maneuver](examples/src/starship_flip)
4. [Mass-spring-damper with an actuator deadband or
   "sticking"](examples/src/oscillator)
5. [Quadrotor flight around obstacles](examples/src/quadrotor)
6. [Space station freeflyer robot](examples/src/freeflyer)
7. [Planar spacecraft rendezvous with discrete
   logic](examples/src/rendezvous_planar)
8. [Apollo transposition and docking maneuver with discrete
   logic](examples/src/rendezvous_3d)

More documentation is on its way :books:

# Citing

If you use this code, kindly cite the following associated publication.

```
@article{SCPTrajOptCSM2021,
  year	       = {2021},
  publisher    = {{IEEE}},
  author       = {Danylo Malyuta and Taylor P. Reynolds and Michael Szmuk
                  and Thomas Lew and Riccardo Bonalli and Marco Pavone
                  and Behcet Acikmese},
  title	       = {Convex Optimization for Trajectory Generation},
  journal      = {{IEEE} Control Systems Magazine (in review)},
  pages        = {arXiv:2106.09125}
}
```
