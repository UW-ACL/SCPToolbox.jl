# SCP for Trajectory Optimization

Sequential convex programming (SCP) enables the real-time generation of
dynamically feasible trajectories for robotic, aerospace, and other
systems. Under the hood, the algorithm relies on optimal control and convex
optimization theory.

This project contains the code for several SCP algorithms, and several examples
of using these algorithms for trajectory generation. It is supplied as part of
the following IEEE Control Systems Magazine publication:

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

The examples can be found in the `examples/` directory, and include:

- Quadrotor path planning with obstacle avoidance
- Freeflyer flight inside of a space station environment with obstacles
- Variable-mass rocket landing using lossless convexification. This does not
  rely on SCP, but rather on pure convex optimization.

More documentation is on its way :books:
