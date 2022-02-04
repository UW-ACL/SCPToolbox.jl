> :information_source: The repository name has changed to `SCPToolbox.jl` in
> order to reflect the project's direction: to develop a general-purpose
> trajectory optimization toolkit using sequential convex programming
> algorithms.

<p align="center">
<a href="media/logo/about.md" title="About the logo">
<img alt="SCP Toolbox"
    title="SCP Toolbox"
    src="media/logo/logo.png"
    width="400px" />
</a>
</p>

<p align="center">
    <a href="http://www.gnu.org/licenses/gpl-3.0.txt"><img src="https://img.shields.io/badge/license-GPL_3-green.svg" alt="License GPL 3" /></a>
    &ensp;&ensp;
    <a href="https://mybinder.org/v2/gh/UW-ACL/SCPToolbox_tutorial/master?labpath=tutorial%2Fsrc%2Fp1_clp.ipynb"><img src="https://mybinder.org/badge_logo.svg" alt="Launch Binder" /></a>
</p>

The <b>SCP Toolbox</b> provides the tools necessary to define and solve
nonconvex trajectory optimization problems. The user-facing part of the toolbox
provides a trajectory problem parser that allows one to define the system
dynamics, state and input constraints, and boundary conditions. Under the hood,
the problem is solved using any one of several _Sequential Convex Programming_
(SCP) algorithms. These algorithms have been successfully demonstrated on a
number of difficult aerospace, autonomous driving, robotics, and other
applications. A major goal of the SCP Toolbox is to provide working reference
implementations of the SCP algorithms. By placing the algorithms behind a
parser that transforms trajectory problems into their abstract mathematical
definitions, the algorithms can be generically tested on a suite of examples
without having to re-implement the underlying algorithms each time.

## Getting Started

Click on the [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/UW-ACL/SCPToolbox_tutorial/master?labpath=tutorial%2Fsrc%2Fp1_clp.ipynb) button to spin up a remote Jupyter environment. Following the included notebooks to get a feel for the toolbox, and finish by solving a self-guided tutorial to land a rocket on the Moon!

## Implemented SCP algorithms

The following algorithms are implemented, and can be found in the
`src/solvers/` directory:

- Penalized trust region ([PTR](https://arxiv.org/abs/1811.10803))
- Successive convexification ([SCvx](https://arxiv.org/abs/1804.06539))
- Guaranteed Sequential Trajectory Optimization ([GuSTO](http://asl.stanford.edu/wp-content/papercite-data/pdf/Bonalli.Cauligi.Bylard.Pavone.ICRA19.pdf))
- Lossless convexification ([LCvx](https://doi.org/10.2514/1.27553))

## Implemented examples

Several example applications show how the algorithms can be used. These can all
be found in the `test/examples/` directory, and include:

1. [Double integrator with friction](test/examples/double_integrator)
2. [Mars rocket landing](test/examples/rocket_landing)
3. [SpaceX Starship landing "flip" maneuver](test/examples/starship_flip)
4. [Mass-spring-damper with an actuator deadband or
   "sticking"](test/examples/oscillator)
5. [Quadrotor flight around obstacles](test/examples/quadrotor)
6. [Space station freeflyer robot](test/examples/freeflyer)
7. [Planar spacecraft rendezvous with discrete
   logic](test/examples/rendezvous_planar)
8. [Apollo transposition and docking maneuver with discrete
   logic](test/examples/rendezvous_3d)

## Citing

If you use the SCP Toolbox, kindly cite the following associated publication.

```
@article{SCPToolboxCSM2022,
  year	       = {2021},
  publisher    = {{IEEE}},
  author       = {Danylo Malyuta and Taylor P. Reynolds and Michael Szmuk
                  and Thomas Lew and Riccardo Bonalli and Marco Pavone
                  and Behcet Acikmese},
  title	       = {Convex Optimization for Trajectory Generation},
  journal      = {{IEEE} Control Systems Magazine (accepted)},
  pages        = {arXiv:2106.09125}
}
```
