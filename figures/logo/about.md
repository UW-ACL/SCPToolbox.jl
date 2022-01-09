The SCP Toolbox logo is composed of the following elements:

- SCP: this is the core algorithmic framework of the toolbox, which stands for
  "**S**equential **C**onvex **P**rogramming";
- _(t)_: most vehicle trajectories evolve in time (even though they can also
  evolve in any other monotonic variable, such as atmospheric density). The
  _(t)_ emphasizes the time-dependence of the optimization framework. It is
  also a happy coincidence that "time" and "toolbox" both start with a _t_!
- There is a red curve in the background that can represent either a trajectory
  or a nonconvex function. The trajectory starting point is the red dot, and
  the end point is the green dot;
- SCP works by iteratively linearizing nonconvex elements of the problem. The
  blue line segments are represent local linearizations of the red curve. The
  yellow dots are the points at which those linearizations are taken, and also
  represent temporal nodes of the discretized trajectory.
