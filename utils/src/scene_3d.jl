#= 3D plot creation utilities.

Sequential convex programming algorithms for trajectory optimization.
Copyright (C) 2021 Autonomous Controls Laboratory (University of Washington),
                   and Autonomous Systems Laboratory (Stanford University)

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
this program.  If not, see <https://www.gnu.org/licenses/>. =#

LangServer = isdefined(@__MODULE__, :LanguageServer)

if LangServer
    include("helper.jl")
    include("plots.jl")
    include("tree.jl")
end

using Statistics
using PyPlot

export Mesh3D, Camera3D, Axis3D, Light3D, Scene3D, Sphere3D, Line3D
export name, rename, add!, move!, scale!, normalize!, render
export scene_pitch, scene_yaw, scene_pan, scene_roll
export world_axis
export Local, Body, Intrinsic, Extrinsic

# ..:: Globals ::..

abstract type AbstractObject3D end

const IntMatrix = Matrix{Int}
const RealTensor = Types.RealTensor #noerr
const OwnerNode{T} = Optional{TreeNode{T}}
const MeshColorSpec = Union{String, Vector{String}}
const MeshWidthSpec = Union{RealValue, RealVector}

"""
List of available frames to reference to:
- `Local` refers to the "local" frame of an object. You can think of it like
  the true frame of the object, regardless of the frame in which the
  geometry/internals of the object are actually defined.
- `Body` refers to the geometry frame of an object, or the projection frame of
  a camera.
"""
@enum(SceneFrame, Local, Body)

"""
Available pose transformation modes:
- `Intrinsic` transforms in succession around the axes of the frame that is in
  the process of being transformed. From Wikipedia [1]: "intrinsic rotations
  are elemental rotations that occur about the axes of a coordinate system XYZ
  attached to a moving body. Therefore, they change their orientation after
  each elemental rotation".
- `Extrinsic` transforms around the static axes of some other frame. From
  Wikipedia [1]: "extrinsic rotations are elemental rotations that occur about
  the axes of the fixed coordinate system xyz. The XYZ system rotates, while
  xyz is fixed".
"""
@enum(PoseUpdateMode, Intrinsic, Extrinsic)

# ..:: Traits ::..

""" Make all scene objects compatible with the tree data structure """
Types.TreeCompatibilityTrait(::Type{<:AbstractObject3D}) = IsTreeCompatible()
Types.owner(x::AbstractObject3D) = begin
    node = x.scene_properties.node
    if isnothing(node)
        msg = @sprintf("Object %s has not been associated with a tree node",
                       name(x))
        throw(ErrorException(msg))
    end
    return node
end

"""
`PoseTrait` describes whether a type can provide a homogeneous transformation
matrix that describes its pose (rotation and translation) in space. There are
two types of poses:

- A "global pose", which describes the translation and rotation of the object
  with respect to another object in the scene. We can also call the global pose
  the "extrinsic pose".

- A "local pose", which is a more special, in that it describes the translation
  and rotation of the object with respect to its own local coordinate
  system. We can also call the local pose the "intrinsic pose".

In other words, we have the following transformation chain:

             local pose               global pose
(body frame) ---------> (local frame) ----------> (other object local frame)

A typical example is for a 3D mesh. After importing the raw mesh, we may want
to rotate/pan/scale it with respect to its local frame. This is done using the
local pose, while the global pose remains unchanged. Objects for which the body
and local frames are the same do not have a local pose. On the other hand, all
scene 3D objects must have a standard pose.

`HasGlobalPose` means that the object has just a global pose. `HasLocalPose`
means that the object has both a global and a local pose.
"""
abstract type PoseTrait end

struct NoPose <: PoseTrait end # struct
struct HasGlobalPose <: PoseTrait end # struct
struct HasLocalPose <: PoseTrait end # struct

""" By default, types do not have a pose. """
PoseTrait(x) = PoseTrait(typeof(x))
PoseTrait(::Type) = NoPose()

""" Any `AbstractObject3D` definitely has a global pose. """
PoseTrait(::Type{<:AbstractObject3D}) = HasGlobalPose()

""" Check if an object has a global pose. """
has_global_pose(x) = has_global_pose(PoseTrait(x), x)
has_global_pose(::Union{HasGlobalPose, HasLocalPose}, x) = true
has_global_pose(::NoPose, x) = false

""" Check if an object has a local pose. """
has_local_pose(x) = has_local_pose(PoseTrait(x), x)
has_local_pose(::HasLocalPose, x) = true
has_local_pose(::Union{NoPose, HasGlobalPose}, x) = false

"""
    get_pose(x[, frame])

Get the pose of an object. `frame` specifies whether you want the pose of the
`Local` frame or of the `Body` frame.

# Arguments
- `x`: a data object that `HasGlobalPose` or `HasLocalPose`.
- `frame`: (optional) get the `Local`, or `Body` frame pose (default is
  `Local`).

# Throws
- `ArgumentError` if `x` has `NoPose`.

# Returns
- `pose`: a homogeneous transformation matrix that defines the relative pose.
"""
get_pose(x, frame::SceneFrame=Local) = begin
    return get_pose(PoseTrait(x), x, frame)
end

get_pose(::NoPose, x::T, frame::SceneFrame) where {T} = begin
    msg = @sprintf("The object %s does not have a pose", T)
    throw(ArgumentError(msg))
end

get_pose(::HasGlobalPose, x, frame::SceneFrame) = (frame==Body) ? homtransf() :
    x.scene_properties.pose

get_pose(::HasLocalPose, x, frame::SceneFrame) = (frame==Body) ? x.local_pose :
    x.scene_properties.pose

"""
    update_pose!(x, value[; frame, mode, reset])

Update the pose of an object's frame.

# Arguments
- `x`: the object whose frame is to be updated.
- `value`: the value to use for the pose update.

# Throws
- `ArgumentError` if the `x` object has `NoPose`.

# Keywords
- `frame`: (optional) which frame to update.
- `mode`: (optional) pose update mode.
- `reset`: (optional) whether to hard-reset the pose to `value`, or to update
  it according to `mode`.
"""
function update_pose!(x, value::RealMatrix;
                      frame::SceneFrame=Local,
                      mode::PoseUpdateMode=Intrinsic,
                      reset::Bool=false)::Nothing
    @assert size(value)==(4, 4)
    update_pose!(PoseTrait(x), x, value, frame, mode, reset)
end # function

update_pose!(::NoPose, x::T, value, frame, mode, reset) where {T} = begin
    msg = @sprintf("The object %s does not have a pose", T)
    throw(ArgumentError(msg))
end

function update_pose!(pose_trait::Union{HasGlobalPose, HasLocalPose},
                      x::AbstractObject3D, value, frame, mode, reset)::Nothing
    if reset
        H = value
    else
        H = get_pose(x, frame)
        H = (mode==Intrinsic) ? H*value : value*H
    end

    if frame==Local || pose_trait isa HasGlobalPose
        x.scene_properties.pose = H
    else
        x.local_pose = H
    end
    return nothing
end # function

# ..:: Data structures ::..

"""
`SceneProperties` holds the fields which all `AbstractObject3D` objects have.
"""
mutable struct SceneProperties{T<:AbstractObject3D}
    name::String        # Object name
    node::OwnerNode{T}  # The tree node which owns the mesh
    pose::RealMatrix    # Relative pose (translation and orientation)

    """
        SceneProperties{T}(name)

    Basic constructor with just a name. Sets a default identity pose (no
    rotation or translation), and no owner node.

    # Arguments
    - `name`: the name identifier string.

    # Returns
    - `props`: the properties object.
    """
    function SceneProperties{T}(name::String)::SceneProperties{T} where {T}
        no_owner = nothing
        default_pose = homtransf()
        props = new{T}(name, no_owner, default_pose)
        return props
    end # function
end # struct

"""
`Mesh3D` defines a 3D mesh that is rendered in the scene. This is a "concrete"
3D object in that it actually appears in the scene.
"""
mutable struct Mesh3D <: AbstractObject3D
    V::RealMatrix              # Vertices
    F::IntMatrix               # Faces association indices
    N::Optional{RealMatrix}    # Face normals (if specified)
    face_color::MeshColorSpec  # Mesh face color ("none" if no color)
    edge_color::MeshColorSpec  # Mesh edge color ("none" if no color)
    edge_width::MeshWidthSpec  # Mesh edge width
    local_pose::RealMatrix  # Pose of body wrt local frame
    scene_properties::SceneProperties{Mesh3D} # 3D scene properties

    """
        Mesh3D(V, F[, N; name, face_color, edge_color, edge_width])

    Basic constructor.

    # Arguments
    - `V`: a 3xN matrix of vertices, where each column is a vertex.
    - `F`: a 3xM or 4xM matrix of faces. If a column has values `[i;j;k;l]`
      then it means that the face is created from the vertices in columns `i`,
      `j`, and `k` of the `V` matrix. If an extra 4th row is present, the value
      `l` corresponds to the column of the `N` matrix which stores the normal
      for this face.
    - `N`: a 3xN matrix of face normals, where each column is a normal.

    # Keywords
    - `name`: (optional) a name for the mesh, which makes it easier to
      reference and to identify what it corresponds to.
    - `face_color`: (optional) color to use for mesh faces. Defaults to none.
    - `edge_color`: (optional) color to use for face edges. Default "black".
    - `edge_width`: (optional) width of mesh edges. Defaults to 0.1.

    # Returns
    - `obj`: the newly created mesh object.
    """
    function Mesh3D(V::RealMatrix,
                    F::IntMatrix,
                    N::Optional{RealMatrix}=nothing;
                    name::String="mesh",
                    face_color::MeshColorSpec="none",
                    edge_color::MeshColorSpec="black",
                    edge_width::MeshWidthSpec=0.1)::Mesh3D

        @assert !(face_color isa Vector) || length(face_color)==size(F, 2)
        @assert !(edge_color isa Vector) || length(edge_color)==size(F, 2)
        @assert !(edge_width isa Vector) || length(edge_width)==size(F, 2)

        default_properties = SceneProperties{Mesh3D}(name)
        default_local_pose = homtransf()

        obj = new(V, F, N, face_color, edge_color, edge_width,
                  default_local_pose, default_properties)

        return obj
    end # function

    """
        Mesh3D(filepath[; name, face_color, edge_color, edge_width])

    Constructor from a Wavefront `obj` file.

    # Arguments
    - `file`: relative or absolute path to the `.obj` file.

    # Keywords
    See the basic constructor docstring.

    # Returns
    - `obj`: the newly created mesh object.
    """
    function Mesh3D(filepath::String;
                    name::String="mesh",
                    face_color::MeshColorSpec="none",
                    edge_color::MeshColorSpec="black",
                    edge_width::MeshWidthSpec=0.1)::Mesh3D

        V, F, N = load_wavefront(filepath)

        obj = Mesh3D(V, F, N, name=name, face_color=face_color,
                     edge_color=edge_color, edge_width=edge_width)

        return obj
    end # function
end # struct

# A mesh has a local pose
PoseTrait(::Type{<:Mesh3D}) = HasLocalPose()

"""
`Camera3D` is a scene camera. Just like a real-life camera, this object is not
rendered in the scene, but rather is the means by which the scene geometry is
projected onto a 2D picture (i.e., "rendered").
"""
mutable struct Camera3D <: AbstractObject3D
    fovy::Real                # [rad] Field of view in the y-direction
    aspect::Real              # Aspect ratio of the camera view plane
    znear::Real               # Distance of the near clip plane
    zfar::Real                # Distance of the far clip plane
    local_pose::RealMatrix    # Pose of camera wrt perspective frame
    scene_properties::SceneProperties{Camera3D} # 3D scene properties

    """
        Camera3D([; fovy, aspect, znear, zfar, name])

    Basic constructor.

    # Keywords
    - `fovy`: (optional) the y-axis field of view opening angle, in degrees.
    - `aspect`: (optional) the view plane aspect ratio.
    - `znear`: (optional) the near clip plane distance in front of camera.
    - `zfar`: (optional) the far clip plane distance in front of camera.
    - `name`: (optional) a name for the camera.

    # Returns
    - `camera`: the camera object.
    """
    function Camera3D(; fovy::Real=25,
                      aspect::Real=1,
                      znear::Real=1,
                      zfar::Real=100,
                      name::String="camera")::Camera3D

        default_properties = SceneProperties{Camera3D}(name)
        default_local_pose = homtransf(pitch=180)

        camera = new(fovy, aspect, znear, zfar, default_local_pose,
                     default_properties)

        return camera
    end # function
end # struct

# A camera has a local pose
PoseTrait(::Type{<:Camera3D}) = HasLocalPose()

"""
`Axis3D` is an axis (aka "frame") scene object. The general use case is that it
is an atomic elemnt that can be used to define a common pose for a set of child
scene objects, without needing to define a concrete rendered object. For
example, `Axis3D` can be a "body frame axis". All `Mesh3D` objects that are
children of the `Axis3D` object then get their pose referred to this
object. Thus, rotating or translating `Axis3D` rotates and translates all the
child meshes, creating a rigid body motion effect. Another use case is for the
`Axis3D` to represent the world frame.
"""
mutable struct Axis3D <: AbstractObject3D
    visible::Bool           # Flag to render the axis
    axis_length::RealValue  # Length of axes when visualized
    axis_width::RealValue   # Thickness of axes when visualized
    scene_properties::SceneProperties{Axis3D} # 3D scene properties

    """
        Axis3D([; name, visible, axis_length, axis_width])

    Basic constructor.

    # Keywords
    - `name`: (optional) a name for the axis.
    - `visible`: (optional) whether to render the axis in the scene.
    - `axis_length`: (optional) the length of the axis mesh, if drawn.
    - `axis_width`: (optional) the width of the axes vectors, if drawn.

    # Returns
    - `axis`: the new axis object.
    """
    function Axis3D(; name::String="axis",
                    visible::Bool=false,
                    axis_length::RealValue=1.0,
                    axis_width::RealValue=0.05)::Axis3D

        default_properties = SceneProperties{Axis3D}(name)

        axis = new(visible, axis_length, axis_width, default_properties)

        return axis
    end # function
end # struct

"""
`Light3D` is a light source object which allows to apply shading to the scene.
"""
mutable struct Light3D <: AbstractObject3D
    az::RealValue # Azimuth angle
    el::RealValue # Elevation angle
    scene_properties::SceneProperties{Light3D} # 3D scene properties

    """
        Light3D(ax, el[; name])

    Basic constructor.

    # Arguments
    - `az`: light source vector azimuth angle (in degrees).
    - `el`: light source vector elevation angle (in degrees).

    # Keywords
    - `name`: (optional) a name for the light source.

    # Returns
    - `light`: the newly created light source.
    """
    function Light3D(az::RealValue,
                     el::RealValue;
                     name::String="light")::Light3D

        @assert az>=0 && az<=360
        @assert el>=0 && el<=90

        default_properties = SceneProperties{Light3D}(name)

        light = new(az, el, default_properties)

        return light
    end # function
end # struct

"""
`ObjectTree` is an alias to a tree node that holds an `Axis3D` object. This
represents the root of the tree representing all objects of the 3D scene. The
root holds an `Axis3D` which represents the "world" coordinate system. The
poses of all objects ultimately refer back to the world system, which servers
to tie together all objects and enable the automatic computation of their
relative poses.
"""
const ObjectTree = TreeNode{Axis3D}

"""
`Scene3D` is the main data structure holding the 3D scene objects, camera, and
other information.
"""
mutable struct Scene3D
    objects::ObjectTree # All the objects in the scene

    """
        Scene3D()

    Default constructor. Creates an empty scene with just a world axis.

    # Returns
    - `scene`: the 3D scene object.
    """
    function Scene3D()::Scene3D

        world_axis = Axis3D(name="cs_world")
        default_scene = TreeNode(world_axis)
        set_owner(world_axis, default_scene)

        scene = new(default_scene)

        return scene
    end # function
end # struct

"""
`BakedScene3D` stores the Matplotlib-ready data for plotting. It stores the
final result of all the heavy computations of projecting the scene onto a
camera. With the `BakedScene3D` object in hand, you can readily make a plot
using Matplotlib. However, this is "raw output" data in the sense that it is
devoid of all the object tree hierarchy, etc., information that was used to
store and manipulate the scene.
"""
mutable struct BakedScene3D
    tris::RealTensor   # Face triangle list
    fc::Vector{String} # Face colors
    ec::Vector{String} # Edge colors
    ew::RealVector     # Edge widths

    """ Default constructor. """
    BakedScene3D(tris, fc, ec, ew) = new(tris, fc, ec, ew)

    """
        BakedScene3D()

    Empty constructor (no geometry to display).

    # Returns
    - `baked`: an "empty" baked scene.
    """
    function BakedScene3D()::BakedScene3D
        tris = RealTensor(undef, 0, 3, 2)
        fc = Vector{String}(undef, 0)
        ec = Vector{String}(undef, 0)
        ew = RealVector(undef, 0)
        baked = new(tris, fc, ec, ew)
        return baked
    end # function
end # struct

# ..:: Methods ::..

"""
    MeshAxis3D(axis)

Generate a mesh to visualize an `Axis3D` object.

# Arguments
- `axis`: the `Axis3D` object to be visualized.

# Returns
- `axis_mesh`: an axis mesh with the same pose as `axis`.
"""
function MeshAxis3D(axis::Axis3D)::Mesh3D

    # Generate mesh along +x
    Vx, Fx = make_x_axis_mesh(axis.axis_width, axis.axis_length)
    fcx = repeat([Red], size(Fx, 2))

    # Generate mesh along +y
    Vy, Fy = make_x_axis_mesh(axis.axis_width, axis.axis_length)
    Vy = homrot(scene_yaw(90))*Vy
    Fy .+= size(Vx, 2)
    fcy = repeat([Green], size(Fy, 2))

    # Generate mesh along +z
    Vz, Fz = make_x_axis_mesh(axis.axis_width, axis.axis_length)
    Vz = homrot(scene_pitch(-90))*Vz
    Fz .+= size(Vx, 2)+size(Vy, 2)
    fcz = repeat([Blue], size(Fz, 2))

    # Concatenate meshes
    V = cat(Vx, Vy, Vz, dims=2)
    F = cat(Fx, Fy, Fz, dims=2)
    fc = cat(fcx, fcy, fcz, dims=1)

    # Create mesh object
    axis_mesh = Mesh3D(V, F, face_color=fc, edge_color="none", edge_width=0)

    return axis_mesh
end # function

"""
    Sphere3D(r[; az, el, name, face_color,
             edge_color, edge_width])

Generate a spherical mesh of a given radius.

# Arguments
- `r`: the sphere radius.

# Keywords
- `az`: (optional) number of steps in the azimuth direction (which spans 0 to
  360 degrees about the `z` axis).
- `el`: (optional) number of steps in the elevation direction (which spans -90 to
  90 degrees "climbing" along the `z` axis).
- For the other arguments, see the `Mesh3D` basic constructor.

# Returns
- `sphere`: the sphere `Mesh3D` object.
"""
function Sphere3D(r::Real;
                  az::Int=30,
                  el::Int=20,
                  name::String="sphere",
                  face_color::MeshColorSpec="none",
                  edge_color::MeshColorSpec="black",
                  edge_width::MeshWidthSpec=0.1)::Mesh3D

    # Parameters
    az_vals = LinRange(0, 2*pi, az+1)
    el_vals = LinRange(-pi/2, pi/2, el)
    coord = (el, az) -> [r*cos(el)*cos(az); r*cos(el)*sin(az); r*sin(el)]

    # Generate the vertices
    V = RealMatrix(undef, 3, 2+(el-2)*az)
    V2I = Dict{RealVector, Int}()
    k = 1
    for i = 1:el
        if i==1 || i==el
            V[:, k] = ((i==1) ? -1 : 1)*[0; 0; r]
            V2I[V[:, k]] = k
            k += 1
        else
            φ = el_vals[i]
            for j = 1:az
                θ = az_vals[j]
                V[:, k] = coord(φ, θ)
                V2I[V[:, k]] = k
                k += 1
            end
        end
    end

    # Generate the faces
    F = Vector{Types.IntVector}(undef, 0)
    for i = 2:el
        if i==2
            vbot = V2I[-1*[0; 0; r]]
            for j = 1:az
                jn = (j==az) ? 1 : j+1
                v3 = V2I[coord(el_vals[i], az_vals[jn])]
                v4 = V2I[coord(el_vals[i], az_vals[j])]
                push!(F, [vbot; v3; v4])
            end
        elseif i==el
            vtop = V2I[[0; 0; r]]
            ip = i-1
            for j = 1:az
                jn = (j==az) ? 1 : j+1
                v3 = V2I[coord(el_vals[ip], az_vals[jn])]
                v4 = V2I[coord(el_vals[ip], az_vals[j])]
                push!(F, [v4; v3; vtop])
            end
        else
            ip = i-1
            for j = 1:az
                jn = (j==az) ? 1 : j+1
                v1 = V2I[coord(el_vals[ip], az_vals[j])]
                v2 = V2I[coord(el_vals[ip], az_vals[jn])]
                v3 = V2I[coord(el_vals[i], az_vals[jn])]
                v4 = V2I[coord(el_vals[i], az_vals[j])]
                push!(F, [v1; v2; v4])
                push!(F, [v2; v3; v4])
            end
        end
    end
    F = hcat(F...)

    # Make the mesh object
    sphere = Mesh3D(V, F,
                    name=name,
                    face_color=face_color,
                    edge_color=edge_color,
                    edge_width=edge_width)

    return sphere
end # function

"""
    Line3D(v1, v2[, nseg][; name, face_color,
           edge_color, edge_width])

Generate a straight line mesh.

# Arguments
- `vs`: one endpoint.
- `vs`: second endpoint.
- `nseg`: (optional) the number of segments composing the line.

# Keywords
- For the other arguments, see the `Mesh3D` basic constructor.

# Returns
- `line`: the line `Mesh3D` object.
"""
function Line3D(v1::RealVector,
                v2::RealVector,
                nseg::Int=100;
                name::String="sphere",
                edge_color::MeshColorSpec="black",
                edge_width::MeshWidthSpec=0.1)

    # Parameters
    vx = LinRange(v1[1], v2[1], nseg+1)
    vy = LinRange(v1[2], v2[2], nseg+1)
    vz = LinRange(v1[3], v2[3], nseg+1)

    # Generate the vertices
    V = RealMatrix(undef, 3, nseg+1)
    V2I = Dict{RealVector, Int}()
    k = 1
    for i = 1:nseg+1
        V[:, k] = [vx[i]; vy[i]; vz[i]]
        V2I[V[:, k]] = k
        k += 1
    end

    # Generate the faces
    F = Vector{Types.IntVector}(undef, 0)
    for i = 2:nseg+1
        v1 = V2I[[vx[i-1]; vy[i-1]; vz[i-1]]]
        v2 = V2I[[vx[i]; vy[i]; vz[i]]]
        push!(F, [v1; v1; v2])
    end
    F = hcat(F...)

    # Make the mesh object
    line = Mesh3D(V, F,
                  name=name,
                  edge_color=edge_color,
                  edge_width=edge_width)

    return line
end # function

"""
    make_x_axis_mesh([; thickness, length, resol])

Make the vertex and face arrays for an axis aligned with the x axis.

# Arguments
- `thickness`: axis thickness (length in the y and z directions).
- `length`: axis length along the +x direction.

# Keywords
- `resol`: (optional) triangulation resolut (i.e., number of segments) along
  the +x direction.

# Returns
- `V`: the vertex matrix.
- `F`: the face association matrix.
"""
function make_x_axis_mesh(thickness::RealValue,
                          length::RealValue;
                          resol::Int=50)::Tuple{RealMatrix, IntMatrix}

    ymin, ymax = -thickness/2, thickness/2
    zmin, zmax = -thickness/2, thickness/2
    xgrid = LinRange(0, length, resol)

    # Generate the vertices
    V = RealMatrix(undef, 3, 4*resol)
    V2I = Dict{RealVector, Int}()
    k = 0
    for i=1:resol
        x = xgrid[i]
        v = [[x; ymin; zmin],
             [x; ymin; zmax],
             [x; ymax; zmax],
             [x; ymax; zmin]]
        for j=1:4
            V[:, k+j] = v[j]
            V2I[v[j]] = k+j
        end
        k += 4
    end

    # Generate the faces
    F = Vector{Types.IntVector}(undef, 0)
    for i=1:resol
        x = xgrid[i]
        if i==1 || i==resol
            # Endcaps
            i1 = V2I[[x, ymin, zmin]]
            i2 = V2I[[x, ymax, zmin]]
            i3 = V2I[[x, ymax, zmax]]
            push!(F, [i1; i2; i3])
            i2 = V2I[[x, ymin, zmax]]
            push!(F, [i1; i2; i3])
        end
        if i<resol
            xn = xgrid[i+1]
            # Switch the sides
            # +y
            i1 = V2I[[x, ymax, zmin]]
            i2 = V2I[[xn, ymax, zmin]]
            i3 = V2I[[x, ymax, zmax]]
            push!(F, [i1; i2; i3])
            i1 = V2I[[xn, ymax, zmax]]
            push!(F, [i1; i2; i3])
            # -y
            i1 = V2I[[x, ymin, zmin]]
            i2 = V2I[[xn, ymin, zmin]]
            i3 = V2I[[x, ymin, zmax]]
            push!(F, [i1; i2; i3])
            i1 = V2I[[xn, ymin, zmax]]
            push!(F, [i1; i2; i3])
            # +z
            i1 = V2I[[x, ymax, zmax]]
            i2 = V2I[[xn, ymax, zmax]]
            i3 = V2I[[xn, ymin, zmax]]
            push!(F, [i1; i2; i3])
            i2 = V2I[[x, ymin, zmax]]
            push!(F, [i1; i2; i3])
            # -z
            i1 = V2I[[x, ymax, zmin]]
            i2 = V2I[[xn, ymax, zmin]]
            i3 = V2I[[xn, ymin, zmin]]
            push!(F, [i1; i2; i3])
            i2 = V2I[[x, ymin, zmin]]
            push!(F, [i1; i2; i3])
        end
    end
    F = hcat(F...)

    return V, F
end # function

""" Aliases to create homogeneous transformation matrices. """
scene_roll(roll::RealValue; deg::Bool=true) = homtransf(roll=roll, deg=deg)
scene_pitch(pitch::RealValue; deg::Bool=true) = homtransf(pitch=pitch, deg=deg)
scene_yaw(yaw::RealValue; deg::Bool=true) = homtransf(yaw=yaw, deg=deg)
scene_pan(; x::RealValue=0.0, y::RealValue=0.0,
          z::RealValue=0.0) = homtransf(RealVector([x; y; z]))

""" Get the object's name. """
name(obj::AbstractObject3D) = obj.scene_properties.name

""" Rename an object. """
rename(obj::AbstractObject3D, name::String) = obj.scene_properties.name=name

""" Set owning node of an object in the object tree. """
set_owner(obj::AbstractObject3D, node::TreeNode) = obj.scene_properties.node=node

""" Get the world frame axis "root" of the object tree. """
world_axis(scene::Scene3D) = scene.objects.data

"""
    add!(parent, obj...)

Add object(s) to the scene. `parent` specifies the parent object, which can be
a `Scene3D` or any `AbstractObject3D`. In the former case, the object is
associated with the world frame. In the latter case, the object is made a child
of the `parent` object, and thus associated with the `parent`'s frame.

# Arguments
- `parent`: the parent object.
- `obj...`: one or more objects or object groups to be added.

# Throws
- `ArgumentError` if `obj` is already in a scene object tree. We cannot add
  duplicate objects to the scene.
- `ErrorException` if `parent` is not associated with a scene object tree.
"""
function add!(parent::Union{Scene3D, AbstractObject3D},
              obj::AbstractObject3D...)::Nothing
    if length(obj)==1
        obj = obj[1]

        # Check that `obj` is not already associated with a tree
        owner_exists = false
        try
            owner(obj)
            owner_exists = true
        catch
        end

        if owner_exists
            fmt = "Object %s is already associated with a node, "*
                "cannot add it again"
            msg = @eval @sprintf($fmt, name($obj))
            throw(ArgumentError(msg))
        end

        # Get the parent node
        if parent isa Scene3D
            parent_node = parent.objects
        else
            parent_node = owner(parent)
        end

        # Add a new child node holding `obj`
        new_node = TreeNode(obj, parent_node)
        set_owner(obj, new_node)
    else
        for obji in obj
            add!(parent, obji)
        end
    end
    return nothing
end # function

"""
    move!(obj[; frame, mode, kwargs...])

Provide a sequence of commands that update the object's translation and
rotation. This operation works to transform the pose of the specified `frame`,
which is either `Local` or `Body`. The `mode` parameter specifies whether to
perform an intrinsic transformation (transform around `frame`) or an extrinsic
transformation (transform around the parent frame for `frame==Local` or the
local frame for `frame==Body`).

# Arguments
- `obj`: the object to transform the pose of.

# Keywords
- `frame`: (optional) the frame whose pose is to be changed. Can be either the
  `Local` or `Body` frames which belong to `obj`.
- `mode`: (optional) the transformation mode. Setting `Extrinsic` uses the axes
  of the parent frame to do the transformation (i.e., parent node's `Local`
  frame if `frame==Local`, and the object's `Local` frame if
  `frame==Body`). Setting `Extrinsic` uses the axes of `frame` *as it is being
  transformed* in order to perform the sequence of transformations.
- `kwargs`: the pose transformation sequence. The possible commands are:
    - `H`: a homogeneous transformation matrix.
    - `x`: translate along `x` axis.
    - `y`: translate along `y` axis.
    - `z`: translate along `z` axis.
    - `roll`: rotate around `x` axis.
    - `pitch`: rotate around `y` axis.
    - `yaw`: rotate around `z` axis.
    - `degree`: a boolean flag whether to treat angles as degrees.

# Throws
- `ErrorException` if the object is not associated with the object tree.
"""
function move!(obj::AbstractObject3D;
               frame::SceneFrame=Local,
               mode::PoseUpdateMode=Intrinsic,
               kwargs...)::Nothing

    owner(obj) # Check that object is associated with object tree

    sequence = collect(kwargs)
    degrees = ((:degree in sequence && sequence[:degree]) ||
        !(:degree in sequence))

    # Perform the sequence of operations
    H_update = homtransf()
    for (op, val) in sequence
        # Compute the atomic transformation
        if op==:H
            H_atom = val
        elseif op==:x
            H_atom = scene_pan(x=val)
        elseif op==:y
            H_atom = scene_pan(y=val)
        elseif op==:z
            H_atom = scene_pan(z=val)
        elseif op==:roll
            H_atom = scene_roll(val, deg=degrees)
        elseif op==:pitch
            H_atom = scene_pitch(val, deg=degrees)
        elseif op==:yaw
            H_atom = scene_yaw(val, deg=degrees)
        end
        # Apply to the overall transformation
        if mode==Intrinsic
            H_update = H_update*H_atom
        else
            H_update = H_atom*H_update
        end
    end

    # Apply the transformation to the object's frame
    update_pose!(obj, H_update; frame=frame, mode=mode)

    return nothing
end # function

"""
    scale!(obj, amount[; relative])

Scale the mesh relative to the centroid of its vertices by `amount`. By default
the scaling is absolute, which means that the scaled mesh's maximum size along
each axis is equal to `value`. You can also do relative scaling, which means
that the mesh grows or shrink by the specified amount with respect to its
original size.

# Arguments
- `obj`: the `Mesh3D` object.
- `amount`: the scaling amount.

# Keywords
- `relative`: (optional) do relative scaling.
"""
function scale!(obj::Mesh3D, amount::RealValue;
                relative::Bool=false)::Nothing
    centroid = (maximum(obj.V, dims=2)+minimum(obj.V, dims=2))/2
    if !relative
        span = maximum(maximum(obj.V, dims=2)-minimum(obj.V, dims=2))
        obj.V = (((obj.V).-centroid)/span)*amount
    else
        obj.V = (((obj.V).-centroid)*amount).+centroid
    end
    return nothing
end # function

""" Normalize the mesh to maximum length 1 along each dimension. """
normalize!(obj::Mesh3D) = scale!(obj, 1; relative=false)

"""
    relative_pose(objA, objB[, src, dest])

Get the relation pose between `src` frame of object `objA` and `dest` frame of
object `objB`. The returned homogeneous transformation matrix `H` transforms
homogeneous vectors from `objA`'s frame to `objB`'s frame. You can specify
`Local` and `Body` for `src` and `dest`. The default for both is `Body`.

# Arguments
- `objA`: the source object.
- `objB`: the destination object.
- `src`: (optional) the source object's frame.
- `dest`: (optional) the destination object's frame.

# Returns
- `bar`: description.
"""
function relative_pose(objA::AbstractObject3D,
                       objB::AbstractObject3D,
                       src::SceneFrame=Local,
                       dest::SceneFrame=Local)::RealMatrix

    # Find a common ancestor node
    nodeA = owner(objA)
    nodeB = owner(objB)
    nodeC = find_common(nodeA, nodeB)

    # Find the relative pose between local A and local B frames
    H_CA = relative_local_pose_linear(nodeA, nodeC)
    H_CB = relative_local_pose_linear(nodeB, nodeC)
    H_BA = hominv(H_CB)*H_CA

    # Update for body frame choice
    if src==Body
        H_BA = H_BA*get_pose(objA, Body)
    end
    if dest==Body
        H_BA = hominv(get_pose(objB, Body))*H_BA
    end

    return H_BA
end # function

"""
    relative_local_pose_linear(descendant, ancestor)

Get the relative pose from the local frame of a `descendant` node to the local
frame of the `ancestor` node. The `ancestor` should be encountered on the path
of nodes climbing from the `descendant` up to the object tree root. If it is
not, an error is thrown

# Arguments
- `descendant`: a node that is lower down the object tree.
- `ancesotr`: a node that is higher up the object tree.

# Throws
- `ArgumentError` if `ancestor` is not encountered among the nodes climbing up
  from `descendant`.

# Returns
- `H`: homogeneous transform from `descendant` local frame to `ancestor` local
  frame.
"""
function relative_local_pose_linear(
    descendant::TreeNode,
    ancestor::TreeNode)::RealMatrix

    H = homtransf()
    node = descendant

    while node!=ancestor
        if is_root(node) && node!=ancestor
            msg = @sprintf("%s not an ancestor of %s",
                           name(ancestor.data), name(descendant.data))
            throw(ArgumentError(msg))
        end
        obj = node.data
        Hi = get_pose(obj, Local)
        H = Hi*H
        node = node.parent
    end

    return H
end # function

"""
    perspective(camera)

Compute the perspective transformation matrix for the camera.

# Arguments
- `camera`: the camera object.

# Returns
- `P`: the perspective transformation 4x4 matrix.
"""
function perspective(camera::Camera3D)::RealMatrix
    h = tan(0.5*deg2rad(camera.fovy))*camera.znear
    w = h*camera.aspect
    P = frustum(-w, w, -h, h, camera.znear, camera.zfar)
    return P
end # function

"""
    frustrum(left, right, bottom, top, near, far)

Define a camera perspective transformation based on six planes that define a
view area bounding box. For documentation on the reasoning and math behind this
computation, see [1].

References

[1] http://learnwebgl.brown37.net/08_projections/projections_perspective.html

# Arguments
- `left`: near clip plane left edge distance from principal axis.
- `right`: near clip plane right edge distance from principal axis.
- `top`: near clip plane top edge distance from principal axis.
- `bottom`: near clip plane bottom edge distance from principal axis.
- `near`: near clip plane distance from camera center.
- `far`: far clip plane distance from camera center.

# Returns
- `P`: the 4x4 perspective transformation matrix.
"""
function frustum(left::Real, right::Real, bottom::Real, top::Real,
                 near::Real, far::Real)::RealMatrix
    P = zeros(4, 4)
    P[1, 1] = 2*near/(right-left)
    P[2, 2] = 2*near/(top-bottom)
    P[3, 3] = -(far+near)/(far-near)
    P[1, 4] = -near*(right+left)/(right-left)
    P[2, 4] = -near*(top+bottom)/(top-bottom)
    P[3, 4] = 2*near*far/(near-far)
    P[4, 3] = -1
    return P
end # function

"""
    load_wavefront(filepath)

Extract the vertex and face arrays from a wavefront object mesh definition. For
a description of how `V` and `F` are to be interpreted, see the docstring of
the `Object3D` basic constructor.

# Arguments
- `filepath`: the relative or absolute path to the `.obj` file.

# Returns
- `V`: floating point array of vertices.
- `F`: integer array of faces.
- `N`: floating point array for face normals.
"""
function load_wavefront(filepath::String)::Tuple{RealMatrix,
                                                 IntMatrix,
                                                 Optional{RealMatrix}}
    V, F, N = [], [], []
    f = open(filepath, "r")
    for line in readlines(f)
        if startswith(line, "#")
            continue
        end
        values = split(line) #noinfo
        if isempty(values)
        elseif values[1]=="v"
            push!(V, [parse(Float64, x) for x in values[2:4]])
        elseif values[1]=="vn"
            push!(N, [parse(Float64, x) for x in values[2:4]])
        elseif values[1]=="f"
            values = map(s->split(s, "//"), values[2:4]) #noinfo
            vertices = map(v->v[1], values)
            if length(values[1])==2
                # There is face normal information
                normal = values[1][2]
                push!(vertices, normal)
            end
            push!(F, [parse(Int, x) for x in vertices])
        end
    end
    close(f)
    V = hcat(V...)
    F = hcat(F...)
    N = isempty(N) ? nothing : hcat(N...)
    return V, F, N
end # function

"""
    render(scene[, camera, path][; canvas_size, canvas_xlim, canvas_ylim,
           canvas_ aspect])

Render the scene.

# Arguments
- `scene`: the 3D scene definition object.
- `camera`: (optional) the camera to use for rendering. Can explicitly provide
  a camera object, or give the camera name (and the matching one will be
  found). Can also not provide anything, in which case a camera will be found
  automatically. In all cases, if there is more than one camera matched, an
  error will be thrown due to ambiguity.

# Keywords
- `canvas_size`: (optional) size of the render window.
- `bg_color`: (optional) the background color, specified as an RGB/RGBA vector
  or a string.

# Throws
- `ArgumentError` if not exactly one camera is found in the scene, when the
  camera to use is specified as a string.
"""
function render(scene::Scene3D,
                camera::Optional{Union{Camera3D, String}}=nothing,
                path::String="./figures/scene3d_render.pdf";
                canvas_size::Tuple{Real, Real}=(5, 5),
                bg_color=zeros(4))::Nothing

    # Get the camera
    if camera isa String || isnothing(camera)

        # Find matching cameras
        if camera isa String
            match_camera = (obj) -> obj isa Camera3D && name(obj)==camera
        else
            match_camera = (obj) -> obj isa Camera3D
        end
        all_cameras = findall(match_camera, scene.objects)

        # Extract the camera object
        if length(all_cameras)>1
            msg = @sprintf("Too many cameras named \"%s\" found (%d)",
                           camera, length(all_cameras))
            throw(ArgumentError(msg))
        elseif length(all_cameras)==0
            msg = @sprintf("No camera \"%s\" found", camera)
            throw(ArgumentError(msg))
        else
            camera = all_cameras[1]
        end
    end

    # Initialize the figure
    fig = create_figure(canvas_size)

    # Add an axis
    ax = fig.add_axes([0, 0, 1, 1],
                      xlim=(-1, 1),
                      ylim=(-1, 1),
                      aspect=1/camera.aspect,
                      frameon=false)

    # Sanitize axes
    ax.set_xticks([])
    ax.set_yticks([])

    # Generate the mesh triangles
    scene_data = bake(scene, camera)

    PolyCollection = PyPlot.matplotlib.collections.PolyCollection
    scene_baked = PolyCollection(
        scene_data.tris,
        closed=true,
        linewidths=scene_data.ew,
        edgecolors=scene_data.ec,
        facecolors=scene_data.fc,
        capstyle="round",
        joinstyle="round")

    ax.add_collection(scene_baked)

    # Save figure to file
    save_figure(path,
                tight_layout=false,
                dpi=800,
                facecolor=bg_color)

    return nothing
end # function

"""
    bake(scene, camera)

Project `Scene3D` onto a `Camera3D`, generating data which is readily plotted
using Matplotlib. This function performs z-depth sorting so that objects
overlap appropriately according to their 3D position.

# Arguments
- `scene`: the 3D scene definition object.
- `camera`: the camera object to use for rendering the scene.

# Returns
- `baked`: the projected scene data that can be used for plotting.
"""
function bake(scene::Scene3D, camera::Camera3D)::BakedScene3D

    # >> Collect data on all objects into lists <<

    tris_3d = RealTensor[]
    fc = Vector{String}[]
    ec = Vector{String}[]
    ew = RealVector[]

    # Perspective projection to camera
    persp = perspective(camera)

    traverse(scene.objects) do obj, _
        # Do not render cameras and invisible axes
        if (obj isa Camera3D ||
            obj isa Light3D ||
            (obj isa Axis3D && !obj.visible))
            return
        end

        # Get the raw mesh in its body frame
        draw_axis = obj isa Axis3D
        if draw_axis
            # Generate a (temporary) mesh representing the axis
            ax_obj = obj
            obj = MeshAxis3D(ax_obj)
            add!(ax_obj, obj)
        end
        verts, faces, normals = obj.V, obj.F, obj.N

        # Relative pose
        rel_pose = relative_pose(obj, camera, Body, Body)

        # Perspective projection
        proj = persp*rel_pose
        verts = proj*[verts; ones(1, size(verts, 2))]
        verts ./= reshape(verts[end, :], 1, :)

        # Extract the face triangles
        num_faces = size(faces, 2)
        tris = RealTensor[]
        id_face_keep = Int[]
        for i=1:num_faces
            face_verts = verts[1:3, faces[1:3, i]]
            # Filter out faces outside the clipping volume
            outside_verts = (face_verts.>1) .| (face_verts.<-1)
            if any(all(outside_verts[i, :]) for i=1:3)
                continue
            end
            # Filter out faces whose normal points away from the camera
            # "backface culling"
            if !isnothing(normals)
                base = face_verts[:, 1]
                normal = normals[:, faces[4, i]]
                tip = base+normal
                proj_base = proj*[base; 1]
                proj_tip = proj*[tip; 1]
                proj_base ./= proj_base[4]
                proj_tip ./= proj_tip[4]
                proj_normal = (proj_base-proj_tip)[1:3]
                if proj_normal[3]<-0.05 # some buffer (exact would be <0)
                    continue
                end
            end
            push!(tris, reshape(face_verts', 1, 3, 3))
            push!(id_face_keep, i)
        end
        tris = cat(tris..., dims=1)
        num_faces = size(tris, 1)

        # Everything is clipped, no geometry to display
        if num_faces==0
            return
        end

        # Find the light source
        light = find_light(obj)
        let_there_be_light = !isnothing(light) && !isnothing(normals)
        if let_there_be_light
            light = PyPlot.matplotlib.colors.LightSource(
                azdeg=light.az, altdeg=light.el)
        end

        # Illuminate the face and edge colors
        face_colors = (obj.face_color isa Vector) ?
            obj.face_color : repeat([obj.face_color], size(faces, 2))
        edge_colors = (obj.edge_color isa Vector) ?
            obj.edge_color : repeat([obj.edge_color], size(faces, 2))
        if let_there_be_light
            # Update face colors according to illumination by the light source
            normals_array = hcat(map(i->normals[:, i], faces[4, :])...)'
            shade = light.shade_normals(normals_array, fraction=1.0)
            for i = 1:size(faces, 2)
                dark_fraction = 1-shade[i]
                face_colors[i] = "#"*hex(RGB(darken_color(
                    face_colors[i], dark_fraction)[1:3]...))
                edge_colors[i] = "#"*hex(RGB(darken_color(
                    edge_colors[i], dark_fraction)[1:3]...))
            end
        end
        face_colors = face_colors[id_face_keep]
        edge_colors = edge_colors[id_face_keep]

        # Save to concatenated list
        push!(tris_3d, tris)
        push!(fc, face_colors)
        push!(ec, edge_colors)
        push!(ew, (obj.edge_width isa Vector) ? obj.edge_width[id_face_keep] :
            repeat([obj.edge_width], num_faces))

        # Remove temporary axis mesh
        if draw_axis
            remove_child!(owner(ax_obj), owner(obj))
        end
    end

    # No geometry to display
    if isempty(tris_3d)
        out = BakedScene3D()
        return out
    end

    tris_3d = cat(tris_3d..., dims=1)
    tris = mapslices(tris_3d, dims=[2, 3]) do verts
        verts[:, 1:2]
    end
    fc = cat(fc..., dims=1)
    ec = cat(ec..., dims=1)
    ew = cat(ew..., dims=1)

    # >> Do z-depth sorting <<

    faces_z = mapslices(tris_3d, dims=[2, 3]) do verts
        verts_z = verts[:, 3]
        mean(verts_z)
    end
    faces_z = -squeeze(faces_z)
    zid = sortperm(faces_z)

    tris = tris[zid, :, :]
    fc = fc[zid]
    ec = ec[zid]
    ew = ew[zid]

    out = BakedScene3D(tris, fc, ec, ew)

    return out
end # function

"""
    find_light(obj)

Find a light source that should be used for illuminating the provided
object. This is done by finding the light source lowest on the tree that is
some sibling or distance sibling of the object. The function proceeds all the
way until the object tree's root, at which point it stops if there is still no
light source found.

# Arguments
- `obj`: the object for which a light source is to be provided.

# Returns
- `light`: the light source, or `nothing` if none.

# Throws
- `ErrorException` if a node is encountered with more than one `Light3D`
  child. This is an ambiguous case (which light to pick?) that we do not
  support.
"""
function find_light(obj::AbstractObject3D)::Optional{Light3D}

    parent = owner(obj)

    while true
        # Find any child nodes that are lights
        child_lights = findall(parent) do data
            return data isa Light3D
        end

        if length(child_lights)>1
            err = ErrorException(
                @sprintf("%s has %d light sources (only 1 allowed)",
                         name(parent), length(child_lights)))
            throw(err)
        elseif length(child_lights)==1
            light = child_lights[1]
            return light
        elseif is_root(parent)
            # At the object tree root - no light source found
            return nothing
        else
            # Go up a tree level
            parent = parent.parent
        end
    end
end # function

"""
    show(io, obj)

Pretty print a 3D mesh object.

# Arguments
- `io`: stream object.
- `obj`: an `Mesh3D` object.
"""
function Base.show(io::IO, obj::Mesh3D)::Nothing
    compact = get(io, :compact, false) #noinfo
    indent = make_indent(io)

    @preprintf(io, indent, "Mesh (%s)", name(obj))

    if !compact
        @preprintf(io, indent, "\n")
        @preprintf(io, indent, "%d vertices\n", size(obj.V, 2))
        @preprintf(io, indent, "%d faces", size(obj.F, 2))
    end

    return nothing
end # function

"""
    show(io, scene)

Pretty print a scene.

# Arguments
- `io`: stream object.
- `obj`: a `Scene3D` object.
"""
function Base.show(io::IO, scene::Scene3D)::Nothing
    compact = get(io, :compact, false) #noinfo
    indent = make_indent(io)

    # >> Print the object counts <<

    match_axis = (obj) -> typeof(obj)==Axis3D
    match_camera = (obj) -> typeof(obj)==Camera3D
    match_light = (obj) -> typeof(obj)==Light3D
    match_mesh = (obj) -> typeof(obj)==Mesh3D

    all_axes = findall(match_axis, scene.objects)
    all_cameras = findall(match_camera, scene.objects)
    all_lights = findall(match_light, scene.objects)
    all_meshes = findall(match_mesh, scene.objects)

    @preprintf(io, indent, "%d cameras\n", length(all_cameras))
    @preprintf(io, indent, "%d lights\n", length(all_lights))
    @preprintf(io, indent, "%d axes\n", length(all_axes))
    @preprintf(io, indent, "%d meshes\n", length(all_meshes))

    vertex_count = 0
    face_count = 0
    traverse(scene.objects) do obj, _
        if obj isa Axis3D && obj.visible
            ax_mesh = MeshAxis3D(obj)
            vertex_count += size(ax_mesh.V, 2)
            face_count += size(ax_mesh.F, 2)
        elseif obj isa Mesh3D
            vertex_count += size(obj.V, 2)
            face_count += size(obj.F, 2)
        end
    end

    @preprintf(io, indent, "\n")
    @preprintf(io, indent, "%d vertices\n", vertex_count)
    @preprintf(io, indent, "%d faces\n", face_count)

    # >> Print the object tree hierarchy <<
    full_tree = String[]
    @preprintf(io, indent, "\nObject tree:\n")
    traverse(scene.objects) do obj, depth
        local_indent = indent*(" "^(2*(depth+1)))
        push!(full_tree, @sprintf("%s%s", local_indent, name(obj)))
    end
    # Remove repeated lines
    line = 1
    short_tree = ""
    while true
        this_line = full_tree[line]
        count_instances = 1
        for next_line = line+1:length(full_tree)
            if full_tree[next_line]==this_line
                count_instances += 1
            else
                break
            end
        end
        if count_instances>1
            short_tree *= @sprintf("%s (%d)\n", this_line, count_instances)
        else
            short_tree *= @sprintf("%s\n", this_line)
        end
        line += count_instances
        if line>length(full_tree)
            break
        end
    end
    println(short_tree)

    return nothing
end # function

"""
    show(io, mime, obj)

A wrapper of `show` functions when `MIME` argument is passed in (it is just
ignored!).
"""
Base.show(io::IO, ::MIME"text/plain",
          constraints::Union{Mesh3D,
                             Scene3D},
          args...; kwargs...) = show(io, constraints, args...; kwargs...)
