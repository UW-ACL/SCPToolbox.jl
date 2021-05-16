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

export Mesh3D, Camera3D, Axis3D, Scene3D
export name, rename, add!, move!, render
export scene_pitch, scene_yaw, scene_pan, scene_roll
export world_axis

# ..:: Globals ::..

abstract type AbstractObject3D end

const IntMatrix = Matrix{Int}
const RealTensor = Types.RealTensor #noerr
const OwnerNode{T} = Optional{TreeNode{T}}
const PolyCollection = PyPlot.matplotlib.collections.PolyCollection

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
`:local` frame or of the `:body` frame.

# Arguments
- `x`: a data object that `HasGlobalPose` or `HasLocalPose`.
- `frame`: (optional) get the `:local`, or `:body` frame pose (default is
  `local`).

# Throws
- `ArgumentError` if `x` has `NoPose`.

# Returns
- `pose`: a homogeneous transformation matrix that defines the relative pose.
"""
get_pose(x, frame::Symbol=:local) = begin
    @assert frame in (:local, :body)
    return get_pose(PoseTrait(x), x, frame)
end

get_pose(::NoPose, x::T, frame::Symbol) where {T} = begin
    msg = @sprintf("The object %s does not have a pose", T)
    throw(ArgumentError(msg))
end

get_pose(::HasGlobalPose, x, frame::Symbol) = begin
    if frame==:body
        return homtransf()
    else
        return x.scene_properties.pose
    end
end

get_pose(::HasLocalPose, x, frame::Symbol) = begin
    if frame==:body
        return x.local_pose
    else
        return x.scene_properties.pose
    end
end

"""
    Signature

Description.

# Arguments
- `foo`: description.

# Returns
- `bar`: description.
"""
update_pose(x, value::RealMatrix, frame::Symbol=:local,
            mode::Symbol=:intrinsic) = begin
                @assert frame in (:local, :body)
                @assert mode in (:extrinsic, :intrinsic)
                @assert size(value)==(4, 4)
                update_pose(PoseTrait(x), x, value, frame, mode)
            end

update_pose(::NoPose, x::T, value, frame, mode) where {T} = begin
    msg = @sprintf("The object %s does not have a pose", T)
    throw(ArgumentError(msg))
end

update_pose(T::Union{HasGlobalPose, HasLocalPose}, x, value, frame, mode) =
    begin
        H = get_pose(x, frame)
        H = (mode==:intrinsic) ? H*value : value*H
        if frame==:local || T isa HasGlobalPose
            x.scene_properties.pose = H
        else
            x.local_pose = H
        end
    end

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
    V::RealMatrix           # Vertices
    F::IntMatrix            # Faces association indices
    local_pose::RealMatrix  # Pose of body wrt local frame
    scene_properties::SceneProperties{Mesh3D} # 3D scene properties

    """
        Mesh3D(V, F[; name])

    Basic constructor.

    # Arguments
    - `V`: a 3xN matrix of vertices, where each column is a vertex.
    - `F`: a 3xM matrix of faces. If a column has values `[i;j;k]` then it
      means that the face is created from the vertices in columns `i`, `j`, and
      `k` of the `V` matrix.
    - `name`: (optional) a name for the mesh, which makes it easier to
      reference and to identify what it corresponds to.

    # Returns
    - `obj`: the newly created mesh object.
    """
    function Mesh3D(V::RealMatrix,
                    F::IntMatrix;
                    name::String="obj")::Mesh3D

        default_properties = SceneProperties{Mesh3D}(name)
        default_local_pose = homtransf()

        obj = new(V, F, default_local_pose, default_properties)

        return obj
    end # function

    """
        Mesh3D(filepath[; name])

    Constructor from a Wavefront `obj` file.

    # Arguments
    - `file`: relative or absolute path to the `.obj` file.
    - `name`: (optional) a name for the mesh.

    # Returns
    - `obj`: the newly created mesh object.
    """
    function Mesh3D(filepath::String;
                    name::String="mesh")::Mesh3D

        V, F = load_wavefront(filepath)

        obj = Mesh3D(V, F, name=name)

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

    # Arguments
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
    visible::Bool                             # Flag to render the axis
    scene_properties::SceneProperties{Axis3D} # 3D scene properties

    """
        Axis3D([; name, visible])

    Basic constructor.

    # Keywords
    - `name`: (optional) a name for the axis.
    - `visible`: (optional) whether to render the axis in the scene.

    # Returns
    - `axis`: the new axis object.
    """
    function Axis3D(; name::String="axis",
                    visible::Bool=false)::Axis3D

        default_properties = SceneProperties{Axis3D}(name)

        axis = new(visible, default_properties)

        return axis
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
`Scene3DOutput` stores the Matplotlib-ready data for plotting. It stores the
final result of all the heavy computations of projecting the scene onto a
camera. With the `Scene3DOutput` object in hand, you can readily make a plot
using Matplotlib. However, this is "raw output" data in the sense that it is
devoid of all the object tree hierarchy, etc., information that was used to
store and manipulate the scene.
"""
mutable struct Scene3DOutput
    tris::RealTensor # Face triangle list
end # struct

# ..:: Methods ::..

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
        try
            owner(obj)
            fmt = "Object %s is already associated with a node, "*
                "cannot add it again"
            msg = @eval @printf($fmt, name($obj))
            throw(ArgumentError(msg))
        catch
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
which is either `:local` or `:body`. The `mode` parameter specifies whether to
perform an intrinsic transformation (transform around `frame`) or an extrinsic
transformation (transform around the parent frame for `frame==:local` or the
local frame for `frame==:body`).

# Arguments
- `obj`: the object to transform the pose of.

# Keywords
- `frame`: (optional) the frame whose pose is to be changed. Can be either the
  `:local` or `:body` frames which belong to `obj`.
- `mode`: (optional) the transformation mode. Setting `:extrinsic` uses the
  axes of the parent frame to do the transformation (i.e., parent node's
  `:local` frame if `frame==:local`, and the object's `:local` frame if
  `frame==:body`). Setting `:extrinsic` uses the axes of `frame` *as it is
  being transformed* in order to perform the sequence of transformations.
- `kwargs`: the post transformation sequence. The possible commands are:
    - `x`: translate along `x` axis.
    - `y`: translate along `y` axis.
    - `z`: translate along `z` axis.
    - `roll`: rotate around `x` axis.
    - `pitch`: rotate around `y` axis.
    - `yaw`: rotate around `z` axis.
    - `degree`: a boolean flag whether to treat angles as degrees.
"""
function move!(obj::AbstractObject3D;
               frame::Symbol=:local,
               mode::Symbol=:intrinsic,
               kwargs...)::Nothing

    @assert frame in (:local, :body)
    @assert mode in (:extrinsic, :intrinsic)

    sequence = collect(kwargs)
    degrees = ((:degree in sequence && sequence[:degree]) ||
        !(:degree in sequence))

    # Perform the sequence of operations
    H_update = homtransf()
    for (op, val) in sequence
        # Compute the atomic transformation
        if op==:x
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
        if mode==:intrisinc
            H_update = H_update*H_atom
        else
            H_update = H_atom*H_update
        end
    end

    # Apply the transformation to the object's frame
    update_pose(obj, H_update, frame, mode)

    return nothing
end # function

"""
    relative_pose(objA, objB[, src, dest])

Get the relation pose between `src` frame of object `objA` and `dest` frame of
object `objB`. The returned homogeneous transformation matrix `H` transforms
homogeneous vectors from `objA`'s frame to `objB`'s frame. You can specify
`:local` and `:body` for `src` and `dest`. The default for both is `:body`.

# Arguments
- `objA`: the source object.
- `objB`: the destination object.
- `src`: (optional) the source object's frame, either `:local` or `:body`.
- `dest`: (optional) the destination object's frame, either `:local` or
  `:body`.

# Returns
- `bar`: description.
"""
function relative_pose(objA::AbstractObject3D,
                       objB::AbstractObject3D,
                       src::Symbol=:local,
                       dest::Symbol=:local)::RealMatrix

    @assert src in (:local, :body)
    @assert dest in (:local, :body)

    # Find a common ancestor node
    nodeA = owner(objA)
    nodeB = owner(objB)
    nodeC = find_common(nodeA, nodeB)

    # Find the relative pose between local A and local B frames
    H_CA = relative_local_pose_linear(nodeA, nodeC)
    H_CB = relative_local_pose_linear(nodeB, nodeC)
    H_BA = hominv(H_CB)*H_CA

    # Update for body frame choice
    if src==:body
        H_BA = H_BA*get_pose(objA, :body)
    end
    if dest==:body
        H_BA = hominv(get_pose(objB, :body))*H_BA
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
        Hi = get_pose(obj, :local)
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
"""
function load_wavefront(filepath::String)::Tuple{RealMatrix, IntMatrix}
    V, F = [], []
    f = open(filepath, "r")
    for line in readlines(f)
        if startswith(line, "#")
            continue
        end
        values = split(line) #noinfo
        if isempty(values)
            println("here")
        elseif values[1]=="v"
            push!(V, [parse(Float64, x) for x in values[2:4]])
        elseif values[1]=="f"
            push!(F, [parse(Int, x) for x in values[2:4]])
        end
    end
    close(f)
    V = hcat(V...)
    F = hcat(F...)
    return V, F
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

# Throws
- `ArgumentError` if not exactly one camera is found in the scene, when the
  camera to use is specified as a string.
"""
function render(scene::Scene3D,
                camera::Optional{Union{Camera3D, String}}=nothing,
                path::String="./figures/scene3d_render.pdf";
                size::Tuple{Real, Real}=(5, 5),
                xlim::Tuple{Real, Real}=(-1, 1),
                ylim::Tuple{Real, Real}=(-1, 1),
                aspect::Real=1)::Nothing

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
    fig = create_figure(size)

    # Add an axis
    ax = fig.add_axes([0, 0, 1, 1],
                      xlim=xlim,
                      ylim=ylim,
                      aspect=aspect,
                      frameon=false)

    # Sanitize axes
    ax.set_xticks([])
    ax.set_yticks([])

    # Generate the mesh triangles
    scene_data = project_scene_to_camera(scene, camera)

    scene_baked = PolyCollection(
        scene_data.tris,
        closed=true,
        linewidth=0.1,
        facecolor=Red,
        edgecolor="black")

    ax.add_collection(scene_baked)

    # Save figure to file
    save_figure(path, tight_layout=false)

    return nothing
end # function

"""
    project_scene_to_camera(scene, camera)

Project `Scene3D` onto a `Camera3D`, generating data which is readily plotted
using Matplotlib. This function performs z-depth sorting so that objects
overlap appropriately according to their 3D position.

# Arguments
- `scene`: the 3D scene definition object.
- `camera`: the camera object to use for rendering the scene.

# Returns
- `baked`: the projected scene data that can be used for plotting.
"""
function project_scene_to_camera(scene::Scene3D,
                                 camera::Camera3D)::Scene3DOutput

    # >> Collect data on all objects into lists <<

    tris_3d = RealTensor[]

    persp = perspective(camera) # Perspect projection to camera

    traverse(scene.objects) do obj, _
        # Do not render cameras and invisible axes
        if (obj isa Camera3D ||
            (obj isa Axis3D && !obj.visible))
            return
        end

        # Get the raw mesh in its body frame
        if obj isa Axis3D
            # TODO
            return # remove
        else
            verts, faces = obj.V, obj.F
        end

        # Relative pose
        rel_pose = relative_pose(obj, camera, :body, :body)

        # Perspective projection
        proj = persp*rel_pose
        verts = proj*[verts; ones(1, size(verts, 2))]
        verts ./= reshape(verts[end, :], 1, :)

        # Extract the face triangles
        num_faces = size(faces, 2)
        tris = RealTensor(undef, num_faces, 3, 3)
        for i=1:num_faces
            face_verts = verts[1:3, faces[:, i]]
            tris[i, :, :] = face_verts'
        end

        # Save to concatenated list
        push!(tris_3d, tris)
    end

    tris_3d = cat(tris_3d..., dims=1)

    # >> Do z-depth sorting <<

    tris = mapslices(tris_3d, dims=[2, 3]) do verts
        verts[:, 1:2]
    end
    faces_z = mapslices(tris_3d, dims=[2, 3]) do verts
        verts_z = verts[:, 3]
        mean(verts_z)
    end
    faces_z = -squeeze(faces_z)
    zid = sortperm(faces_z)
    tris = tris[zid, :, :]

    out = Scene3DOutput(tris)

    return out
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
    match_mesh = (obj) -> typeof(obj)==Mesh3D

    all_axes = findall(match_axis, scene.objects)
    all_cameras = findall(match_camera, scene.objects)
    all_meshes = findall(match_mesh, scene.objects)

    @preprintf(io, indent, "%d cameras\n", length(all_cameras))
    @preprintf(io, indent, "%d axes\n", length(all_axes))
    @preprintf(io, indent, "%d meshes\n", length(all_meshes))

    # >> Print the object tree hierarchy <<
    @preprintf(io, indent, "\nObject tree:\n")
    traverse(scene.objects) do obj, depth
        local_indent = indent*(" "^(2*(depth+1)))
        @preprintf(io, local_indent, "%s\n", name(obj))
    end

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
