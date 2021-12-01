#= Generic tree implementation.

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

using Printf

export TreeCompatibilityTrait, IsTreeCompatible
export owner

export TreeNode
export is_root, is_leaf, set_parent!, traverse, find_common,
    add_child!, remove_child!

# ..:: Globals ::..

abstract type AbstractTreeNode end

# ..:: Traits ::..

"""
`TreeCompatibilityTrait` describes whether a type is compatible for use with
the tree data structure. This means that we expect the type of provide certain
methods and data that make tree operations work.
"""
abstract type TreeCompatibilityTrait end

struct IsTreeCompatible <: TreeCompatibilityTrait end # struct
struct NotTreeCompatible <: TreeCompatibilityTrait end # struct

""" By default, types are incompatible with being used in the tree. """
TreeCompatibilityTrait(x) = TreeCompatibilityTrait(typeof(x))
TreeCompatibilityTrait(::Type) = NotTreeCompatible()

""" Check if an object is compatible with the tree. """
tree_compatible(x) = tree_compatible(TreeCompatibilityTrait(x), x)
tree_compatible(::IsTreeCompatible, x) = true
tree_compatible(::NotTreeCompatible, x) = false

"""
    owner(x)

Get the tree node which owns the data object `x`. This is the basic way by
which we can access the tree data structure when we just have some piece of
data that is stored in the tree.

# Arguments
- `x`: a data object that `IsTreeCompatible`.

# Returns
The owning tree node.
"""
owner(x) = owner(TreeCompatibilityTrait(x), x)
owner(::IsTreeCompatible, x::T) where {T} = begin
    msg = @sprintf("Please implement the owner function for %s", T)
    throw(ErrorException(msg))
end
owner(::NotTreeCompatible, x::T) where {T} = begin
    msg = @sprintf("%s is incompatible with the tree", T)
    throw(ErrorException(msg))
end

# ..:: Data structures ::..

mutable struct TreeNode{T} <: AbstractTreeNode
    parent::Optional{TreeNode} # Parent node
    children::Vector{TreeNode} # Child nodes
    data::T                    # Data container

    """
        TreeNode(data[, parent])

    Basic constructor. Creates a node with some data, and optionally sets its
    parent. If `parent` is provided, the constructor appends the newly created
    node to the list `parent.children`.

    # Arguments
    - `data`: an `IsTreeCompatible` object holding any data that you want
      associated with the node.
    - `parent`: (optional) the node parent.

    # Returns
    - `node`: the newly created node.
    """
    function TreeNode(data, parent::Optional{TreeNode}=nothing)::TreeNode

        if !tree_compatible(data)
            T = typeof(data)
            msg = @sprintf("%s is not compatible with the tree", T)
            throw(ArgumentError(msg))
        end

        T = typeof(data)
        node = new{T}(parent, [], data)

        if !isnothing(parent)
            add_child!(parent, node)
        end

        return node
    end
end # struct

# ..:: Methods ::..

""" Check if the node is tree root (no parent). """
is_root(node::TreeNode) = isnothing(node.parent)

""" Check if the node is tree leaf (no child). """
is_leaf(node::TreeNode) = isempty(node.children)

"""
    add_child!(parent, node...)

Add child(children) node(s) to the parent node.

# Arguments
- `parent`: the parent node.
- `node...`: the child(children) node(s).
"""
function add_child!(parent::TreeNode, node::TreeNode...)::Nothing
    push!(parent.children, node...)
    return nothing
end

"""
    remove_child!(parent, child)

Unlink child node from parent.

# Arguments
- `parent`: the parent node.
- `child`: the child node.
"""
function remove_child!(parent::TreeNode, child::TreeNode)::Nothing
    filter!(c->c!=child, parent.children)
    return nothing
end

"""
    set_parent!(node, parent)

Set a parent for the node.

# Arguments
- `node`: the node whose parent is to be set.
- `parent`: the new parent.
"""
function set_parent!(node::TreeNode, parent::TreeNode)::Nothing
    node.parent = parent
    return nothing
end

"""
    traverse(action, node[, depth][; ignore])

Traverse the tree starting from `node` and perform `action` on every node.

# Arguments
- `action`: the action to perform on each node. The function signature must be
  `action(data, depth)` where `data` is the current node's data and `depth` is
  the node depth (`depth=0` for `node`).
- `node`: the node to start traversing from.
- `depth`: (optional) the starting depth value. If not provided, start from 0.

# Keywords
- `ignore`: (optional) a list of nodes (and all their children) to ignore in
  the search.
"""
function traverse(action::Function,
                  node::TreeNode,
                  depth::Int=0;
                  ignore::Vector{TreeNode}=TreeNode[])::Nothing

    if node in ignore
        return nothing
    end

    action(node.data, depth)

    for child in node.children
        traverse(action, child, depth+1; ignore=ignore)
    end

    return nothing
end

"""
    findall(matcher, node[; ignore])

Find all matches in the tree starting from `node`.

# Arguments
- `matcher`: the matching expression. The function signature must be
  `matcher(data)` where `data` is the node's data, and it should return true to
  signify a match, and false otherwise.
- `node`: the root node to start matching from.

# Keywords
- `ignore`: (optional) a list of nodes (and all their children) to ignore in
  the search.

# Returns
- `match_list`: a list of matching node data.
"""
function Base.findall(matcher::Function,
                      node::TreeNode;
                      ignore::Vector{TreeNode}=TreeNode[])::Vector
    match_list = []
    traverse(node, ignore=ignore) do data, _
        matcher(data) ? push!(match_list, data) : nothing
    end
    return match_list
end

"""
    find_common(A, B[; ignore])

Find a node which `A` and `B` have in common. This has three possibile
outcomes:
- The common node is `A`, in which case `A` is an ancestor (i.e., parent, or
  parent of parent, etc.) of `B`
- The common node is `B`, in which case `A` is a descendant of `B`
- The common node is some internal node `C` that is not the tree root
- The common node is the tree root

# Arguments
- `A`: a node in the tree.
- `B`: another node in the tree.

# Arguments
- `ignore`: (optional) a list of nodes (and all their children) to ignore
  during `findall` search.

# Returns
- `C`: the common ancestor node of `A` and `B`.
"""
function find_common(A::TreeNode, B::TreeNode;
                     ignore::Vector{TreeNode}=TreeNode[])::TreeNode

    matches = findall(A, ignore=ignore) do x
        owner(x)==B
    end

    if !isempty(matches)
        return A
    else
        return find_common(A.parent, B, ignore=TreeNode[A])
    end
end
