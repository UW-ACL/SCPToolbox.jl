#= Data structures for working with differentiable functions.

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

if isdefined(@__MODULE__, :LanguageServer)
    include("general.jl")
    include("cone.jl")
end

import JuMP: value

export jacobian, set_jacobian!
export @value, @jacobian

# ..:: Globals ::..

# Convenience typealiases
const FunctionAtomicValueType = Types.Variable
const InputArgumentType = Types.VariableAbstractArray
const FunctionValueType = Types.VariableAbstractArray
const JacobianValueType = Types.VariableAbstractArray
const JacobianKeys = Union{Int, Tuple{Int, Int}}
const JacobianDictType = Dict{JacobianKeys, JacobianValueType}
const FunctionValueOutputType = Union{FunctionAtomicValueType,
                                      FunctionValueType}

# ..:: Data structures ::..

"""
`TypedFunction{T, N}` is a wrapper class for a function which is supposed to
accept the input arguments `N...` (a splatted `Tuple`) and return the argument
`T` (which can again be a `Tuple`, or some single type).

The idea is that there will be a runtime error when the function gets
evaluated, and it will read something like `MethodError: Cannot convert an
object of type ... to an object of type ...`. Note that there will be no
compile-time error, since Julia does not yet support asserting based on the
function signature (see the [ongoing
discussion](https://github.com/JuliaLang/julia/issues/13984))!
"""
struct TypedFunction
    f::Function       # The core function
    input_type::Type  # The function input argument type
    output_type::Type # The function return value type

    """
        TypedFunction(f, in, out)

    Basic constructor, wraps the function `f`.

    # Arguments
    - `f`: the function to be wrapped.
    - `in`: the function input type (use `Tuple` for multiple inputs).
    - `out`: the function output type (use `Tuple` for multiple outputs).

    # Returns
    - `wrapper`: the wrapper.
    """
    function TypedFunction(f::Function, in::Type, out::Type)::TypedFunction
        in = (in<:Tuple) ? in : Tuple{in}
        wrapper = new(f, in, out)
        return wrapper
    end # function
end # struct

"""
`DifferentialFunctionOutput` is a convenience structure to systematize how the
function value and its Jacobians are evaluated. It stores the function value
and a dictionary representing the Jacobian values. Methods are provided for
accessing this data in a more user-friendly manner. """
struct DifferentiableFunctionOutput
    value::FunctionValueType # Function value
    J::JacobianDictType      # Jacobian values

    """
        DifferentiableFunctionOutput(f)

    Basic constructor. Just sets the function value, while the Jacobians can be
    added later via the `set_jacobian!` function.

    # Arguments
    - `f`: the function value.

    # Returns
    - `fout`: the function output structure.
    """
    function DifferentiableFunctionOutput(f::T)::DFOut where T

        # Convert to array
        if !(T<:AbstractArray)
            f = [f]
        end

        # Make sure the array is at least 1-dimensional
        if ndims(f)<1
            f = reshape(f, 1)
        end

        J = JacobianDictType()

        fout = new(f, J)

        return fout
    end # function
end # struct

const DFOut = DifferentiableFunctionOutput

"""
`DifferentiableFunction` wraps a core user-defined function call in a data
structure that can be queried for the resulting function value and its
Jacobians.
"""
mutable struct DifferentiableFunction
    f::TypedFunction    # The core function doing the computation
    xargs::Int          # Number of variable arguments
    pargs::Int          # Number of parameter arguments
    consts::Ref         # Other constants that the function depends on
    out::Ref{DFOut}     # The result from the most recent call to f
    evaluated::Bool     # Status lag if function has been evaluated

    """
        DifferentiableFunction(f, xargs, pargs, other)

    Basic constructor. The user specifies the core callable function that does
    the actual computation, and specified how many variable and parameter
    inputs it received, as well as a structure carrying any other values that
    are used in the core computation.

    The core function is expected to have the call format:

    ```julia
    f(x_1, ..., x_args, p_1, ..., p_pargs, other, jacobian)
    ```

    The first `xargs` arguments correspond to variables, the next `pargs`
    arguments correspond to parameters. We make the distinction between
    variables and parameters for duality-based sensitivity analysis. In the
    sensitivity analysis, our core interest is in how the solution varies due
    to changes in the *parameters*. The argument `other` specifies a structure
    of other constants that the function can use for its definition, while
    `jacobian` is a boolean value which, if `false`, signals that the function
    may omit Jacobian computation (which saves on compute time).

    # Arguments
    - `f`: the core function that does the computation of the function value
      and Jacobians.
    - `xargs`: the number of variable arguments.
    - `pargs`: the number of parameter arguments.
    - `other`: some structure that is used by `f` to define it (think of things
      like vehicle and environment parameters, etc.).

    # Returns
    - `DF`: the differentiable function structure.
    """
    function DifferentiableFunction(
        f::Function,
        xargs::Int,
        pargs::Int,
        other)::DifferentiableFunction

        # Make the core function typesafe
        in_type = Tuple{fill(InputArgumentType, xargs+pargs)...,
                        typeof(other), Bool}
        out_type = DifferentiableFunctionOutput
        F = TypedFunction(f, in_type, out_type)

        # Other parameters
        out = Ref{DFOut}()
        evaluated = false
        consts = Ref(other)

        DF = new(F, xargs, pargs, consts, out, evaluated)

        return DF
    end # function
end # struct

const DiffblF = DifferentiableFunction

# ..:: Methods ::..

"""
    throw_type_error(fw, which, args...)

Throw a type error because `args...` is not the expected type for the function.

# Arguments
- `fw`: the function wrapper object.
- `which`: either "input" or "output".
- `args...`: the argument list whose type is to be checked.

# Throws
- `SCP_BAD_ARGUMENT`: if there is a type mismatch.
"""
function throw_type_error(fw::TypedFunction, which::String, args)
    expected_type = (which=="input") ? fw.input_type : fw.output_type
    msg = @sprintf("bad %s argument type.", which)
    msg *= @sprintf(" Expected %s", expected_type)
    msg *= @sprintf(" but got %s", typeof(args))
    err = SCPError(0, SCP_BAD_ARGUMENT, msg)
    throw(err)
end

"""
    fw(arg...)

Evaluate the function, type safely.

# Arguments
- `args...`: the argument list to the core function being wrapped.

# Returns
- `out`: the output of the core function being wrapped.
"""
function (fw::TypedFunction)(args...)

    # Check the input type
    if !( typeof(args)<:fw.input_type )
        throw_type_error(fw, "input", args)
    end

    out = fw.f(args...) # Core wrapped function call

    # Check the return value type
    if !( typeof(out)<:fw.output_type )
        throw_type_error(fw, "output", out)
    end

    return out
end # function

"""
    value(out[; scalar])

Get the function value. If you know that it is a 1-dimensional array and want
to return a scalar, then pass in `scalar=true`.

# Arguments
- `out`: the function output object.

# Keywords
- `scalar`: (optional) whether to output a scalar value.

# Returns
- `value`: the function value.
"""
function value(out::DFOut; scalar::Bool=false)::FunctionValueOutputType
    value = out.value
    if scalar
        if length(value)>1
            msg = @sprintf("Cannot convert a value of size %s to a scalar",
                           size(value))
            err = SCPError(0, SCP_BAD_ARGUMENT, msg)
            throw(err)
        end
        value = value[1]
    end
    return value
end # function

"""
    jacobian(out, key[; permute])

Get the Jacobian value. Suppose that the function depends on N arguments,
i.e. ``f(x_1, \\dots, x_N)``. The key can be either an integer (e.g. `i`) or a
two-element tuple (e.g. `(i, j)`). In both cases, this function returns the
derivative of ``f`` with respect to the variables indicated by `key`. For
example, ``\\frac{\\partial f}{\\partial x_i}`` or ``\\frac{\\partial^2
f}{\\partial x_i\\partial x_j}``.

Because the derivatives can be exchanged places, the function will permute
`key` is necessary in order to retrive the corresponding Jacobian. If such a
permutation is successful, the transposed Jacobian is returned.

# Arguments
- `out`: the function output structure.
- `key`: which Jacobian to retrieve.

# Keywords
- `permute`: (optiona) whether to permute the key in order to try to get the
  Jacobian.

# Returns
- `J`: the Jacobian value. Most generally a tensor, which happens when ``f`` is
  a matrix-value function being differentiated with respect to a vector.
"""
function jacobian(out::DFOut, key::JacobianKeys;
                  permute::Bool=true)::JacobianValueType
    tuple_key = length(key)>1
    if !haskey(out.J, key)
        if permute && tuple_key
            # Try permuting the key
            key = reverse(key)
            return transpose(jacobian(out, key; permute=false))
        end
        plural = tuple_key ? "s" : ""
        msg = @sprintf("Jacobian with respect to argument%s %s not defined",
                       plural, key)
        err = SCPError(0, SCP_BAD_ARGUMENT, msg)
        throw(err)
    end
    J = out.J[key]
    return J
end # function

""" Get the dictionary of all jacobians """
all_jacobians(out::DFOut)::JacobianDictType = out.J

"""
    set_jacobian!(our, key, J)

Set the Jacobian value. For more information see the docstring of `jacobian`.

# Arguments
- `our`: the function output structure.
- `key`: which Jacobian to set.
- `J`: the Jacobian value.
"""
function set_jacobian!(out::DFOut, key::JacobianKeys,
                       J::JacobianValueType)::Nothing
    out.J[key] = J
    return nothing
end # function

"""
    F(args...[; jacobians, scalar])

Call the core method. The result is cached inside the `DifferentiableFunction`
structure. The call is done type-safely in the sense that the input and output
types are checked to match the required specification. An error is thrown if
this is not met.

# Arguments
- `args`: the arguments to the function. This is checked to be the same number
  of arguments as `xargs+pargs` specified when creating the
  `DifferentiableFunction` structure..

# Keywords
- `jacobians`: (optional) set to true in order to compute the Jacobians as
  well.
- `scalar`: (optional) whether to output a scalar value.

# Returns
- `f_value`: the function value from the call.
"""
function (F::DiffblF)(args::InputArgumentType...;
                      jacobians::Bool=false,
                      scalar::Bool=false)::FunctionValueOutputType
    nargin = length(args)
    narg_expected = F.xargs+F.pargs
    if nargin!=narg_expected
        msg = string("argument count mismatch for the function call ",
                     @sprintf("(expected %d but got %d arguments)",
                              narg_expected, nargin))
        err = SCPError(0, SCP_BAD_ARGUMENT, msg)
        throw(err)
    end
    # Make the call to the core function
    F.out[] = F.f(args..., F.consts[], jacobians)
    F.evaluated = true
    f_value = value(F.out[], scalar=scalar)
    return f_value
end # function

"""
Convenience methods that pass the calls down to `DifferentiableFunctionOutput`.
"""
value(F::DiffblF; scalar::Bool=false)::FunctionValueOutputType =
    value(F.out[], scalar=scalar)
jacobian(F::DiffblF, key::JacobianKeys)::JacobianValueType =
    jacobian(F.out[], key)
all_jacobians(F::DiffblF)::JacobianDictType = all_jacobians(F.out[])

# ..:: Macros ::..

"""
Simple wrapper of `DifferentiableFunctionOutput` constructor, see its
documentation.
"""
macro value(f)
    :( DifferentiableFunctionOutput($(esc(f))) )
end # macro

""" Simple wrapper of `set_jacobian!`, see its documentation. """
macro jacobian(out, key, J)
    :( set_jacobian!($(esc.([out, key, J])...)) )
end # macro
