// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_CONFIG_CONFIG_HPP_
#define GKO_PUBLIC_CORE_CONFIG_CONFIG_HPP_


#include <map>
#include <string>
#include <unordered_map>

#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/config/type_descriptor.hpp>
#include <ginkgo/core/solver/solver_base.hpp>


namespace gko {
namespace config {


class registry;


class pnode;


/**
 * parse is the main entry point to create an Ginkgo LinOpFactory based on
 * some file configuration. It reads a configuration stored as a property tree
 * and creates the desired type.
 *
 * General rules for configuration
 * 1. The configuration can be used to define factory parameters and class
 *    template parameters. Any factory parameter that is not defined in the
 *    configuration will fallback to their default value. Any template parameter
 *    that is not defined will fallback to the type_descriptor argument
 * 2. The new `"type"` key determines which Ginkgo object to create. The value
 *    for this key is the desired class name with namespaces (except for
 *    `gko::`, `experimental::`, `stop::`). Any template parameters a class
 *    might have are left out. Only classes with a factory are supported. For
 *    example, the configuration `"type": "solver::Cg"` specifies that a Cg
 *    solver will be created. Note: template parameters can either be given in
 *    the configuration as separate key-value pairs, or in the type_descriptor.
 * 3. Factory and class template parameters can be defined with key-value pairs
 *    that are derived from the class they are referring to. When a factory has
 *    a parameter with the function `with_<key>(value)`, then the configuration
 *    allows `"<key>": value` to define this parameter. When a class has a
 *    template parameter `template<typename/specific_type... key> class`, then
 *    the configuration allows `"<key>": value`  to the template parameter. The
 *    supported values of the template parameter depend on the context. For
 *    index and value types, these are listed under 4. Currently, we do not
 *    support gko::array parameter.
 * 4. Values for template parameters are represented with strings. The following
 *    datatypes, with postfix to indicate their size, are supported: int32,
 *    int64, float32, float64, complex<float32>, complex<float64>.
 * 5. All keys use snake_case including template parameters. Factory parameter
 *    keys are already defined with snake_case in their factories, while class
 *    template arguments need to be translated, i.e. `ValueType -> value_type`.
 * 6. The allowed values for factory parameters depend on the type the parameter
 *    is stored as. There are three distinct options:
 *    - POD types (bool, integer, floating point, or enum): Except for enum,
 *      the value has to be the POD type. For enums, a string value is used to
 *      represent them. The string has to be one of the possible enum values.
 *      An example of this type of parameter is the `krylov_dim` parameter for
 *      the Gmres solver.
 *    - LinOp (smart) pointers: The value has to be a string. The string is used
 *      to look up a LinOp object in the registry.
 *      An example is the `generated_preconditioner` parameter for iterative
 *      solvers such as Cg.
 *    - LinOpFactory (smart) pointers: The value can either be a string, or a
 *      nested configuration. The string has the same behavior as for LinOp
 *      pointers, i.e. an LinOpFactory object from the registry is taken. The
 *      nested configuration has to adhere to the general configuration rules
 *      again. See the examples below for some use-cases.
 *      An example is the `preconditioner` parameter for iterative solvers
 *      such as Cg.
 *    - CriterionFactory (smart) pointers: The value can either be a string, or
 *      a nested configuration. It has the same behavior as for LinOpFactory.
 *    - A vector of the types above: The value has to be an array with the
 *      inner values specified as above.
 * 7. Complex values are represented as an 2-element array [real, imag]. If the
 *    array contains only one value, it will be considered as a complex number
 *    with an imaginary part = 0. An empty array will be treated as zero.
 * 8. Keys that expect array of objects also accept single object which is
 *    interpreted as a 1-element array. This means the following configurations
 *    are equivalent if the key expects an array value: `"<key>": [{object}]`
 *    and `"<key>": {object}`.
 * 9. The stopping criteria for a solver can alternatively be defined through a
 *    simple key-value map, where each key corresponds to a single criterion.
 *    The available keys are:
 *    - "iteration": <integer>, corresponds to gko::stop::Iteration
 *    - "relative_residual_norm": <floating point>, corresponds to
 *      gko::stop::ResidualNorm build with gko::stop::mode::rhs_norm
 *    - "initial_residual_norm": <floating point>, corresponds to
 *      gko::stop::ResidualNorm build with gko::stop::mode::initial_resnorm
 *    - "absolute_residual_norm": <floating point>, corresponds to
 *      gko::stop::ResidualNorm build with gko::stop::mode::absolute
 *    - "relative_implicit_residual_norm": <floating point>, corresponds to
 *      gko::stop::ImplicitResidualNorm build with gko::stop::mode::rhs_norm
 *    - "initial_implicit_residual_norm": <floating point>, corresponds to
 *      gko::stop::ImplicitResidualNorm build with
 *      gko::stop::mode::initial_resnorm
 *    - "absolute_implicit_residual_norm": <floating point>, corresponds to
 *      gko::stop::ImplicitResidualNorm build with gko::stop::mode::absolute
 *    - "time": <integer>, corresponds to gko::stop::Time
 *    The simplified definition also allows for setting the `ValueType` template
 *    parameter as discussed in 4. and 5.
 *
 * All configurations (except the simplified stopping criteria) need to specify
 * the resulting type by the field:
 * ```
 * "type": "some_supported_ginkgo_type"
 * ```
 * The result will be a deferred_factory_parameter, which is an intermediate
 * step before a LinOpFactory. Providing an Executor through the function
 * `.on(exec)` will then create the factory with the parameters as defined in
 * the configuration.
 *
 * Given a configuration that is defined as
 * ```
 * "type": "solver::Gmres",
 * "krylov_dim": 20,
 * "criteria": [
 *   {"type": "Iteration", "max_iters": 10},
 *   {"type": "ResidualNorm", "reduction_factor": 1e-6}
 * ]
 * ```
 * then passing it to this function like this:
 * ```c++
 * auto gmres_factory = parse(config, context);
 * ```
 * will create a factory for a GMRES solver, with the parameters `krylov_dim`
 * set to 20, and a combined stopping criteria, consisting of an Iteration
 * criteria with maximal 10 iterations, and a ResidualNorm criteria with a
 * reduction factor of 1e-6.
 *
 * The criteria parameter can alternatively be defined as
 * ```
 * "criteria": {
 *   "iteration": 10,
 *   "relative_residual_norm": 1e-6
 * }
 * ```
 *
 * By default, the factory will use the value type double, and index type
 * int32 when creating templated types. This can be changed by passing in a
 * type_descriptor. For example:
 * ```c++
 * auto gmres_factory =
 *     parse(config, context,
 *           make_type_descriptor<float32, int32>());
 * ```
 * will lead to a GMRES solver that uses `float` as its value type.
 * Additionally, the config can be used to set these types through the fields:
 * ```
 * value_type: "some_value_type"
 * ```
 * These types take precedence over the type descriptor and they are used for
 * every created object beginning from the config level they are defined on and
 * every deeper nested level, until a new type is defined. So, considering the
 * following example
 * ```
 * type: "solver::Ir",
 * value_type: "float32"
 * solver: {
 *   type: "solver::Gmres",
 *   preconditioner: {
 *     type: "preconditioner::Jacobi"
 *     value_type: "float64"
 *   }
 * }
 * ```
 * both the Ir and Gmres are using `float32` as a value type, and the
 * Jacobi uses `float64`.
 *
 * @param config  The property tree which must include `type` for the class
 *                base.
 * @param context  The registry which stores the building function map and the
 *                 storage for generated objects.
 * @param td  The default value and index type. If any object that
 *            is created as part of this configuration has a templated type,
 *            then the value and/or index type from the descriptor will be used.
 *            Any definition of the value and/or index type within the config
 *            will take precedence over the descriptor. If `void` is used for
 *            one or both of the types, then the corresponding type has to be
 *            defined in the config, otherwise the parsing will fail.
 *
 * @return a deferred_factory_parameter which creates an LinOpFactory after
 *         `.on(exec)` is called on it.
 */
deferred_factory_parameter<gko::LinOpFactory> parse(
    const pnode& config, const registry& context,
    const type_descriptor& td = make_type_descriptor<>());


}  // namespace config
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_CONFIG_CONFIG_HPP_
