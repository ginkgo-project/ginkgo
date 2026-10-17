// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_CONFIG_WRITE_CONFIG_HPP_
#define GKO_PUBLIC_CORE_CONFIG_WRITE_CONFIG_HPP_


#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/config/property_tree.hpp>
#include <ginkgo/core/stop/criterion.hpp>


namespace gko {
namespace config {


/**
 * write_config is the inverse of parse: it writes the parameters of a
 * LinOpFactory as the property tree that parse reads.
 *
 * The result is a map with the following entries:
 * - `"type"`: the key that selects the class in the registry, e.g.
 *   `"solver::Cg"` or `"preconditioner::Jacobi"`.
 * - `"value_type"`: the value type of the class, e.g. `"float64"`, if the class
 *   has one.
 * - every factory parameter that parse reads, in the same spelling and encoding
 *   that parse expects, including the parameters left at their default
 *   values:
 *   - booleans, integers and floating point values as such, enums as the
 *     strings parse accepts for them, complex values as a `[real, imag]` array;
 *   - nested factories as nested maps; a factory parameter that is not set is
 *     written as an empty node;
 *   - the stopping criteria (`"criteria"`) always as an array of criterion
 *     maps. A combined criterion (see gko::stop::combine) is written as the
 *     criteria it combines;
 *   - LinOp parameters (e.g. `generated_preconditioner`) as the string
 *     `"<object>"`, a placeholder for the object, which a configuration
 *     cannot contain. They are left out when they are not set.
 *
 * The index type is not written, because parse does not accept it in the
 * configuration: it takes the index type from its type_descriptor argument.
 * Parsing the result with a type_descriptor that has the index type of the
 * written factory then creates a factory with the same parameters.
 *
 * Function-valued parameters (e.g. selectors) are not written, since parse
 * does not read them either.
 *
 * @param factory  the factory to write
 *
 * @return the configuration of the factory
 *
 * @throw NotSupported  if the type of the factory, or of one of its nested
 *                      factories, has no writer, or if it has a parameter value
 *                      that parse can not read, e.g. Jacobi's block_pointers.
 */
pnode write_config(const LinOpFactory& factory);


/**
 * write_config writes the parameters of a stopping criterion factory as the
 * property tree that parse reads for an entry of `"criteria"`.
 *
 * The result is a map with `"type"` (e.g. `"Iteration"` or `"ResidualNorm"`),
 * `"value_type"` if the criterion has one, and every parameter that parse
 * reads for the criterion. A combined criterion (see gko::stop::combine) is
 * written as an array of the criteria it combines, nested combinations
 * flattened, which parse accepts as the value of `"criteria"`.
 *
 * @param factory  the criterion factory to write
 *
 * @return the configuration of the criterion factory
 *
 * @throw NotSupported  if the type of the factory has no writer
 */
pnode write_config(const stop::CriterionFactory& factory);


}  // namespace config
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_CONFIG_WRITE_CONFIG_HPP_
