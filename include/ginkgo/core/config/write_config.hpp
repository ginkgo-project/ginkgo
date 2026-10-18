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
 * It supports the following types of parse's registry (named by the `"type"`
 * that selects them), with every value and index type that parse can create
 * them with:
 * - the solvers solver::Bicg, solver::Bicgstab, solver::Cg, solver::Cgs,
 *   solver::Fcg, solver::Gmres, solver::Ir, solver::Minres, solver::PipeCg,
 *   solver::LowerTrs, solver::UpperTrs and solver::Multigrid;
 * - the factorizations factorization::Ic and factorization::Ilu;
 * - the preconditioners preconditioner::Ic, preconditioner::Ilu (with either
 *   `reverse_apply`), preconditioner::Isai (with every `isai_type`) and
 *   preconditioner::Jacobi;
 * - the coarsening methods multigrid::Pgm, multigrid::Pmis and
 *   multigrid::UniformCoarsening.
 *
 * Every other type of the registry has no writer and throws NotSupported:
 * solver::CbGmres, solver::Chebyshev, solver::Direct, solver::Gcr,
 * solver::Idr, factorization::Cholesky, factorization::Lu,
 * factorization::ParIc, factorization::ParIct, factorization::ParIlu,
 * factorization::ParIlut, preconditioner::GaussSeidel,
 * preconditioner::NullspaceProjection, preconditioner::Sor, multigrid::Rs and,
 * with MPI, preconditioner::Schwarz. So does a factory type that is not in the
 * registry.
 *
 * The result is a map with the following entries:
 * - `"type"`: the key that selects the class in the registry, e.g.
 *   `"solver::Cg"` or `"preconditioner::Jacobi"`.
 * - `"value_type"`: the value type of the class, e.g. `"float64"`, if the class
 *   has one. solver::Multigrid has none; its nested factories write their own.
 * - the template parameters of the class other than its value and index
 *   types, under the keys that parse selects them with, e.g.
 *   `"reverse_apply"` of preconditioner::Ilu and `"isai_type"` of
 *   preconditioner::Isai.
 * - every factory parameter that parse reads, in the same spelling and encoding
 *   that parse expects, including the parameters left at their default
 *   values:
 *   - booleans, integers and floating point values as such, enums as the
 *     strings parse accepts for them, complex values as a `[real, imag]` array;
 *   - nested factories as nested maps. A factory parameter that is not set is
 *     left out, so parse leaves it at its default. A matrix::IdentityFactory
 *     given as a solver's `preconditioner` or as solver::Ir's `solver` is left
 *     out as well, since the default of both is the identity;
 *   - lists of nested factories (e.g. `mg_level`, the smoothers and
 *     `coarsest_solver` of solver::Multigrid) as arrays of nested maps, in the
 *     order of the list. An empty list is left out, so parse leaves it at its
 *     default (e.g. Multigrid's default smoother or coarsest solver);
 *   - the stopping criteria (`"criteria"`) always as an array of criterion
 *     maps. A combined criterion (see gko::stop::combine) is written as the
 *     criteria it combines;
 *   - LinOp parameters (e.g. `generated_preconditioner`) as the string
 *     `"<object>"`, a placeholder for the object, which a configuration
 *     cannot contain. They are left out when they are not set.
 *
 * The result contains no empty node, so it can be stored as JSON or YAML and
 * read back with the readers in extensions/config.
 *
 * The index type is not written, because parse does not accept it in the
 * configuration: it takes the index type of the factory and of all its nested
 * factories from its type_descriptor argument. Parsing the result with a
 * type_descriptor that has the index type of the written factory then creates
 * a factory with the same parameters, if every nested factory with an index
 * type uses that same index type. A factory whose nested factories mix index
 * types is written as it is, without an error, and is parsed back with the
 * type_descriptor's index type for all of them.
 *
 * Function-valued parameters (the `level_selector` and `solver_selector` of
 * solver::Multigrid) are not written, since parse does not read them either.
 * A factory that sets them is written as if they were not set, so its
 * configuration creates a factory that uses the default selectors.
 *
 * @param factory  the factory to write
 *
 * @return the configuration of the factory
 *
 * @throw NotSupported  if the type of the factory, or of one of its nested
 *                      factories, has no writer (e.g. an IdentityFactory
 *                      other than a solver's `preconditioner` or Ir's
 *                      `solver`), or if it has a parameter value that parse
 *                      can not read, e.g. Jacobi's block_pointers or a null
 *                      entry in a list of factories.
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
