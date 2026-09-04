// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_SOLVER_BICG_HPP_
#define GKO_PUBLIC_CORE_SOLVER_BICG_HPP_


#include <vector>

#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/config/config.hpp>
#include <ginkgo/core/config/registry.hpp>
#include <ginkgo/core/log/logger.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/identity.hpp>
#include <ginkgo/core/matrix/multivector.hpp>
#include <ginkgo/core/solver/solver_base.hpp>
#include <ginkgo/core/stop/combined.hpp>
#include <ginkgo/core/stop/criterion.hpp>


namespace gko {
namespace solver {


/**
 * BICG or the Biconjugate gradient method is a Krylov subspace solver.
 *
 * Being a generic solver, it is capable of solving general matrices, including
 * non-s.p.d matrices. Though, the memory and the computational requirement of
 * the BiCG solver are higher than of its s.p.d solver counterpart, it has
 * the capability to solve generic systems.
 *
 * BiCG is based on the bi-Lanczos tridiagonalization method and in exact
 * arithmetic should terminate in at most \f$ N \f$ iterations (\f$ 2N \f$
 * matrix-vector products — one per iteration with \f$ A \f$ and
 * \f$ A^H \f$ each).
 * It couples two Krylov sequences and maintains residuals \f$ r_k \f$,
 * shadow residuals \f$ \tilde r_k \f$, and search directions
 * \f$ p_k, \tilde p_k \f$ that satisfy the biorthogonality conditions
 * \f$ \tilde r_i^H r_j = 0 \f$ for \f$ i \ne j \f$. Each iteration
 * performs the coupled update
 * \f[
 *   \alpha_k = \frac{\tilde r_k^H r_k}{\tilde p_k^H A p_k}, \qquad
 *   r_{k+1}       = r_k       - \alpha_k A p_k, \qquad
 *   \tilde r_{k+1} = \tilde r_k - \alpha_k A^H \tilde p_k.
 * \f]
 * It forms the basis of cheaper variants such as BiCGSTAB and CGS, which
 * avoid the explicit \f$ A^H \f$ apply.
 *
 * @par References
 * - Fletcher, R. *Conjugate gradient methods for indefinite systems.*
 *   Numerical Analysis (Dundee 1975), Lecture Notes in Mathematics 506,
 *   Springer, 1976. <https://doi.org/10.1007/BFb0080116>
 *
 * @tparam ValueType  precision of matrix elements
 *
 * @ingroup solvers
 * @ingroup LinOp
 */
template <typename ValueType = default_precision>
class Bicg
    : public LinOp,
      public EnablePreconditionedIterativeSolver<ValueType, Bicg<ValueType>>,
      public Transposable {
    GKO_ASSERT_SUPPORTED_VALUE_TYPE;

public:
    using value_type = ValueType;
    using transposed_type = Bicg<ValueType>;

    std::unique_ptr<LinOp> transpose() const override;

    std::unique_ptr<LinOp> conj_transpose() const override;

    /**
     * Return true as iterative solvers use the data in x as an initial guess.
     *
     * @return true as iterative solvers use the data in x as an initial guess.
     */
    bool apply_uses_initial_guess() const override { return true; }

    class Factory;

    struct parameters_type
        : enable_preconditioned_iterative_solver_factory_parameters<
              parameters_type, Factory> {};

    GKO_ENABLE_LIN_OP_FACTORY(Bicg, parameters, Factory);
    GKO_ENABLE_BUILD_METHOD(Factory);

    /**
     * Create the parameters from the property_tree.
     * Because this is directly tied to the specific type, the value/index type
     * settings within config are ignored and type_descriptor is only used
     * for children configs.
     *
     * @param config  the property tree for setting
     * @param context  the registry
     * @param td_for_child  the type descriptor for children configs. The
     *                      default uses the value type of this class.
     *
     * @return parameters
     */
    static parameters_type parse(const config::pnode& config,
                                 const config::registry& context,
                                 const config::type_descriptor& td_for_child =
                                     config::make_type_descriptor<ValueType>());

protected:
    void apply_impl(const LinOp* b, LinOp* x) const override;

    void apply_dense_impl(const matrix::MultiVector<ValueType>* b,
                          matrix::MultiVector<ValueType>* x) const;

    void apply_impl(const LinOp* alpha, const LinOp* b, const LinOp* beta,
                    LinOp* x) const override;

    explicit Bicg(std::shared_ptr<const Executor> exec) : LinOp(std::move(exec))
    {}

    explicit Bicg(const Factory* factory,
                  std::shared_ptr<const LinOp> system_matrix)
        : LinOp(factory->get_executor(),
                gko::transpose(system_matrix->get_size())),
          EnablePreconditionedIterativeSolver<ValueType, Bicg<ValueType>>{
              std::move(system_matrix), factory->get_parameters()},
          parameters_{factory->get_parameters()}
    {}
};


template <typename ValueType>
struct workspace_traits<Bicg<ValueType>> {
    using Solver = Bicg<ValueType>;
    // number of vectors used by this workspace
    static int num_vectors(const Solver&);
    // number of arrays used by this workspace
    static int num_arrays(const Solver&);
    // array containing the num_vectors names for the workspace vectors
    static std::vector<std::string> op_names(const Solver&);
    // array containing the num_arrays names for the workspace vectors
    static std::vector<std::string> array_names(const Solver&);
    // array containing all varying scalar vectors (independent of problem size)
    static std::vector<int> scalars(const Solver&);
    // array containing all varying vectors (dependent on problem size)
    static std::vector<int> vectors(const Solver&);

    // residual vector
    constexpr static int r = 0;
    // preconditioned residual vector
    constexpr static int z = 1;
    // p vector
    constexpr static int p = 2;
    // q vector
    constexpr static int q = 3;
    // "transposed" residual vector
    constexpr static int r2 = 4;
    // "transposed" preconditioned residual vector
    constexpr static int z2 = 5;
    // "transposed" p vector
    constexpr static int p2 = 6;
    // "transposed" q vector
    constexpr static int q2 = 7;
    // beta scalar
    constexpr static int beta = 8;
    // previous rho scalar
    constexpr static int prev_rho = 9;
    // current rho scalar
    constexpr static int rho = 10;
    // constant 1.0 scalar
    constexpr static int one = 11;
    // constant -1.0 scalar
    constexpr static int minus_one = 12;

    // stopping status array
    constexpr static int stop = 0;
    // reduction tmp array
    constexpr static int tmp = 1;
};


}  // namespace solver
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_SOLVER_BICG_HPP_
