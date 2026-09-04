// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_SOLVER_CGS_HPP_
#define GKO_PUBLIC_CORE_SOLVER_CGS_HPP_


#include <vector>

#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/config/config.hpp>
#include <ginkgo/core/config/registry.hpp>
#include <ginkgo/core/log/logger.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/identity.hpp>
#include <ginkgo/core/solver/solver_base.hpp>
#include <ginkgo/core/stop/combined.hpp>
#include <ginkgo/core/stop/criterion.hpp>


namespace gko {
namespace solver {


/**
 * CGS or the conjugate gradient square method is an iterative type Krylov
 * subspace method which is suitable for general systems.
 *
 * CGS rests on the identity that BiCG produces a residual of the form
 * \f$ r_k^{\mathrm{BiCG}} = \psi_k(A) r_0 \f$, where \f$ \psi_k \f$ is the
 * residual polynomial. Squaring this polynomial yields the CGS residual
 * \f[
 *   r_k^{\mathrm{CGS}} = \psi_k(A)^2 r_0,
 * \f]
 * so the iteration avoids the explicit \f$ A^H \f$ apply that BiCG
 * requires, and it contracts the residual twice as fast per step when
 * BiCG converges.
 *
 * Starting from \f$ r_0 = b - A x_0 \f$, a shadow residual
 * \f$ \tilde r_0 \f$ (Ginkgo uses \f$ \tilde r_0 = r_0 \f$),
 * \f$ q_{-1} = p_{-1} = 0 \f$ and \f$ \rho_{-1} = 1 \f$, one iteration
 * with the preconditioner \f$ M \approx A^{-1} \f$ reads
 * \f[
 *   \begin{aligned}
 *     \rho_k     &= \langle \tilde r_0, r_k \rangle,
 *       &\qquad \beta_k   &= \rho_k / \rho_{k-1}, \\
 *     u_k        &= r_k + \beta_k q_{k-1},
 *       &\qquad p_k      &= u_k + \beta_k
 *                             (q_{k-1} + \beta_k p_{k-1}), \\
 *     \hat v_k   &= A M p_k,
 *       &\qquad \alpha_k  &= \rho_k /
 *                             \langle \tilde r_0, \hat v_k \rangle, \\
 *     q_k        &= u_k - \alpha_k \hat v_k,
 *       &\qquad \hat u_k  &= M (u_k + q_k), \\
 *     x_{k+1}    &= x_k + \alpha_k \hat u_k,
 *       &\qquad r_{k+1}   &= r_k - \alpha_k A \hat u_k.
 *   \end{aligned}
 * \f]
 * The pair \f$ u_k, q_k \f$ takes the place of the single BiCG search
 * direction, which is why the update of \f$ x \f$ uses their sum. The
 * trade-off for dropping \f$ A^H \f$ is that the squared polynomial also
 * squares the error: the residuals lose monotonicity and can oscillate
 * strongly on poorly conditioned systems. When this matters in practice,
 * \ref gko::solver::Bicgstab "BiCGSTAB" replaces the second application of
 * \f$ \psi_k \f$ by a locally minimizing polynomial for the same number of
 * operator applies and one extra inner product.
 *
 * The implementation in Ginkgo makes use of the merged kernel to make the best
 * use of data locality. The inner operations in one iteration of CGS are merged
 * into 3 separate steps.
 *
 * @tparam ValueType precision of matrix elements
 *
 * @ingroup solvers
 * @ingroup LinOp
 */
template <typename ValueType = default_precision>
class Cgs
    : public LinOp,
      public EnablePreconditionedIterativeSolver<ValueType, Cgs<ValueType>>,
      public Transposable {
    GKO_ASSERT_SUPPORTED_VALUE_TYPE;

public:
    using value_type = ValueType;
    using transposed_type = Cgs<ValueType>;

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

    GKO_ENABLE_LIN_OP_FACTORY(Cgs, parameters, Factory);
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

    void validate_data() const override;

protected:
    void apply_impl(const LinOp* b, LinOp* x) const override;

    template <typename VectorType>
    void apply_dense_impl(const VectorType* b, VectorType* x) const;

    void apply_impl(const LinOp* alpha, const LinOp* b, const LinOp* beta,
                    LinOp* x) const override;

    explicit Cgs(std::shared_ptr<const Executor> exec) : LinOp(std::move(exec))
    {}

    explicit Cgs(const Factory* factory,
                 std::shared_ptr<const LinOp> system_matrix)
        : LinOp(factory->get_executor(),
                gko::transpose(system_matrix->get_size())),
          EnablePreconditionedIterativeSolver<ValueType, Cgs<ValueType>>{
              std::move(system_matrix), factory->get_parameters()},
          parameters_{factory->get_parameters()}
    {}
};


template <typename ValueType>
struct workspace_traits<Cgs<ValueType>> {
    using Solver = Cgs<ValueType>;
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
    // r tilde vector
    constexpr static int r_tld = 1;
    // p vector
    constexpr static int p = 2;
    // q vector
    constexpr static int q = 3;
    // u vector
    constexpr static int u = 4;
    // u hat vector
    constexpr static int u_hat = 5;
    // v hat vector
    constexpr static int v_hat = 6;
    // t vector
    constexpr static int t = 7;
    // alpha scalar
    constexpr static int alpha = 8;
    // beta scalar
    constexpr static int beta = 9;
    // beta scalar
    constexpr static int gamma = 10;
    // previous rho scalar
    constexpr static int prev_rho = 11;
    // current rho scalar
    constexpr static int rho = 12;
    // constant 1.0 scalar
    constexpr static int one = 13;
    // constant -1.0 scalar
    constexpr static int minus_one = 14;

    // stopping status array
    constexpr static int stop = 0;
    // reduction tmp array
    constexpr static int tmp = 1;
};


}  // namespace solver
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_SOLVER_CGS_HPP_
