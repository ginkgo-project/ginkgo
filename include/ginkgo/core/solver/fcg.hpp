// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_SOLVER_FCG_HPP_
#define GKO_PUBLIC_CORE_SOLVER_FCG_HPP_


#include <vector>

#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/config/config.hpp>
#include <ginkgo/core/config/registry.hpp>
#include <ginkgo/core/log/logger.hpp>
#include <ginkgo/core/matrix/identity.hpp>
#include <ginkgo/core/solver/solver_base.hpp>
#include <ginkgo/core/stop/combined.hpp>
#include <ginkgo/core/stop/criterion.hpp>


namespace gko {
namespace solver {


/**
 * FCG or the flexible conjugate gradient method is an iterative type Krylov
 * subspace method which is suitable for symmetric positive definite methods.
 *
 * Though this method performs very well for symmetric positive definite
 * matrices, it is in general not suitable for general matrices.
 *
 * In contrast to the standard CG, which uses the Fletcher-Reeves formula
 * \f[
 *   \beta_k = \frac{\langle r_k, z_k \rangle}
 *                  {\langle r_{k-1}, z_{k-1} \rangle},
 * \f]
 * the flexible CG uses the Polak-Ribière formula
 * \f[
 *   \beta_k = \frac{\langle r_k - r_{k-1}, z_k \rangle}
 *                  {\langle r_{k-1}, z_{k-1} \rangle}
 * \f]
 * for the next search direction. In CG the denominator
 * \f$ \langle r_{k-1}, z_{k-1} \rangle \f$ is exactly the numerator that
 * the previous iteration already computed, so a single dot product per
 * iteration is enough. FCG still needs \f$ \langle r_k, z_k \rangle \f$
 * as the denominator of the next iteration, and has to compute
 * \f$ \langle r_k - r_{k-1}, z_k \rangle \f$ on top of it — one extra dot
 * product, and therefore one extra global reduction, per iteration.
 *
 * In exchange, \f$ \beta_k \f$ no longer relies on the search directions
 * staying \f$ A \f$-conjugate, so the preconditioner \f$ M \f$ (and
 * therefore \f$ z = M r \f$) may change between iterations — useful when
 * the preconditioner is itself an inner iterative solve, a randomized
 * smoother, or otherwise not a fixed linear operator.
 *
 * The implementation in Ginkgo makes use of the merged kernel to make the best
 * use of data locality. The inner operations in one iteration of FCG are
 * merged into 2 separate steps.
 *
 * @tparam ValueType precision of matrix elements
 *
 * @ingroup solvers
 * @ingroup LinOp
 */
template <typename ValueType = default_precision>
class Fcg
    : public LinOp,
      public EnablePreconditionedIterativeSolver<ValueType, Fcg<ValueType>>,
      public Transposable {
    GKO_ASSERT_SUPPORTED_VALUE_TYPE;

public:
    using value_type = ValueType;
    using transposed_type = Fcg<ValueType>;

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

    GKO_ENABLE_LIN_OP_FACTORY(Fcg, parameters, Factory);
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
    void apply_impl(const AbstractMultiVector* b,
                    AbstractMultiVector* x) const override;

    void apply_impl(const AbstractMultiVector* alpha,
                    const AbstractMultiVector* b,
                    const AbstractMultiVector* beta,
                    AbstractMultiVector* x) const override;

    explicit Fcg(std::shared_ptr<const Executor> exec) : LinOp(std::move(exec))
    {}

    explicit Fcg(const Factory* factory,
                 std::shared_ptr<const LinOp> system_matrix)
        : LinOp(factory->get_executor(),
                gko::transpose(system_matrix->get_size())),
          EnablePreconditionedIterativeSolver<ValueType, Fcg<ValueType>>{
              std::move(system_matrix), factory->get_parameters()},
          parameters_{factory->get_parameters()}
    {}
};


template <typename ValueType>
struct workspace_traits<Fcg<ValueType>> {
    using Solver = Fcg<ValueType>;
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
    // t vector
    constexpr static int t = 4;
    // beta scalar
    constexpr static int beta = 5;
    // previous rho scalar
    constexpr static int prev_rho = 6;
    // current rho scalar
    constexpr static int rho = 7;
    // current rho_t scalar
    constexpr static int rho_t = 8;
    // constant 1.0 scalar
    constexpr static int one = 9;
    // constant -1.0 scalar
    constexpr static int minus_one = 10;

    // stopping status array
    constexpr static int stop = 0;
    // reduction tmp array
    constexpr static int tmp = 1;
};


}  // namespace solver
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_SOLVER_FCG_HPP_
