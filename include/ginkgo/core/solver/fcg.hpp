// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
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
#include <ginkgo/core/log/logger.hpp>
#include <ginkgo/core/matrix/dense.hpp>
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
 * In contrast to the standard CG based on the Polack-Ribiere formula, the
 * flexible CG uses the Fletcher-Reeves formula for creating the orthonormal
 * vectors spanning the Krylov subspace. This increases the computational cost
 * of every Krylov solver iteration but allows for non-constant preconditioners.
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
    : public EnableLinOp<Fcg<ValueType>>,
      public EnablePreconditionedIterativeSolver<ValueType, Fcg<ValueType>>,
      public Transposable {
    friend class EnableLinOp<Fcg>;
    friend class EnablePolymorphicObject<Fcg, LinOp>;

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

protected:
    void apply_impl(const LinOp* b, LinOp* x) const override;

    template <typename VectorType>
    void apply_dense_impl(const VectorType* b, VectorType* x) const;

    void apply_impl(const LinOp* alpha, const LinOp* b, const LinOp* beta,
                    LinOp* x) const override;

    explicit Fcg(std::shared_ptr<const Executor> exec)
        : EnableLinOp<Fcg>(std::move(exec))
    {}

    explicit Fcg(const Factory* factory,
                 std::shared_ptr<const LinOp> system_matrix)
        : EnableLinOp<Fcg>(factory->get_executor(),
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
    // alpha scalar
    constexpr static int alpha = 5;
    // beta scalar
    constexpr static int beta = 6;
    // previous rho scalar
    constexpr static int prev_rho = 7;
    // current rho scalar
    constexpr static int rho = 8;
    // current rho_t scalar
    constexpr static int rho_t = 9;
    // constant 1.0 scalar
    constexpr static int one = 10;
    // constant -1.0 scalar
    constexpr static int minus_one = 11;

    // stopping status array
    constexpr static int stop = 0;
    // reduction tmp array
    constexpr static int tmp = 1;
};


}  // namespace solver
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_SOLVER_FCG_HPP_
