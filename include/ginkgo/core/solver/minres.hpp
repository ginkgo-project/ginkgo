// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_SOLVER_MINRES_HPP_
#define GKO_PUBLIC_CORE_SOLVER_MINRES_HPP_


#include <vector>

#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/log/logger.hpp>
#include <ginkgo/core/solver/solver_base.hpp>
#include <ginkgo/core/stop/combined.hpp>
#include <ginkgo/core/stop/criterion.hpp>


namespace gko {
namespace solver {


/**
 * Minres is an iterative type Krylov subspace method, which is suitable for
 * indefinite and full-rank symmetric/hermitian operators. It is an
 * specialization of the Gmres method for symmetric/hermitian operators, and can
 * be computed using short recurrences, similar to the CG method.
 *
 * The implementation in Ginkgo makes use of the merged kernel to make the best
 * use of data locality. The inner operations in one iteration of Minres are
 * merged into 2 separate steps.

 * @note: The Minres solver only reports an approximation of the residual norm
 *        directly to the stopping criteria. Neither the actual residual, nor
 *        the actual residual norm are reported. Thus, to get the minimal
 *        overhead, the gko::stop::ImplicitResidualNorm criteria should be used.
 *        The gko::stop::ResidualNorm criteria will require an additional
 *        matrix-vector product and global reduction.
 *
 * @tparam ValueType  precision of matrix elements
 *
 * @ingroup solvers
 * @ingroup LinOp
 */
template <typename ValueType = default_precision>
class Minres
    : public EnableLinOp<Minres<ValueType>>,
      public EnablePreconditionedIterativeSolver<ValueType, Minres<ValueType>>,
      public Transposable {
    friend class EnableLinOp<Minres>;
    friend class EnablePolymorphicObject<Minres, LinOp>;

public:
    using value_type = ValueType;
    using transposed_type = Minres;

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

    GKO_ENABLE_LIN_OP_FACTORY(Minres, parameters, Factory);
    GKO_ENABLE_BUILD_METHOD(Factory);

protected:
    void apply_impl(const LinOp* b, LinOp* x) const override;

    template <typename VectorType>
    void apply_dense_impl(const VectorType* dense_b, VectorType* dense_x) const;

    void apply_impl(const LinOp* alpha, const LinOp* b, const LinOp* beta,
                    LinOp* x) const override;

    explicit Minres(std::shared_ptr<const Executor> exec)
        : EnableLinOp<Minres>(std::move(exec))
    {}

    explicit Minres(const Factory* factory,
                    std::shared_ptr<const LinOp> system_matrix)
        : EnableLinOp<Minres>(factory->get_executor(),
                              gko::transpose(system_matrix->get_size())),
          EnablePreconditionedIterativeSolver<ValueType, Minres>{
              std::move(system_matrix), factory->get_parameters()},
          parameters_{factory->get_parameters()}
    {}
};


template <typename ValueType>
struct workspace_traits<Minres<ValueType>> {
    using Solver = Minres<ValueType>;
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
    // v vector
    constexpr static int v = 4;
    // z_tilde vector
    constexpr static int z_tilde = 5;
    // p_prev vector
    constexpr static int p_prev = 6;
    // q_prev vector
    constexpr static int q_prev = 7;
    // alpha scalar
    constexpr static int alpha = 8;
    // beta scalar
    constexpr static int beta = 9;
    // gamma scalar
    constexpr static int gamma = 10;
    // delta scalar
    constexpr static int delta = 11;
    // next eta scalar
    constexpr static int eta_next = 12;
    // current eta scalar
    constexpr static int eta = 13;
    // tau scalar
    constexpr static int tau = 14;
    // previous cos scalar
    constexpr static int cos_prev = 15;
    // current cos scalar
    constexpr static int cos = 16;
    // previous sin scalar
    constexpr static int sin_prev = 17;
    // current sin scalar
    constexpr static int sin = 18;
    // constant 1.0 scalar
    constexpr static int one = 19;
    // constant -1.0 scalar
    constexpr static int minus_one = 20;

    // stopping status array
    constexpr static int stop = 0;
    // reduction tmp array
    constexpr static int tmp = 1;
};


}  // namespace solver
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_SOLVER_MINRES_HPP_
