/*******************************<GINKGO LICENSE>******************************
Copyright 2017-2018

Karlsruhe Institute of Technology
Universitat Jaume I
University of Tennessee

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#ifndef GKO_CORE_SOLVER_FCG_HPP_
#define GKO_CORE_SOLVER_FCG_HPP_


#include "core/base/array.hpp"
#include "core/base/lin_op.hpp"
#include "core/base/math.hpp"
#include "core/base/types.hpp"
#include "core/matrix/identity.hpp"


namespace gko {
namespace solver {


template <typename>
class FcgFactory;

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
 */
template <typename ValueType = default_precision>
class Fcg : public EnableLinOp<Fcg<ValueType>>, public PreconditionedMethod {
    friend class EnableLinOp<Fcg>;
    friend class EnablePolymorphicObject<Fcg, LinOp>;
    friend class FcgFactory<ValueType>;

public:
    using EnableLinOp<Fcg>::convert_to;
    using EnableLinOp<Fcg>::move_to;

    using value_type = ValueType;

    /**
     * Gets the system matrix of the linear system.
     *
     * @return  The system matrix.
     */
    std::shared_ptr<const LinOp> get_system_matrix() const
    {
        return system_matrix_;
    }

    /**
     * Gets the maximum number of iterations of the FCG solver.
     *
     * @return  The maximum number of iterations.
     */
    int get_max_iters() const noexcept { return max_iters_; }

    /**
     * Gets the relative residual goal of the solver.
     *
     * @return  The relative residual goal.
     */
    remove_complex<value_type> get_rel_residual_goal() const noexcept
    {
        return rel_residual_goal_;
    }

protected:
    explicit Fcg(std::shared_ptr<const Executor> exec) : EnableLinOp<Fcg>(exec)
    {}

    Fcg(std::shared_ptr<const Executor> exec, int max_iters,
        remove_complex<value_type> rel_residual_goal,
        std::shared_ptr<const LinOp> system_matrix)
        : EnableLinOp<Fcg>(exec,
                           system_matrix->get_dimensions().transpose().fill()),
          system_matrix_(std::move(system_matrix)),
          max_iters_(max_iters),
          rel_residual_goal_(rel_residual_goal)
    {}

    void apply_impl(const LinOp *b, LinOp *x) const override;

    void apply_impl(const LinOp *alpha, const LinOp *b, const LinOp *beta,
                    LinOp *x) const override;

private:
    std::shared_ptr<const LinOp> system_matrix_{};
    int max_iters_{};
    remove_complex<value_type> rel_residual_goal_{};
};


/**
 * The FcgFactory class is derived from the LinOpFactory class and is
 * used to generate the FCG solver.
 */
template <typename ValueType = default_precision>
class FcgFactory
    : public EnablePolymorphicObject<FcgFactory<ValueType>, LinOpFactory>,
      public PreconditionedMethodFactory {
    friend class EnablePolymorphicObject<FcgFactory, LinOpFactory>;

public:
    using value_type = ValueType;
    /**
     * Creates the FCG factory.
     *
     * @param exec The executor on which the FCG solver is to be created.
     * @param max_iters  The maximum number of iterations to be pursued.
     * @param rel_residual_goal  The relative residual required for
     * convergence.
     *
     * @return The newly created FCG factory.
     */
    static std::unique_ptr<FcgFactory> create(
        std::shared_ptr<const Executor> exec, int max_iters,
        remove_complex<value_type> rel_residual_goal)
    {
        return std::unique_ptr<FcgFactory>(
            new FcgFactory(std::move(exec), max_iters, rel_residual_goal));
    }

    /**
     * Gets the maximum number of iterations of the FCG solver.
     *
     * @return  The maximum number of iterations.
     */
    int get_max_iters() const noexcept { return max_iters_; }

    /**
     * Gets the relative residual goal of the solver.
     *
     * @return  The relative residual goal.
     */
    remove_complex<value_type> get_rel_residual_goal() const noexcept
    {
        return rel_residual_goal_;
    }

protected:
    FcgFactory(std::shared_ptr<const Executor> exec)
        : EnablePolymorphicObject<FcgFactory, LinOpFactory>(std::move(exec)),
          PreconditionedMethodFactory(
              matrix::IdentityFactory<ValueType>::create(std::move(exec))),
          max_iters_{},
          rel_residual_goal_{}
    {}

    explicit FcgFactory(std::shared_ptr<const Executor> exec, int max_iters,
                        remove_complex<value_type> rel_residual_goal)
        : EnablePolymorphicObject<FcgFactory, LinOpFactory>(std::move(exec)),
          PreconditionedMethodFactory(
              matrix::IdentityFactory<ValueType>::create(std::move(exec))),
          max_iters_(max_iters),
          rel_residual_goal_(rel_residual_goal)
    {}

    std::unique_ptr<LinOp> generate_impl(
        std::shared_ptr<const LinOp> base) const override;

    int max_iters_;
    remove_complex<value_type> rel_residual_goal_;
};


}  // namespace solver
}  // namespace gko


#endif  // GKO_CORE_SOLVER_FCG_HPP
