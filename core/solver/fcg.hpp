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
#include "core/base/convertible.hpp"
#include "core/base/lin_op.hpp"
#include "core/base/math.hpp"
#include "core/base/types.hpp"


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
class Fcg : public LinOp, public ConvertibleTo<Fcg<ValueType>> {
    friend class FcgFactory<ValueType>;

public:
    using value_type = ValueType;

    void copy_from(const LinOp *other) override;

    void copy_from(std::unique_ptr<LinOp> other) override;

    void apply(const LinOp *b, LinOp *x) const override;

    void apply(const LinOp *alpha, const LinOp *b, const LinOp *beta,
               LinOp *x) const override;

    std::unique_ptr<LinOp> clone_type() const override;

    void clear() override;

    void convert_to(Fcg *result) const override;

    void move_to(Fcg *result) override;

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

private:
    Fcg(std::shared_ptr<const Executor> exec, int max_iters,
        remove_complex<value_type> rel_residual_goal,
        std::shared_ptr<const LinOp> system_matrix)
        : LinOp(exec, system_matrix->get_num_cols(),
                system_matrix->get_num_rows(),
                system_matrix->get_num_rows() * system_matrix->get_num_cols()),
          system_matrix_(std::move(system_matrix)),
          max_iters_(max_iters),
          rel_residual_goal_(rel_residual_goal)
    {}

    static std::unique_ptr<Fcg> create(
        std::shared_ptr<const Executor> exec, int max_iters,
        remove_complex<value_type> rel_residual_goal,
        std::shared_ptr<const LinOp> system_matrix)
    {
        return std::unique_ptr<Fcg>(new Fcg(std::move(exec), max_iters,
                                            rel_residual_goal,
                                            std::move(system_matrix)));
    }

    std::shared_ptr<const LinOp> system_matrix_;
    int max_iters_;
    remove_complex<value_type> rel_residual_goal_;
};

/**
 * The FcgFactory class is derived from the LinOpFactory class and is
 * used to generate the FCG solver.
 */
template <typename ValueType = default_precision>
class FcgFactory : public LinOpFactory {
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

    std::unique_ptr<LinOp> generate(
        std::shared_ptr<const LinOp> base) const override;

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
    explicit FcgFactory(std::shared_ptr<const Executor> exec, int max_iters,
                        remove_complex<value_type> rel_residual_goal)
        : LinOpFactory(std::move(exec)),
          max_iters_(max_iters),
          rel_residual_goal_(rel_residual_goal)
    {}

    int max_iters_;
    remove_complex<value_type> rel_residual_goal_;
};


}  // namespace solver
}  // namespace gko


#endif  // GKO_CORE_SOLVER_FCG_HPP
