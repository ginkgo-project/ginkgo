/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2023, the Ginkgo authors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#ifndef GKO_PUBLIC_CORE_SOLVER_BATCH_LOWER_TRS_HPP_
#define GKO_PUBLIC_CORE_SOLVER_BATCH_LOWER_TRS_HPP_


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/batch_csr.hpp>
#include <ginkgo/core/matrix/batch_dense.hpp>
#include <ginkgo/core/matrix/batch_diagonal.hpp>
#include <ginkgo/core/matrix/batch_identity.hpp>
#include <ginkgo/core/matrix/identity.hpp>

namespace gko {
namespace solver {


/**
 * Solves a batch of linear systems using a batched lower triangular solver.
 *
 * @tparam ValueType  precision of matrix elements
 *
 * @ingroup solvers
 * @ingroup BatchLinOp
 */
template <typename ValueType = default_precision>
class BatchLowerTrs : public EnableBatchLinOp<BatchLowerTrs<ValueType>>,
                      public BatchTransposable {
    friend class EnableBatchLinOp<BatchLowerTrs>;
    friend class EnablePolymorphicObject<BatchLowerTrs, BatchLinOp>;

public:
    using value_type = ValueType;
    using real_type = gko::remove_complex<ValueType>;
    using transposed_type = BatchLowerTrs<ValueType>;

    /**
     * Returns the system operator (matrix) of the linear system.
     *
     * @return the system operator (matrix)
     */
    std::shared_ptr<const BatchLinOp> get_system_matrix() const
    {
        return system_matrix_;
    }

    /**
     * @return The left scaling vector.
     */
    std::shared_ptr<const BatchLinOp> get_left_scaling_op() const
    {
        return left_scaling_;
    }

    /**
     * @return The right scaling vector.
     */
    std::shared_ptr<const BatchLinOp> get_right_scaling_op() const
    {
        return right_scaling_;
    }

    std::unique_ptr<BatchLinOp> transpose() const override;

    std::unique_ptr<BatchLinOp> conj_transpose() const override;

    GKO_CREATE_FACTORY_PARAMETERS(parameters, Factory)
    {
        /**
         * Batch diagonal matrix for scaling the system matrix from the left
         * before the solve. Note that `right_scaling_op` must also be set if
         * this is set.
         */
        std::shared_ptr<const BatchLinOp> GKO_FACTORY_PARAMETER_SCALAR(
            left_scaling_op, nullptr);

        /**
         * Batch diagonal matrix for scaling the system matrix from the right
         * before the solve. Note that `left_scaling_op` must also be set if
         * this is set.
         */
        std::shared_ptr<const BatchLinOp> GKO_FACTORY_PARAMETER_SCALAR(
            right_scaling_op, nullptr);

        /**
         * @brief Optimization parameter that skips the sorting of the input
         *        matrix (only skip if it is known that it is already sorted)
         * (in reference to BatchCsr matrix format).
         *
         * The triangular solve algorithm requires the input matrix to be
         * sorted. If it is, this parameter can be set to `true` to skip the
         * sorting for better performance.
         */
        bool GKO_FACTORY_PARAMETER_SCALAR(skip_sorting, false);
    };
    GKO_ENABLE_BATCH_LIN_OP_FACTORY(BatchLowerTrs, parameters, Factory);
    GKO_ENABLE_BUILD_METHOD(Factory);

protected:
    void apply_impl(const BatchLinOp* b, BatchLinOp* x) const override;

    void apply_impl(const BatchLinOp* alpha, const BatchLinOp* b,
                    const BatchLinOp* beta, BatchLinOp* x) const override;

    explicit BatchLowerTrs(std::shared_ptr<const Executor> exec)
        : EnableBatchLinOp<BatchLowerTrs>(std::move(exec))
    {}

    explicit BatchLowerTrs(const Factory* factory,
                           std::shared_ptr<const BatchLinOp> system_matrix)
        : EnableBatchLinOp<BatchLowerTrs>(
              factory->get_executor(),
              gko::transpose(system_matrix->get_size())),
          parameters_{factory->get_parameters()},
          system_matrix_{std::move(system_matrix)}
    {
        GKO_ASSERT_BATCH_HAS_SQUARE_MATRICES(system_matrix_);

        left_scaling_ = parameters_.left_scaling_op;
        right_scaling_ = parameters_.right_scaling_op;

        auto exec = factory->get_executor();

        using BIdentity = matrix::BatchIdentity<value_type>;
        using BDiag = matrix::BatchDiagonal<value_type>;

        // scale the system matrix in the solver linop generation step instead
        // of solver object apply (similar to what is done for batch iterative
        // solvers)
        const bool to_scale =
            std::dynamic_pointer_cast<const BDiag>(left_scaling_) &&
            std::dynamic_pointer_cast<const BDiag>(right_scaling_);

        if (to_scale) {
            auto a_scaled_smart = gko::share(gko::clone(system_matrix_.get()));
            gko::matrix::two_sided_batch_transform(
                exec, (as<const BDiag>(left_scaling_)).get(),
                (as<const BDiag>(right_scaling_)).get(), a_scaled_smart.get());
            system_matrix_ = a_scaled_smart;
        }

        if (!to_scale && left_scaling_ && right_scaling_) {
            GKO_NOT_SUPPORTED(left_scaling_);
        }

        if (!to_scale && (left_scaling_ || right_scaling_)) {
            throw std::runtime_error("One-sided scaling is not supported!");
        }

        if (!to_scale) {
            // this enables transpose for non-scaled solvers
            left_scaling_ =
                gko::share(BIdentity::create(exec, system_matrix_->get_size()));
            right_scaling_ =
                gko::share(BIdentity::create(exec, system_matrix_->get_size()));
        }
    }

private:
    std::shared_ptr<const BatchLinOp> system_matrix_{};
    std::shared_ptr<const BatchLinOp> left_scaling_{};
    std::shared_ptr<const BatchLinOp> right_scaling_{};
};


}  // namespace solver
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_SOLVER_BATCH_LOWER_TRS_HPP_
