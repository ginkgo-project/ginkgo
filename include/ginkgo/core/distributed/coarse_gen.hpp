/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2021, the Ginkgo authors
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

#ifndef GKO_PUBLIC_CORE_DISTRIBUTED_COARSE_GEN_HPP_
#define GKO_PUBLIC_CORE_DISTRIBUTED_COARSE_GEN_HPP_


#include <vector>


#include <ginkgo/config.hpp>


#if GINKGO_BUILD_MPI


#include <ginkgo/core/base/composition.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/distributed/matrix.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/multigrid/multigrid_level.hpp>


namespace gko {
namespace experimental {
namespace distributed {


/**
 *
 * @tparam ValueType  precision of matrix elements
 * @tparam IndexType  precision of matrix indexes
 *
 * @ingroup MultigridLevel
 * @ingroup Multigrid
 * @ingroup LinOp
 * TODO: GlobalIndexType ?
 */
template <typename ValueType = default_precision, typename IndexType = int32>
class CoarseGen : public EnableLinOp<CoarseGen<ValueType, IndexType>>,
                  public multigrid::EnableMultigridLevel<ValueType> {
    friend class EnableLinOp<CoarseGen>;
    friend class polymorphic_object_traits<CoarseGen>;

public:
    using value_type = ValueType;
    using index_type = IndexType;

    /**
     * Returns the system operator (matrix) of the linear system.
     *
     * @return the system operator (matrix)
     */
    std::shared_ptr<const LinOp> get_system_matrix() const
    {
        return system_matrix_;
    }

    /**
     * Returns the aggregate group.
     *
     * Aggregate group whose size is same as the number of rows. Stores the
     * mapping information from row index to coarse row index.
     * i.e., agg[row_idx] = coarse_row_idx.
     *
     * @return the aggregate group.
     */
    IndexType* get_coarse_indices_map() noexcept
    {
        return coarse_indices_map_.get_data();
    }

    /**
     * @copydoc CoarseGen::get_agg()
     *
     * @note This is the constant version of the function, which can be
     *       significantly more memory efficient than the non-constant version,
     *       so always prefer this version.
     */
    const IndexType* get_const_coarse_indices_map() const noexcept
    {
        return coarse_indices_map_.get_const_data();
    }

    enum class strategy_type { aggregation, selection };

    GKO_CREATE_FACTORY_PARAMETERS(parameters, Factory)
    {
        /**
         * The strategy to generate the coarse matrix
         */
        strategy_type GKO_FACTORY_PARAMETER_SCALAR(strategy,
                                                   strategy_type::selection);

        /**
         * The strategy to generate the coarse matrix
         */
        size_type GKO_FACTORY_PARAMETER_SCALAR(spacing, 2);

        /**
         * If the system matrix is Hermitian, then optimizations allow for easy
         * generation of the weight matrix.
         */
        bool GKO_FACTORY_PARAMETER_SCALAR(hermitian, false);

        /**
         * The maximum number of iterations. We use the same default value as
         * NVIDIA COARSE Reference Manual (October 2017, API Version 2,
         * https://github.com/NVIDIA/COARSE/blob/main/doc/COARSE_Reference.pdf).
         */
        unsigned GKO_FACTORY_PARAMETER_SCALAR(max_iterations, 15u);

        /**
         * The maximum ratio of unassigned number, which is valid in the
         * interval 0.0 ~ 1.0. We use the same default value as NVIDIA COARSE
         * Reference Manual (October 2017, API Version 2,
         * https://github.com/NVIDIA/COARSE/blob/main/doc/COARSE_Reference.pdf).
         */
        double GKO_FACTORY_PARAMETER_SCALAR(max_unassigned_ratio, 0.05);

        /**
         * Use the deterministic assign_to_exist_agg method or not.
         *
         * If deterministic is set to true, always get the same aggregated group
         * from the same matrix. Otherwise, the aggregated group might be
         * different depending on the execution ordering.
         */
        bool GKO_FACTORY_PARAMETER_SCALAR(deterministic, false);

        /**
         * The `system_matrix`, which will be given to this factory, must be
         * sorted (first by row, then by column) in order for the algorithm
         * to work. If it is known that the matrix will be sorted, this
         * parameter can be set to `true` to skip the sorting (therefore,
         * shortening the runtime).
         * However, if it is unknown or if the matrix is known to be not sorted,
         * it must remain `false`, otherwise, this multigrid_level might be
         * incorrect.
         */
        bool GKO_FACTORY_PARAMETER_SCALAR(skip_sorting, false);
    };
    GKO_ENABLE_LIN_OP_FACTORY(CoarseGen, parameters, Factory);
    GKO_ENABLE_BUILD_METHOD(Factory);

protected:
    void apply_impl(const LinOp* b, LinOp* x) const override
    {
        this->get_composition()->apply(b, x);
    }

    void apply_impl(const LinOp* alpha, const LinOp* b, const LinOp* beta,
                    LinOp* x) const override
    {
        this->get_composition()->apply(alpha, b, beta, x);
    }

    explicit CoarseGen(std::shared_ptr<const Executor> exec)
        : EnableLinOp<CoarseGen>(std::move(exec))
    {}

    explicit CoarseGen(const Factory* factory,
                       std::shared_ptr<const LinOp> system_matrix)
        : EnableLinOp<CoarseGen>(factory->get_executor(),
                                 system_matrix->get_size()),
          multigrid::EnableMultigridLevel<ValueType>(system_matrix),
          parameters_{factory->get_parameters()},
          system_matrix_{system_matrix},
          coarse_indices_map_(factory->get_executor(),
                              system_matrix_->get_size()[0])
    {
        GKO_ASSERT(parameters_.max_unassigned_ratio <= 1.0);
        GKO_ASSERT(parameters_.max_unassigned_ratio >= 0.0);
        if (system_matrix_->get_size()[0] != 0) {
            if (parameters_.strategy == strategy_type::aggregation) {
                this->generate_with_aggregation();
            } else if (parameters_.strategy == strategy_type::selection) {
                this->generate_with_selection();
            }
        }
    }

    void generate_with_aggregation();

    void generate_with_selection();

private:
    std::shared_ptr<const LinOp> system_matrix_{};
    array<IndexType> coarse_indices_map_{};
};


}  // namespace distributed
}  // namespace experimental
}  // namespace gko


#endif
#endif  // GKO_PUBLIC_CORE_DISTRIBUTED_COARSE_GEN_HPP_
