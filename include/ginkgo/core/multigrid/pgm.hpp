// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_MULTIGRID_PGM_HPP_
#define GKO_PUBLIC_CORE_MULTIGRID_PGM_HPP_


#include <vector>


#include <ginkgo/core/base/composition.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/multigrid/multigrid_level.hpp>

namespace gko {
namespace multigrid {


/**
 * Parallel graph match (Pgm) is the aggregate method introduced in the
 * paper M. Naumov et al., "AmgX: A Library for GPU Accelerated Algebraic
 * Multigrid and Preconditioned Iterative Methods". Current implementation only
 * contains size = 2 version.
 *
 * Pgm creates the aggregate group according to the matrix value not the
 * structure. Pgm gives two steps (one-phase handshaking) to group the
 * elements.
 * 1: get the strongest neighbor of each unaggregated element.
 * 2: group the elements whose strongest neighbor is each other.
 * repeating until reaching the given conditions. After that, the
 * un-aggregated elements are assigned to an aggregated group
 * or are left alone.
 *
 * @tparam ValueType  precision of matrix elements
 * @tparam IndexType  precision of matrix indexes
 *
 * @ingroup MultigridLevel
 * @ingroup Multigrid
 * @ingroup LinOp
 */
template <typename ValueType = default_precision, typename IndexType = int32>
class Pgm : public EnableLinOp<Pgm<ValueType, IndexType>>,
            public EnableMultigridLevel<ValueType> {
    friend class EnableLinOp<Pgm>;
    friend class EnablePolymorphicObject<Pgm, LinOp>;

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
    IndexType* get_agg() noexcept { return agg_.get_data(); }

    /**
     * @copydoc Pgm::get_agg()
     *
     * @note This is the constant version of the function, which can be
     *       significantly more memory efficient than the non-constant version,
     *       so always prefer this version.
     */
    const IndexType* get_const_agg() const noexcept
    {
        return agg_.get_const_data();
    }

    GKO_CREATE_FACTORY_PARAMETERS(parameters, Factory)
    {
        /**
         * The maximum number of iterations. We use the same default value as
         * NVIDIA AMGX Reference Manual (October 2017, API Version 2,
         * https://github.com/NVIDIA/AMGX/blob/main/doc/AMGX_Reference.pdf).
         */
        unsigned GKO_FACTORY_PARAMETER_SCALAR(max_iterations, 15u);

        /**
         * The maximum ratio of unassigned number, which is valid in the
         * interval 0.0 ~ 1.0. We use the same default value as NVIDIA AMGX
         * Reference Manual (October 2017, API Version 2,
         * https://github.com/NVIDIA/AMGX/blob/main/doc/AMGX_Reference.pdf).
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
    GKO_ENABLE_LIN_OP_FACTORY(Pgm, parameters, Factory);
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

    explicit Pgm(std::shared_ptr<const Executor> exec)
        : EnableLinOp<Pgm>(std::move(exec))
    {}

    explicit Pgm(const Factory* factory,
                 std::shared_ptr<const LinOp> system_matrix)
        : EnableLinOp<Pgm>(factory->get_executor(), system_matrix->get_size()),
          EnableMultigridLevel<ValueType>(system_matrix),
          parameters_{factory->get_parameters()},
          system_matrix_{system_matrix},
          agg_(factory->get_executor(), system_matrix_->get_size()[0])
    {
        GKO_ASSERT(parameters_.max_unassigned_ratio <= 1.0);
        GKO_ASSERT(parameters_.max_unassigned_ratio >= 0.0);
        if (system_matrix_->get_size()[0] != 0) {
            // generate on the existed matrix
            this->generate();
        }
    }

    void generate();

private:
    std::shared_ptr<const LinOp> system_matrix_{};
    array<IndexType> agg_;
};


template <typename ValueType = default_precision, typename IndexType = int32>
using AmgxPgm GKO_DEPRECATED(
    "This class is deprecated and will be removed in the next "
    "major release. Please use Pgm instead.") = Pgm<ValueType, IndexType>;


}  // namespace multigrid
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_MULTIGRID_PGM_HPP_
