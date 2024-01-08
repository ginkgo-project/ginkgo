// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_MULTIGRID_FIXED_COARSENING_HPP_
#define GKO_PUBLIC_CORE_MULTIGRID_FIXED_COARSENING_HPP_


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
 * FixedCoarsening is a very simple coarse grid generation algorithm. It selects
 * the coarse matrix from the fine matrix by with user-specified indices.
 *
 * The user needs to specify the indices (with global numbering) of the fine
 * matrix, they wish to be in the coarse matrix. The restriction and
 * prolongation matrices will map to and from the coarse space without any
 * interpolation or weighting.
 *
 *
 * @tparam ValueType  precision of matrix elements
 * @tparam IndexType  precision of matrix indexes
 *
 * @ingroup MultigridLevel
 * @ingroup Multigrid
 * @ingroup LinOp
 */
template <typename ValueType = default_precision, typename IndexType = int32>
class FixedCoarsening
    : public EnableLinOp<FixedCoarsening<ValueType, IndexType>>,
      public EnableMultigridLevel<ValueType> {
    friend class EnableLinOp<FixedCoarsening>;
    friend class EnablePolymorphicObject<FixedCoarsening, LinOp>;

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


    GKO_CREATE_FACTORY_PARAMETERS(parameters, Factory)
    {
        /**
         * This array specifies the rows of the fine matrix which should be
         * selected as rows for the coarse matrix.
         */
        array<IndexType> GKO_FACTORY_PARAMETER_VECTOR(coarse_rows, nullptr);

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
    GKO_ENABLE_LIN_OP_FACTORY(FixedCoarsening, parameters, Factory);
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

    explicit FixedCoarsening(std::shared_ptr<const Executor> exec)
        : EnableLinOp<FixedCoarsening>(std::move(exec))
    {}

    explicit FixedCoarsening(const Factory* factory,
                             std::shared_ptr<const LinOp> system_matrix)
        : EnableLinOp<FixedCoarsening>(factory->get_executor(),
                                       system_matrix->get_size()),
          EnableMultigridLevel<ValueType>(system_matrix),
          parameters_{factory->get_parameters()},
          system_matrix_{system_matrix}
    {
        if (system_matrix_->get_size()[0] != 0) {
            // generate on the existing matrix
            this->generate();
        }
    }

    void generate();

private:
    std::shared_ptr<const LinOp> system_matrix_{};
};


}  // namespace multigrid
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_MULTIGRID_FIXED_COARSENING_HPP_
