// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_MULTIGRID_RS_HPP_
#define GKO_PUBLIC_CORE_MULTIGRID_RS_HPP_


#include <ginkgo/core/base/composition.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/multigrid/multigrid_level.hpp>

#include "ginkgo/core/base/math.hpp"


namespace gko {
namespace multigrid {


/**
 * Rs implements the Ruge–Stueben (classical) Algebraic Multigrid (AMG)
 * coarsening strategy for M-matrices. Given a sparse system $Ax = b$,
 * it produces one level of an AMG hierarchy: a C/F splitting, a prolongation
 * operator $P$, and a coarse-grid operator $A_c = R A P$.
 *
 * Coarsening proceeds in three steps. First, neighbour $j$ is marked as
 * *strongly influencing* row $i$ when $-a_{ij} \ge \theta \cdot
 * \max_{k \neq i}(-a_{ik})$. Second, a greedy pass selects C-points by
 * repeatedly picking the undecided node with the most strong neighbours,
 * marking its undecided strong neighbours as F-points, and updating neighbour
 * counts accordingly. Third, we create the coarse grid and compute the
 * interpolation via the classical RS direct interpolation formula, accounting
 * for both strong C- and F-neighbours.
 *
 * Ruge, J. W., & Stueben, K. (1987). Algebraic multigrid, Multigrid Methods
 * (Vol. 3, pp. 73–130). Society for Industrial and Applied Mathematics.
 * https://doi.org/10.1137/1.9781611971057.ch4
 *
 * @tparam ValueType  precision of matrix elements
 * @tparam IndexType  precision of matrix indexes
 *
 * @ingroup MultigridLevel
 * @ingroup Multigrid
 * @ingroup LinOp
 */
template <typename ValueType = default_precision, typename IndexType = int32>
class Rs : public EnableLinOp<Rs<ValueType, IndexType>>,
           public EnableMultigridLevel<ValueType> {
    friend class EnableLinOp<Rs>;
    friend class EnablePolymorphicObject<Rs, LinOp>;
    GKO_ASSERT_SUPPORTED_VALUE_AND_INDEX_TYPE;

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
         * Strength-of-connection threshold, theta
         */
        remove_complex<value_type> GKO_FACTORY_PARAMETER_SCALAR(
            strength_threshold, 0.25);

        bool GKO_FACTORY_PARAMETER_SCALAR(skip_sorting, false);

        bool GKO_FACTORY_PARAMETER_SCALAR(skip_m_matrix_check, false);
    };
    GKO_ENABLE_LIN_OP_FACTORY(Rs, parameters, Factory);
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

    explicit Rs(std::shared_ptr<const Executor> exec)
        : EnableLinOp<Rs>(std::move(exec))
    {}

    explicit Rs(const Factory* factory,
                std::shared_ptr<const LinOp> system_matrix)
        : EnableLinOp<Rs>(factory->get_executor(), system_matrix->get_size()),
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


#endif  // GKO_PUBLIC_CORE_MULTIGRID_RS_HPP_
