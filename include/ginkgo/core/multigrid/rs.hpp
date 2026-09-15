// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_MULTIGRID_RS_HPP_
#define GKO_PUBLIC_CORE_MULTIGRID_RS_HPP_


#include <tuple>

#include <ginkgo/core/base/composition.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/config/config.hpp>
#include <ginkgo/core/distributed/matrix.hpp>
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
class Rs : public LinOp, public EnableMultigridLevel<ValueType> {
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
        // Strength-of-connection threshold, theta
        double GKO_FACTORY_PARAMETER_SCALAR(strength_threshold, 0.25);

        // Skips Csr sorting if set to true
        bool GKO_FACTORY_PARAMETER_SCALAR(skip_sorting, false);

        // RS-coarsening only works for M-matrices. If this is true, skips this
        // (potentially heavy) check
        bool GKO_FACTORY_PARAMETER_SCALAR(skip_m_matrix_check, false);
    };
    GKO_ENABLE_LIN_OP_FACTORY(Rs, parameters, Factory);
    GKO_ENABLE_BUILD_METHOD(Factory);

    static parameters_type parse(
        const config::pnode& config, const config::registry& context,
        const config::type_descriptor& td_for_child =
            config::make_type_descriptor<ValueType, IndexType>());

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

    explicit Rs(std::shared_ptr<const Executor> exec) : LinOp(std::move(exec))
    {}

    explicit Rs(const Factory* factory,
                std::shared_ptr<const LinOp> system_matrix)
        : LinOp(factory->get_executor(), system_matrix->get_size()),
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

    /**
     * Generates the coarsening operators for a single, process-local matrix.
     *
     * @param local_matrix  the local (diagonal) block to coarsen
     * @param off_diag_matrix  the off-diagonal block of a distributed matrix,
     *                         or nullptr. It only enters the
     *                         strength-of-connection threshold, see
     *                         GKO_DECLARE_RS_COMPUTE_SOC_AND_RUN_RS_KERNEL.
     * @param num_forced_c_points  number of rows in `forced_c_points`
     * @param forced_c_points  local rows that must end up in the coarse set,
     *                         regardless of what the greedy pass decided
     *
     * @return a tuple with prolongation, coarse, and restriction linop
     */
    std::tuple<std::shared_ptr<LinOp>, std::shared_ptr<LinOp>,
               std::shared_ptr<LinOp>>
    generate_local(
        std::shared_ptr<const matrix::Csr<ValueType, IndexType>> local_matrix,
        const matrix::Csr<ValueType, IndexType>* off_diag_matrix = nullptr,
        size_type num_forced_c_points = 0,
        const IndexType* forced_c_points = nullptr);

#if GINKGO_BUILD_MPI
    /**
     * Communicates the coarse index of every local row a neighboring rank
     * couples to, in the coarse matrix' global indexing.
     *
     * All of those rows are forced C-points, so each of them has exactly one
     * coarse index - which is what keeps the prolongation block-diagonal and
     * this exchange down to a single index per halo entry.
     *
     * @tparam GlobalIndexType  Global index type
     *
     * @param matrix  a distributed matrix
     * @param coarse_partition  the coarse partition, used to compute the new
     *                          global indices
     * @param local_fine_to_coarse  the local fine-to-coarse map
     *
     * @return the coarse global index of every off-diag column
     */
    template <typename GlobalIndexType>
    array<GlobalIndexType> communicate_off_diag_coarse_idxs(
        std::shared_ptr<const experimental::distributed::Matrix<
            ValueType, IndexType, GlobalIndexType>>
            matrix,
        std::shared_ptr<
            experimental::distributed::Partition<IndexType, GlobalIndexType>>
            coarse_partition,
        const array<IndexType>& local_fine_to_coarse);
#endif

private:
    std::shared_ptr<const LinOp> system_matrix_{};
    // the fine-to-coarse map of the last generate_local call. The distributed
    // path needs it after generate_local returned, to tell the neighbors which
    // coarse index their halo rows became.
    array<IndexType> fine_to_coarse_{};
};


}  // namespace multigrid
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_MULTIGRID_RS_HPP_
