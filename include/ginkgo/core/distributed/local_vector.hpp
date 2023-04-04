#ifndef GINKGO_LOCAL_VECTOR_HPP
#define GINKGO_LOCAL_VECTOR_HPP

#include <ginkgo/core/distributed/base.hpp>
#include <ginkgo/core/distributed/lin_op.hpp>
#include <ginkgo/core/distributed/sharing_info.hpp>
#include <ginkgo/core/matrix/dense.hpp>


namespace gko {
namespace experimental {
namespace distributed {

/**
 * Distributed vector that relies only on local information.
 *
 * Name subject to change
 */
template <typename ValueType>
class LocalVector : public EnableDistributedLinOp<LocalVector<ValueType>>,
                    public distributed::DistributedBase {
    LocalVector(std::shared_ptr<const Executor> exec, mpi::communicator comm,
                std::shared_ptr<sharing_info<ValueType, int32>> comm_info);

    /**
     * Updates shared DOFs with the values from their shared ranks.
     *
     * After this call all DOFs have the values of the corresponding global
     * DOFs. Considering a distributed operator that can be written as A = sum_i
     * R_i^T A_i R_i, then the corresponding global vectors are u = sum_i
     * R_i^T D_i u_i. The local vector can then be recovered by u_j = R_j u.
     * This operation combines these two steps into one as u_j = R_j sum_i R_i^T
     * D_i u_i, which eliminates the need for a global vector.
     * TODO: has to be blocking, but could be made non-blocking by partitioning
     * the DOFs by owned/non-owned
     */
    void make_consistent();

    /**
     * point-wise operation the same as dense
     */

    /**
     * reductions use weights defined in sharing_info to
     * - remove (zero-out) non-owned DOFs
     * - scale by sqrt(1/#Owned)
     */

    /**
     * Return views to the underlying dense vector
     */
    std::unique_ptr<const matrix::Dense<ValueType>> get_dense() const;
    std::unique_ptr<matrix::Dense<ValueType>> get_dense();

private:
    std::unique_ptr<matrix::Dense<ValueType>> data;
    std::shared_ptr<const sharing_info<ValueType, int32>> comm_info;
};


}  // namespace distributed
}  // namespace experimental
}  // namespace gko


#endif  // GINKGO_LOCAL_VECTOR_HPP
