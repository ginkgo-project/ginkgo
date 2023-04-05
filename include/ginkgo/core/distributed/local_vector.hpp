#ifndef GINKGO_LOCAL_VECTOR_HPP
#define GINKGO_LOCAL_VECTOR_HPP

#include <ginkgo/core/distributed/base.hpp>
#include <ginkgo/core/distributed/lin_op.hpp>
#include <ginkgo/core/distributed/sharing_info.hpp>
#include <ginkgo/core/matrix/dense.hpp>


namespace gko {
namespace experimental {

struct mask {
    gko::array<uint32> mask;
    size_type num_masked_elements;
};


namespace matrix {
template <typename ValueType>
class MaskedDense {
public:
    MaskedDense(std::shared_ptr<::gko::matrix::Dense<ValueType>> data,
                gko::array<uint8> mask);
    /**
     * has normal binary ops + fill
     */

private:
};
}  // namespace matrix


namespace distributed {

/**
 * Distributed vector that relies only on local information.
 *
 * Name subject to change
 */
template <typename ValueType>
class LocalVector : public EnableDistributedLinOp<LocalVector<ValueType>>,
                    public distributed::DistributedBase {
public:
    LocalVector(std::shared_ptr<const Executor> exec, mpi::communicator comm,
                dim<2> size,
                std::shared_ptr<sharing_info<ValueType, int32>> comm_info);
    LocalVector(std::shared_ptr<const Executor> exec, mpi::communicator comm,
                std::unique_ptr<::gko::matrix::Dense<ValueType>> data,
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
     * TODO: calling this twice may result in a inconsistent vector (add
     * sharing_mode)
     */
    void make_consistent();

    /**
     * Updates shared DOFs but overwrites the sharing mode.
     *
     * Useful for the assembly of non-overlapping distributed DOFs.
     */
    void make_consistent(sharing_mode overwrite_mode);

    /**
     * point-wise operation the same as dense
     */

    /**
     * reductions use multiplicity defined in sharing_info to either
     * - remove (zero-out) non-owned DOFs
     * - scale by sqrt(1/#Owned)
     */

    /**
     * Return views to the underlying dense vector
     */
    std::unique_ptr<const ::gko::matrix::Dense<ValueType>> get_dense() const;
    std::unique_ptr<::gko::matrix::Dense<ValueType>> get_dense();

    /**
     * Returns a masked view on the shared DOFs (not owned or partially owned)
     */
    std::unique_ptr<matrix::MaskedDense<ValueType>> get_shared();
    /**
     * Returns a masked view on the DOFs exclusive to this processor (may
     * contain DOFs that are shared on other processors)
     */
    std::unique_ptr<matrix::MaskedDense<ValueType>> get_exclusive();

private:
    std::unique_ptr<::gko::matrix::Dense<ValueType>> data;
    std::shared_ptr<const sharing_info<ValueType, int32>> comm_info;
};


}  // namespace distributed
}  // namespace experimental
}  // namespace gko


#endif  // GINKGO_LOCAL_VECTOR_HPP
