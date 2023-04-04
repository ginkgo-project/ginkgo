#ifndef GINKGO_SHARING_INFO_HPP
#define GINKGO_SHARING_INFO_HPP


namespace gko {
namespace experimental {
namespace distributed {


/**
 * Describes how received values should be handled.
 *
 * Ideally, this could also be a user defined binary operation.
 * But that requires being able to pass this through to kernels, which we don't
 * support atm.
 */
enum class sharing_mode {
    // Replace local DOFs with the incoming values from remote processors. This
    // is a partition of unity.
    set,
    // Add the incoming values from remote processors to the local DOFs. This is
    // not a partition of unity.
    add,
    // Perform a partition of unity with the provided weights.
    weighted
};


/**
 * Struct to describe a shared DOF.
 *
 * Alternatively, this could be a SOA.
 */
template <typename LocalIndexType>
struct shared_idx {
    // index within this rank
    LocalIndexType local_idx;
    // the index within the local indices of the remote_rank
    LocalIndexType remote_idx;
    // rank that shares the local DOF
    int remote_rank;
};


/**
 * Holds the necessary information for a decomposition of DOFs without relying
 * on global data. It consists of
 * - send/recv sizes, offsets, indices
 * - weights for shared DOFs
 * - transformation to combine received DOFs with existing DOFs
 *
 * The communicate step will perform the operation
 * ```
 * u_i = D_i R_i sum_j R_j^T v_j
 * ```
 * which combines the gather and scatter step for a gobal vector defined by a
 * partition of unity `u = sum_i R_i^T D_i R_i u`.
 *
 * It can be constructed from purely local information, i.e. which local DOF
 * corresponds to which DOF on other processes (in their local numbering).
 */
template <typename ValueType, typename LocalIndexType>
struct sharing_info {
    /**
     * Extracts communication pattern from a list of shared DOFs.
     */
    sharing_info(mpi::communicator comm,
                 const array<shared_idx<LocalIndexType>>& shared_idxs,
                 std::integral_constant<sharing_mode, sharing_mode::add> tag);
    sharing_info(mpi::communicator comm,
                 const array<shared_idx<LocalIndexType>>& shared_idxs,
                 std::integral_constant<sharing_mode, sharing_mode::set> tag);

    /**
     * Extracts communication pattern from a list of shared DOFs
     */
    sharing_info(mpi::communicator comm,
                 const array<shared_idx<LocalIndexType>>& shared_idxs,
                 const array<ValueType>& weights);

    static std::unique_ptr<sharing_info> create_from_send_info(
        mpi::communicator comm, const std::vector<int> send_sizes,
        const array<LocalIndexType> send_idxs);

    /**
     * Does the all-to-all communication on the given input and output vectors.
     * This handles the different supported sharing modes, and updates the
     * buffers accordingly. Could also be made into a two parameter function, by
     * using send=recv
     */
    template <typename SendType, typename RecvType>
    void communicate(mpi::communicator comm,
                     const matrix::Dense<SendType>* send,
                     matrix::Dense<RecvType>* recv) const;

    /**
     * Returns the weight for a receiving DOF. If idx is not received then
     * it returns 1.0.
     */
    ValueType get_multiplicity(LocalIndexType idx) const;

    /**
     * contains all necessary data for an all-to-all_v communication,
     * and gather/scatter indices
     */
    struct all_to_all_t {
        all_to_all_t(mpi::communicator comm,
                     const array<shared_idx<LocalIndexType>>& shared_idxs);

        // default variable all-to-all data
        std::vector<int> send_sizes;
        std::vector<int> send_offsets;
        std::vector<int> recv_sizes;
        std::vector<int> recv_offsets;

        /**
         * DOFs to send. These are dofs that are shared with other ranks,
         * but this rank owns them.
         */
        gko::array<LocalIndexType> send_idxs;
        /**
         * DOFs to send. These are dofs that are shared with other ranks,
         * but other ranks own them. May overlap with send_idxs.
         */
        gko::array<LocalIndexType> recv_idxs;
    };
    all_to_all_t all_to_all;

    /**
     * Stores the multiplicity of the receiving DOFs.
     * The multiplicity is stored as 1/sqrt(#Owners), if the mode is `add`.
     * If the mode is `set` then the value will always be 0, as no receiving DOF
     * is owned by this rank.
     */
    struct multiplicity_t {
        /**
         * maybe a better storage scheme using bitmaps would be possible
         */
        gko::array<LocalIndexType> idxs;
        gko::array<ValueType> weights;

        /**
         * strictly only needs the recv_idxs
         */
        multiplicity_t(const array<shared_idx<LocalIndexType>>& shared_idxs,
                       sharing_mode mode);
    };
    multiplicity_t multiplicity;

    sharing_mode mode;
};


}  // namespace distributed
}  // namespace experimental
}  // namespace gko

#endif  // GINKGO_SHARING_INFO_HPP
