// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_DISTRIBUTED_INDEX_MAP_HPP_
#define GKO_PUBLIC_CORE_DISTRIBUTED_INDEX_MAP_HPP_


#include <ginkgo/config.hpp>
#include <ginkgo/core/base/collection.hpp>
#include <ginkgo/core/base/dense_cache.hpp>
#include <ginkgo/core/base/index_set.hpp>
#include <ginkgo/core/distributed/base.hpp>
#include <ginkgo/core/distributed/lin_op.hpp>
#include <ginkgo/core/distributed/partition.hpp>
#include <ginkgo/core/matrix/dense.hpp>


namespace gko::experimental::distributed {


template <typename IndexType>
struct semi_global_index {
    comm_index_type proc_id;
    IndexType local_idx;
};

/**
 * \brief Index space classification for the locally stored indices.
 */
enum class index_space {
    local,      //!< indices that are locally owned
    non_local,  //!< indices that are owned by other processes
    combined    //!< both local and non_local indices
};


/**
 * \brief This class defines mappings between global and local indices.
 *
 * Given an index space $I = [0, \dots, N)$ that is partitioned into $P$
 * disjoint subsets $I_k, k = 1, \dots, P$, this class defines for each
 * subset an extended global index set $I_k \subset \hat{I}_k$. The extended
 * index set contains the global indices owned by part $k$, as well as
 * remote indices $R_k = \hat{I}_k \setminus I_k$, which are also accessed by
 * part $k$, but owned by parts $l \neq k$.
 * The local indices (index_space::combined) are then defined as $[0, \dots,
 * |\hat{I}_k|)$. Additionally, the local indices can be separated into locally
 * owned (index_space::local) and non-locally owned (index_space::non_local).
 * The locally owned indices are defined as $[0, \dots, |I_k|)$, and the
 * non-locally owned as $[0, \dots, |R_k|)$.
 * With these index sets, the following mappings are defined:
 * - $c_k : \hat{I}_k \mapsto [0, \dots, |\hat{I}_k|)$ which maps global indices
 *   into the combined/full local index space,
 * - $l_k: I_k \mapsto [0, \dots, |I_k|)$ which maps global indices into the
 *   locally owned index space,
 * - $r_k: R_k \mapsto [0, \dots, |R_k|)$ which maps global indices into the
 *   non-locally owned index space.
 * The wanted map can be selected by passing an index_space.
 *
 * The index map for $I_k$ has no knowledge about any other index maps for
 * $I_l, l \neq k$. In particular, any global index passed to the `get_local`
 * map that is not part of the specified index space, will be mapped to an
 * invalid_index.
 *
 * \tparam LocalIndexType type for local indices
 * \tparam GlobalIndexType type for global indices
 */
template <typename LocalIndexType, typename GlobalIndexType = int64>
struct index_map {
    using part_type = Partition<LocalIndexType, GlobalIndexType>;

    /**
     * \brief Maps global indices to local indices
     * \param global_ids the global indices to map
     * \param is the index space in which the returned local indices are defined
     * \return the mapped local indices. Any global index that is not in the
     *         specified index space is mapped to invalid_index.
     */
    [[nodiscard]] array<LocalIndexType> get_local(
        const array<GlobalIndexType>& global_ids,
        index_space is = index_space::combined) const;

    /**
     * \brief get size of index_space::local
     */
    [[nodiscard]] size_type get_local_size() const
    {
        return partition_->get_part_size(rank_);
    }

    /**
     * \brief get size of index_space::non_local
     */
    [[nodiscard]] size_type get_non_local_size() const
    {
        return remote_global_idxs_.get_flat().get_size();
    }

    index_map(std::shared_ptr<const Executor> exec,
              std::shared_ptr<const part_type> part, comm_index_type rank,
              const array<GlobalIndexType>& recv_connections,
              const array<comm_index_type>& send_target_ids,
              const collection::array<LocalIndexType>& send_connections);

    /**
     * \brief Creates a new index map
     *
     * The passed in recv_connections may contain duplicates, which will be
     * filtered out.
     *
     * \param exec the executor
     * \param comm the communicator
     * \param part the partition of the global index set
     * \param recv_connections the global indices that are not owned by this
     *                         rank, but accessed by it
     */
    index_map(std::shared_ptr<const Executor> exec, mpi::communicator comm,
              std::shared_ptr<const part_type> part,
              const array<GlobalIndexType>& recv_connections);

    /**
     * \brief Creates an empty index map.
     */
    index_map() = default;

    /**
     * \brief get the index set $R_k$ for this rank.
     *
     * The indices are orderd by their owning rank and global index.
     */
    const collection::array<GlobalIndexType>& get_remote_global_idxs() const
    {
        return remote_global_idxs_;
    }

    /**
     * \brief get the index set $R_k$, but mapped to their respective local
     *        index space.
     *
     * The indices are in the same way as get_remote_global_idxs.
     *
     * The set $R_k = \hat{I}_k \setminus I_k$ can also be written as the union
     * of the intersection of $\hat{I}_k$ with other disjoint sets
     * $I_l, l \neq k$, i.e.
     * $R_k = \bigcup_{j \neq k} \hat{I}_k \cap I_j = \bigcup_{j \neq k}
     * R_{k,j}$. The $R_{k,j}$ can then be mapped by $l_j$ to get the local
     * indices wrt. part $j$.
     */
    const collection::array<LocalIndexType>& get_remote_local_idxs() const
    {
        return remote_local_idxs_;
    }

    /**
     * \brief get the locally owned indices that are accessed by other ranks.
     *
     * This is the set $S_k \subset I_k$, with
     * $S_k = \bigcup{j \neq k} \hat{I}_j \cap I_k$, which contains the
     * global indices of this rank that are also accessed by other ranks.
     * For the returned set, the indices are mapped by $l_k$.
     */
    const collection::array<LocalIndexType>& get_local_shared_idxs() const
    {
        return local_idxs_;
    }

    /**
     * \brief get the rank ids which contain indices accesses by this rank.
     *
     * The order matches the order of the sets in get_remote_global_idxs and
     * get_remote_local_idxs.
     */
    const array<comm_index_type>& get_recv_target_ids() const
    {
        return recv_target_ids_;
    }

    /**
     * \brief get the rank ids which contain indices accesses by other ranks.
     *
     * The order matches the order of the sets in get_local_shared_idxs.
     */
    const array<comm_index_type>& get_send_target_ids() const
    {
        return send_target_ids_;
    }

    /**
     * \brief get the associated executor.
     */
    [[nodiscard]] std::shared_ptr<const Executor> get_executor() const
    {
        return exec_;
    }

private:
    std::shared_ptr<const Executor> exec_;
    std::shared_ptr<const part_type> partition_;
    comm_index_type rank_;

    array<comm_index_type> recv_target_ids_;
    collection::array<LocalIndexType> remote_local_idxs_;
    collection::array<GlobalIndexType> remote_global_idxs_;
    array<LocalIndexType> recv_set_offsets_;
    array<comm_index_type> send_target_ids_;
    collection::array<LocalIndexType> local_idxs_;
};


}  // namespace gko::experimental::distributed


#endif  // GKO_PUBLIC_CORE_DISTRIBUTED_INDEX_MAP_HPP_
