// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_DISTRIBUTED_INDEX_MAP_HPP_
#define GKO_PUBLIC_CORE_DISTRIBUTED_INDEX_MAP_HPP_


#include <ginkgo/core/base/segmented_array.hpp>
#include <ginkgo/core/distributed/partition.hpp>


namespace gko {
namespace experimental {
namespace distributed {


/**
 * \brief Index space classification for the locally stored indices.
 *
 * The definitions of the enum values is clarified in @ref index_map.
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
 * subset an extended global index set $\hat{I}_k \supset I_K$. The extended
 * index set contains the global indices owned by part $k$, as well as
 * remote indices $R_k = \hat{I}_k \setminus I_k$, which are also accessed by
 * part $k$, but owned by parts $l \neq k$.
 * At the core, this class provides mappings from the global index space $I$
 * into different local index spaces. The combined local index space
 * (index_space::combined) is then defined as
 * $[0, \dots, |\hat{I}_k|)$. Additionally, the combined index space can be
 * separated into locally owned (index_space::local) and non-locally owned
 * (index_space::non_local). The locally owned indices are defined as
 * $[0, \dots, |I_k|)$, and the non-locally owned as $[0, \dots, |R_k|)$.
 * With these index sets, the following mappings are defined:
 *
 * - $c_k : \hat{I}_k \mapsto [0, \dots, |\hat{I}_k|)$ which maps global indices
 *   into the combined/full local index space (denoted as
 *   index_space::combined),
 * - $l_k: I_k \mapsto [0, \dots, |I_k|)$ which maps global indices into the
 *   locally owned index space (denoted as index_space::local),
 * - $r_k: R_k \mapsto [0, \dots, |R_k|)$ which maps global indices into the
 *   non-locally owned index space (denoted as index_space::non_local).
 *
 * The required map can be selected by passing the appropriate type of an
 * index_space.
 *
 * The index map for $I_k$ has no knowledge about any other index maps for
 * $I_l, l \neq k$. In particular, any global index passed to the `map_to_local`
 * map that is not part of the specified index space, will be mapped to an
 * invalid_index.
 *
 * \tparam LocalIndexType  type for local indices
 * \tparam GlobalIndexType  type for global indices
 */
template <typename LocalIndexType, typename GlobalIndexType = int64>
struct index_map {
    using partition_type = Partition<LocalIndexType, GlobalIndexType>;

    /**
     * \brief Maps global indices to local indices
     *
     * \param global_ids  the global indices to map
     * \param index_space_v  the index space in which the returned local indices
     *                       are defined
     *
     * \return  the mapped local indices. Any global index that is not in the
     *          specified index space is mapped to invalid_index.
     */
    array<LocalIndexType> map_to_local(const array<GlobalIndexType>& global_ids,
                                       index_space index_space_v) const;

    /**
     * \brief get size of index_space::local
     */
    size_type get_global_size() const;

    /**
     * \brief get size of index_space::local
     */
    size_type get_local_size() const;

    /**
     * \brief get size of index_space::non_local
     */
    size_type get_non_local_size() const;

    /**
     * \brief Creates a new index map.
     *
     * The passed in recv_connections may contain duplicates, which will be
     * filtered out.
     *
     * \param exec  the executor
     * \param partition  the partition of the global index set
     * \param rank  the id of the global index space subset
     * \param recv_connections  the global indices that are not owned by this
     *                          rank, but accessed by it
     */
    index_map(std::shared_ptr<const Executor> exec,
              std::shared_ptr<const partition_type> partition,
              comm_index_type rank,
              const array<GlobalIndexType>& recv_connections);

    /**
     * \brief Creates an empty index map.
     */
    index_map(std::shared_ptr<const Executor> exec);

    /**
     * \brief get the index set $R_k$ for this rank.
     *
     * The indices are ordered by their owning rank and global index.
     */
    const segmented_array<GlobalIndexType>& get_remote_global_idxs() const;

    /**
     * \brief get the index set $R_k$, but mapped to their respective local
     *        index space.
     *
     * The indices are grouped by their owning rank and sorted according to
     * their global index within each group.
     *
     * The set $R_k = \hat{I}_k \setminus I_k$ can also be written as the union
     * of the intersection of $\hat{I}_k$ with other disjoint sets
     * $I_l, l \neq k$, i.e.
     * $R_k = \bigcup_{j \neq k} \hat{I}_k \cap I_j = \bigcup_{j \neq k}
     * R_{k,j}$. The set $R_{k,j}$ can then be mapped by $l_j$ to get the local
     * indices wrt. part $j$. The indices here are mapped by $l_j$.
     */
    const segmented_array<LocalIndexType>& get_remote_local_idxs() const;

    /**
     * \brief get the rank ids which contain indices accessed by this rank.
     *
     * The order matches the order of the sets in get_remote_global_idxs and
     * get_remote_local_idxs.
     */
    const array<comm_index_type>& get_remote_target_ids() const;

    /**
     * \brief get the associated executor.
     */
    std::shared_ptr<const Executor> get_executor() const;

    index_map(const index_map& other);

    index_map(index_map&& other) noexcept;

    index_map& operator=(const index_map& other);

    index_map& operator=(index_map&& other);

private:
    std::shared_ptr<const Executor> exec_;
    std::shared_ptr<const partition_type> partition_;
    comm_index_type rank_;

    array<comm_index_type> remote_target_ids_;
    segmented_array<LocalIndexType> remote_local_idxs_;
    segmented_array<GlobalIndexType> remote_global_idxs_;
};


}  // namespace distributed
}  // namespace experimental
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_DISTRIBUTED_INDEX_MAP_HPP_
