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


namespace gko {
namespace experimental::distributed {
namespace constraints {


template <typename, typename = void>
struct collection_has_get_min : std::false_type {};

template <typename T>
struct collection_has_get_min<
    T, std::void_t<decltype(::gko::collection::get_min(std::declval<T>()))>>
    : std::true_type {};

template <typename T>
constexpr bool collection_has_get_min_v = collection_has_get_min<T>::value;


template <typename, typename = void>
struct collection_has_get_max : std::false_type {};

template <typename T>
struct collection_has_get_max<
    T, std::void_t<decltype(::gko::collection::get_max(std::declval<T>()))>>
    : std::true_type {};

template <typename T>
constexpr bool collection_has_get_max_v = collection_has_get_max<T>::value;


template <typename, typename = void>
struct collection_has_get_size : std::false_type {};

template <typename T>
struct collection_has_get_size<
    T, std::void_t<decltype(::gko::collection::get_size(std::declval<T>()))>>
    : std::true_type {};

template <typename T>
constexpr bool collection_has_get_size_v = collection_has_get_size<T>::value;


}  // namespace constraints


/**
 * A representation of indices that are shared with other processes.
 *
 * The indices are grouped by the id of the shared process. The index
 * group can have two different formats, which is the same for all groups:
 * - blocked: the indices are a contiguous block, represented by a span.
 * - interleaved: the indices are not contiguous, represented by an index_set.
 *
 * Blocked indices have to start after a specified interval of local indices.
 * There is no such restriction for interleaved indices.
 *
 * @tparam IndexType  the type of the indices.
 */
template <typename IndexType, template <class> class StorageMap>
struct remote_indices_pair {
    static_assert(constraints::collection_has_get_min_v<StorageMap<IndexType>>,
                  "StorageMap type needs to provide an overload for "
                  "`gko::collection::get_min`.");
    static_assert(constraints::collection_has_get_max_v<StorageMap<IndexType>>,
                  "StorageMap type needs to provide an overload for "
                  "`gko::collection::get_max`.");
    static_assert(constraints::collection_has_get_size_v<StorageMap<IndexType>>,
                  "StorageMap type needs to provide an overload for "
                  "`gko::collection::get_size`.");

    using index_storage_type = StorageMap<IndexType>;

    remote_indices_pair(array<comm_index_type> target_ids_,
                        StorageMap<IndexType> idxs_)
        : target_ids(std::move(target_ids_)),
          idxs(std::move(idxs_)),
          begin(idxs.empty() ? 0 : collection::get_min(idxs)),
          end(idxs.empty() ? 0 : collection::get_max(idxs))
    {
        GKO_THROW_IF_INVALID(target_ids.get_size() == idxs.size(),
                             "target_ids and idxs need to have the same size");
    }

    array<comm_index_type> target_ids;
    index_storage_type idxs;
    IndexType begin;
    IndexType end;
};


/**
 * A partition of the local interval [0, n) into different classes of index
 * ownership.
 *
 * The following classes are used:
 * - receive indices: these indices are owned by other processes and not by this
 * process.
 * - send indices: these indices are owned by this process and needed by other
 * processes (which might also own them).
 * - local indices: these indices are owned by this process.
 * Sets from the indices of the different classes need not be disjoint. The send
 * and local indices are explicitly not disjoint. Receive and send indices may
 * also overlap.
 *
 * Provides optimized storage formats for different scenarios through the type
 * overlap_indices.
 *
 * @note Future work: support index weights to account for non-exclusive
 * ownership.
 *
 * @tparam IndexType  the type of the indices.
 */
template <typename IndexType = int32>
class localized_partition {
public:
    using index_type = IndexType;
    using send_indices_type = remote_indices_pair<IndexType, collection::array>;
    using recv_indices_type = remote_indices_pair<IndexType, collection::span>;

    size_type get_local_end() const { return local_end_; }

    const send_indices_type& get_send_indices() const
    {
        return overlap_send_idxs_;
    }

    const recv_indices_type& get_recv_indices() const
    {
        return overlap_recv_idxs_;
    }

    /**
     * The end sentinel of the partition.
     * @return
     */
    size_type get_end() const
    {
        return std::max(local_end_,
                        static_cast<size_type>(std::max(
                            overlap_send_idxs_.end, overlap_recv_idxs_.end)));
    }

    std::shared_ptr<const Executor> get_executor() const { return exec_; }

    /*
     * Creates an overlapping partition where the receiving indices are
     * blocked at the end.
     *
     * The partition covers the interval `[0, n)` where `n = local_size +
     * sum(target_sizes)`. The local indices are the interval `[0, local_size)`,
     * and the recv indices are the interval `[local_size, sum(target_sizes))`.
     * The last interval is composed of the sub-intervals ´[local_size,
     * local_size + target_sizes[0])`,
     * `[local_size + target_sizes[0], local_size + target_sizes[0] +
     * target_sizes[1])`, etc. The process-id for each group is denoted in
     * target_ids.
     *
     * The send indices are not blocked, so they have to be specified as a
     * vector of index_set and process-id pairs.
     *
     * Example:
     * ```c++
     * size_type local_size = 6;
     * std::vector<...> send_idxs{
     *   std::make_pair(index_set(exec, {1, 2}), 2),
     *   std::make_pair(index_set(exec, {2, 3, 4}), 1)};
     * array<comm_index_type> target_ids{exec, {1, 2}};
     * array<size_type> target_sizes{exec, {3, 2}};
     *
     * auto part = overlapping_partition<>::build_from_blocked_recv(
     *   exec, local_size, send_idxs,
     *   target_ids, target_sizes);
     *
     * // resulting indices:
     * // partition = [0, 11);
     * // recv_idxs = [6, 9) /1/, [9, 11) /2/
     * // send_idxs = {1, 2} /2/, {2, 3, 4} /1/
     * ```
     */
    static std::shared_ptr<localized_partition> build_from_blocked_recv(
        std::shared_ptr<const Executor> exec, size_type local_size,
        const std::vector<std::pair<array<index_type>, comm_index_type>>&
            send_idxs,
        const array<comm_index_type>& recv_ids,
        const std::vector<comm_index_type>& recv_sizes);

    static std::shared_ptr<localized_partition> build_from_remote_send_indices(
        std::shared_ptr<const Executor> exec, mpi::communicator comm,
        size_type local_size, const array<comm_index_type>& recv_ids,
        const std::vector<comm_index_type>& recv_sizes,
        const array<IndexType>& remote_send_indices);

private:
    localized_partition(std::shared_ptr<const Executor> exec,
                        size_type local_size,
                        send_indices_type overlap_send_idxs,
                        recv_indices_type overlap_recv_idxs)
        : exec_(exec),
          local_end_(local_size),
          overlap_send_idxs_(std::move(overlap_send_idxs)),
          overlap_recv_idxs_(std::move(overlap_recv_idxs))
    {}

    std::shared_ptr<const Executor> exec_;

    // owned by this process, interval [0, local_size_) (exclusively or shared)
    size_type local_end_;
    // owned by this and used by other processes (subset of local_idxs_)
    send_indices_type overlap_send_idxs_;
    // owned by other processes (doesn't exclude this also owning them)
    recv_indices_type overlap_recv_idxs_;
};


template <typename IndexType>
struct semi_global_index {
    comm_index_type proc_id;
    IndexType local_idx;
};


template <typename LocalIndexType, typename GlobalIndexType = int64>
struct index_map {
    using part_type = Partition<LocalIndexType, GlobalIndexType>;

    semi_global_index<LocalIndexType> get_semi_global(GlobalIndexType id) const;

    LocalIndexType get_local(comm_index_type process_id,
                             LocalIndexType semi_global_id) const;

    array<LocalIndexType> get_local(
        comm_index_type process_id,
        const array<LocalIndexType>& semi_global_ids) const;

    array<LocalIndexType> get_local(
        const array<comm_index_type>& process_ids,
        const collection::array<LocalIndexType>& semi_global_ids) const;

    array<LocalIndexType> get_local(const GlobalIndexType global_ids) const;

    array<LocalIndexType> get_local(
        const array<GlobalIndexType>& global_ids) const GKO_NOT_IMPLEMENTED;

    index_map(std::shared_ptr<const Executor> exec,
              std::shared_ptr<const part_type> part,
              const array<GlobalIndexType>& recv_connections,
              const collection::array<LocalIndexType>& send_connections);

    index_map() = default;

    const collection::array<GlobalIndexType>& get_remote_global_idxs() const
    {
        return remote_global_idxs_;
    }

    const collection::array<LocalIndexType>& get_remote_local_idxs() const
    {
        return remote_local_idxs_;
    }

private:
    std::shared_ptr<const Executor> exec_;
    std::shared_ptr<const part_type> partition_;
    comm_index_type rank_;

    array<comm_index_type> target_ids_;
    collection::array<LocalIndexType>
        remote_local_idxs_;  // need to find more general names
    collection::array<GlobalIndexType>
        remote_global_idxs_;  // need to find more general names
    collection::array<LocalIndexType> local_idxs_;
    std::vector<comm_index_type> id_set_size_;
    std::vector<LocalIndexType> id_set_offsets_;
};


/**
 * Get all rows of the input vector that are local to this process.
 */
template <typename ValueType, typename IndexType>
std::unique_ptr<gko::matrix::Dense<ValueType>> get_local(
    gko::matrix::Dense<ValueType>* vector,
    const localized_partition<IndexType>* part)
{
    GKO_ASSERT(vector->get_size()[0] == part->get_end());
    return vector->create_submatrix(span{0, part->get_local_end()},
                                    span{0, vector->get_size()[1]});
}


/**
 * Get all rows of the input vector that are not local to this process.
 */
template <typename ValueType, typename IndexType>
std::unique_ptr<gko::matrix::Dense<ValueType>> get_non_local(
    gko::matrix::Dense<ValueType>* vector,
    const localized_partition<IndexType>* part)
{
    GKO_ASSERT(vector->get_size()[0] == part->get_end());
    return vector->create_submatrix(
        span{part->get_local_end(), vector->get_size()[0]},
        span{0, vector->get_size()[1]});
}


}  // namespace experimental::distributed
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_DISTRIBUTED_INDEX_MAP_HPP_
