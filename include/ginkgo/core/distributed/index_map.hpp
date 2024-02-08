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


template <typename IndexType>
struct semi_global_index {
    comm_index_type proc_id;
    IndexType local_idx;
};


// template <typename LocalIndexType>
// struct localized_index_map {
//     LocalIndexType get_local(comm_index_type process_id,
//                              LocalIndexType semi_global_id) const;
//
//     array<LocalIndexType> get_local(
//         comm_index_type process_id,
//         const array<LocalIndexType>& semi_global_ids) const;
//
//     array<LocalIndexType> get_local(
//         const array<comm_index_type>& process_ids,
//         const collection::array<LocalIndexType>& semi_global_ids) const;
//
//     size_type get_local_size() const
//     {
//         return local_size_;
//     }
//
//     localized_index_map(std::shared_ptr<const Executor> exec,
//               std::shared_ptr<const part_type> part,
//               const array<GlobalIndexType>& recv_connections,
//               const collection::array<LocalIndexType>& send_connections);
//
//     localized_index_map() = default;
//
//     const collection::array<LocalIndexType>& get_remote_local_idxs() const
//     {
//         return remote_local_idxs_;
//     }
//
// private:
//     std::shared_ptr<const Executor> exec_;
//     LocalIndexType local_size_;
//
//     array<comm_index_type> target_ids_;
//     collection::array<LocalIndexType>
//         remote_local_idxs_;  // need to find more general names
//     collection::array<LocalIndexType> local_idxs_;
//     std::vector<comm_index_type> id_set_size_;
//     std::vector<LocalIndexType> id_set_offsets_;
// };


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

    [[nodiscard]] size_type get_local_size() const
    {
        return partition_->get_part_size(rank_);
    }

    [[nodiscard]] size_type get_extended_local_size() const
    {
        return get_local_size() + remote_global_idxs_.get_flat().get_size();
    }

    index_map(std::shared_ptr<const Executor> exec,
              std::shared_ptr<const part_type> part, comm_index_type rank,
              const array<GlobalIndexType>& recv_connections,
              const array<comm_index_type>& send_target_ids,
              const collection::array<LocalIndexType>& send_connections);

    index_map(std::shared_ptr<const Executor> exec, mpi::communicator comm,
              std::shared_ptr<const part_type> part,
              const array<GlobalIndexType>& recv_connections);

    index_map() = default;

    const collection::array<GlobalIndexType>& get_remote_global_idxs() const
    {
        return remote_global_idxs_;
    }

    const collection::array<LocalIndexType>& get_remote_local_idxs() const
    {
        return remote_local_idxs_;
    }

    const collection::array<LocalIndexType>& get_local_shared_idxs() const
    {
        return local_idxs_;
    }

    const array<comm_index_type>& get_recv_target_ids() const
    {
        return recv_target_ids_;
    }

    const array<comm_index_type>& get_send_target_ids() const
    {
        return send_target_ids_;
    }

    [[nodiscard]] std::shared_ptr<const Executor> get_executor() const
    {
        return exec_;
    }

private:
    std::shared_ptr<const Executor> exec_;
    std::shared_ptr<const part_type> partition_;
    comm_index_type rank_;

    array<comm_index_type> recv_target_ids_;
    collection::array<LocalIndexType>
        remote_local_idxs_;  // need to find more general names
    collection::array<GlobalIndexType>
        remote_global_idxs_;  // need to find more general names
    array<LocalIndexType> recv_set_offsets_;
    array<comm_index_type> send_target_ids_;
    collection::array<LocalIndexType> local_idxs_;
};


// /**
//  * Get all rows of the input vector that are local to this process.
//  */
// template <typename ValueType, typename IndexType>
// std::unique_ptr<gko::matrix::Dense<ValueType>> get_local(
//     gko::matrix::Dense<ValueType>* vector,
//     const localized_partition<IndexType>* part)
// {
//     GKO_ASSERT(vector->get_size()[0] == part->get_end());
//     return vector->create_submatrix(span{0, part->get_local_end()},
//                                     span{0, vector->get_size()[1]});
// }
//
//
// /**
//  * Get all rows of the input vector that are not local to this process.
//  */
// template <typename ValueType, typename IndexType>
// std::unique_ptr<gko::matrix::Dense<ValueType>> get_non_local(
//     gko::matrix::Dense<ValueType>* vector,
//     const localized_partition<IndexType>* part)
// {
//     GKO_ASSERT(vector->get_size()[0] == part->get_end());
//     return vector->create_submatrix(
//         span{part->get_local_end(), vector->get_size()[0]},
//         span{0, vector->get_size()[1]});
// }
}  // namespace experimental::distributed
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_DISTRIBUTED_INDEX_MAP_HPP_
