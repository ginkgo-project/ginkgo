// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_DISTRIBUTED_LOCALIZED_PARTITION_HPP_
#define GKO_PUBLIC_CORE_DISTRIBUTED_LOCALIZED_PARTITION_HPP_


#include <ginkgo/config.hpp>
#include <ginkgo/core/base/dense_cache.hpp>
#include <ginkgo/core/base/index_set.hpp>
#include <ginkgo/core/distributed/base.hpp>
#include <ginkgo/core/distributed/lin_op.hpp>
#include <ginkgo/core/matrix/dense.hpp>


namespace gko::experimental::distributed {
namespace detail {


template <typename T>
struct template_span {
    /**
     * Creates a span representing a point `point`.
     *
     * The `begin` of this span is set to `point`, and the `end` to `point + 1`.
     *
     * @param point  the point which the span represents
     */
    GKO_ATTRIBUTES constexpr template_span(T point) noexcept
        : template_span{point, point + 1}
    {}

    GKO_ATTRIBUTES constexpr template_span(T begin, T end) noexcept
        : begin(begin), end(end)
    {}
    /**
     * Checks if a span is valid.
     *
     * @return true if and only if `this->begin <= this->end`
     */
    GKO_ATTRIBUTES constexpr bool is_valid() const { return begin <= end; }

    /**
     * Returns the length of a span.
     *
     * @return `this->end - this->begin`
     */
    GKO_ATTRIBUTES constexpr size_type length() const { return end - begin; }

    /**
     * Beginning of the span.
     */
    T begin;

    /**
     * End of the span.
     */
    T end;
};
}  // namespace detail


template <typename T>
using span_collection = std::vector<detail::template_span<T>>;


template <typename T>
struct array_collection {
    using iterator = array<T>*;
    using const_iterator = const array<T>*;

    template <typename SizeType>
    array_collection(std::shared_ptr<const Executor> exec,
                     const std::vector<SizeType>& sizes)
        : array_collection(
              array<T>(exec, std::accumulate(sizes.begin(), sizes.end(), 0)),
              sizes)
    {}

    template <typename SizeType>
    array_collection(array<T> buffer, const std::vector<SizeType>& sizes)
        : buffer_(std::move(buffer))
    {
        auto exec = buffer_.get_executor();
        std::vector<size_type> offsets(sizes.size() + 1);
        std::partial_sum(sizes.begin(), sizes.end(), offsets.begin() + 1);
        buffer_.resize_and_reset(offsets.back());
        for (size_type i = 0; i < sizes.size(); ++i) {
            elems_.push_back(make_array_view(exec, sizes[i],
                                             buffer_.get_data() + offsets[i]));
        }
    }

    array<T>& operator[](size_type i) { return elems_[i]; }
    const array<T>& operator[](size_type i) const { return elems_[i]; }

    size_type size() const { return elems_.size(); }

    iterator begin() { return elems_.begin(); }
    const_iterator begin() const { return elems_.begin(); }

    iterator end() { return elems_.end(); }
    const_iterator end() const { return elems_.end(); }

    bool empty() const { return elems_.empty(); }

    array<T>& get_flat() { return buffer_; }
    const array<T>& get_flat() const { return buffer_; }

private:
    array<T> buffer_;
    std::vector<array<T>> elems_;
};


namespace detail {

template <typename IndexType>
IndexType get_min(const span_collection<IndexType>& spans)
{
    if (spans.empty()) {
        return std::numeric_limits<IndexType>::max();
    }
    return std::min_element(
               spans.begin(), spans.end(),
               [](const auto& a, const auto& b) { return a.begin < b.begin; })
        ->begin;
}


template <typename IndexType>
IndexType get_max(const span_collection<IndexType>& spans)
{
    if (spans.empty()) {
        return std::numeric_limits<IndexType>::min();
    }
    return std::max_element(
               spans.begin(), spans.end(),
               [](const auto& a, const auto& b) { return a.end < b.end; })
        ->end;
}


template <typename IndexType>
size_type get_size(const span_collection<IndexType>& spans)
{
    if (spans.empty()) {
        return 0;
    }
    return get_max(spans) - get_min(spans);
}


template <typename IndexType>
size_type get_size(const template_span<IndexType>& span)
{
    return span.length();
}


template <typename IndexType>
IndexType get_min(const array_collection<IndexType>& arrs);


template <typename IndexType>
IndexType get_max(const array_collection<IndexType>& arrs);


template <typename IndexType>
size_type get_size(const array_collection<IndexType>& arrs)
{
    return arrs.get_flat().get_size();
}


template <typename IndexType>
size_type get_size(const array<IndexType>& arr)
{
    return arr.get_size();
}

}  // namespace detail

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
    remote_indices_pair(array<comm_index_type> target_ids_,
                        StorageMap<IndexType> idxs_)
        : target_ids(std::move(target_ids_)),
          idxs(std::move(idxs_)),
          begin(idxs.empty() ? 0 : detail::get_min(idxs)),
          end(idxs.empty() ? 0 : detail::get_max(idxs))
    {
        GKO_THROW_IF_INVALID(target_ids.get_size() == idxs.size(),
                             "target_ids and idxs need to have the same size");
    }

    array<comm_index_type> target_ids;
    StorageMap<IndexType> idxs;
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
    using send_indices_type = remote_indices_pair<IndexType, array_collection>;
    using recv_indices_type = remote_indices_pair<IndexType, span_collection>;

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


// template <typename IndexType>
// IndexType compute_compressed_col(
//     const overlap_indices<array<IndexType>>& remote_send_idxs,
//     comm_index_type process_id, IndexType local_col_id)
// {
//     // this is actually the recv offset.
//     IndexType offset = 0;
//     for (comm_index_type pid = 0; pid < process_id; ++pid) {
//         offset += remote_send_idxs.get_indices(pid).get_size();
//     }
//     return offset + local_col_id;
// }
//
//
// template <typename IndexType>
// struct semi_global_index {
//     comm_index_type proc_id;
//     IndexType local_idx;
// };
//
// template <typename LocalIndexType, typename GlobalIndexType = int64>
// struct NonLocalIndexMap {
//     GlobalIndexType get_global(LocalIndexType id) GKO_NOT_IMPLEMENTED;
//
//     LocalIndexType get_local(comm_index_type process_id,
//                              LocalIndexType local_id);
//
//     array<LocalIndexType> get_local(
//         const array<comm_index_type>& process_ids,
//         const flat_container<array<comm_index_type>>& local_ids);
//
//     array<LocalIndexType> get_local();
//
//
//     std::unique_ptr<NonLocalIndexMap> create(
//         std::shared_ptr<const Executor>,
//         ptr_param<const localized_partition<LocalIndexType>> part)
//     {}
//
// private:
//     overlap_indices<array<LocalIndexType>> remote_send_idxs_;
//     overlap_indices<array<LocalIndexType>> send_idxs_;
// };


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


}  // namespace gko::experimental::distributed


#endif  // GKO_PUBLIC_CORE_DISTRIBUTED_LOCALIZED_PARTITION_HPP_
