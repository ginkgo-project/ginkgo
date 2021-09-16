/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2021, the Ginkgo authors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#include <ginkgo/core/distributed/matrix.hpp>
#include <ginkgo/core/distributed/partition.hpp>
#include <ginkgo/core/distributed/vector.hpp>


#include <numeric>


#include <ginkgo/core/base/mpi.hpp>


#include "core/distributed/partition_kernels.hpp"


namespace gko {
namespace distributed {
namespace partition {


GKO_REGISTER_OPERATION(count_ranges, partition::count_ranges);
GKO_REGISTER_OPERATION(build_from_mapping, partition::build_from_mapping);
GKO_REGISTER_OPERATION(build_from_contiguous, partition::build_from_contiguous);
GKO_REGISTER_OPERATION(build_ranks, partition::build_ranks);
GKO_REGISTER_OPERATION(is_ordered, partition::is_ordered);
GKO_REGISTER_OPERATION(build_block_gathered_permute,
                       partition::build_block_gathered_permute);


}  // namespace partition


template <typename LocalIndexType>
std::unique_ptr<Partition<LocalIndexType>>
Partition<LocalIndexType>::build_from_mapping(
    std::shared_ptr<const Executor> exec, const Array<comm_index_type>& mapping,
    comm_index_type num_parts)
{
    auto local_mapping = make_temporary_clone(exec, &mapping);
    size_type num_ranges{};
    exec->run(partition::make_count_ranges(*local_mapping.get(), num_ranges));
    auto result = Partition::create(exec, num_parts, num_ranges);
    exec->run(
        partition::make_build_from_mapping(*local_mapping.get(), result.get()));
    result->compute_range_ranks();
    result->compute_block_gather_permutation();
    return result;
}


template <typename LocalIndexType>
std::unique_ptr<Partition<LocalIndexType>>
Partition<LocalIndexType>::build_from_contiguous(
    std::shared_ptr<const Executor> exec,
    const Array<global_index_type>& ranges)
{
    auto local_ranges = make_temporary_clone(exec, &ranges);
    auto result = Partition::create(
        exec, static_cast<comm_index_type>(ranges.get_num_elems() - 1),
        ranges.get_num_elems() - 1);
    exec->run(partition::make_build_from_contiguous(*local_ranges.get(),
                                                    result.get()));
    result->compute_range_ranks();
    result->compute_block_gather_permutation();
    return result;
}


template <typename LocalIndexType>
std::unique_ptr<Partition<LocalIndexType>>
Partition<LocalIndexType>::build_from_local_range(
    std::shared_ptr<const Executor> exec, local_index_type local_start,
    local_index_type local_end, std::shared_ptr<const mpi::communicator> comm)
{
    global_index_type range[2] = {static_cast<global_index_type>(local_start),
                                  static_cast<global_index_type>(local_end)};

    // make all range_ends available on each rank
    Array<global_index_type> ranges_start_end(exec->get_master(),
                                              comm->size() * 2);
    ranges_start_end.fill(0);
    mpi::all_gather(range, 2, ranges_start_end.get_data(), 2, comm);

    // remove duplicates
    Array<global_index_type> ranges(exec->get_master(), comm->size() + 1);
    auto ranges_se_data = ranges_start_end.get_const_data();
    ranges.get_data()[0] = ranges_se_data[0];
    for (int i = 1; i < ranges_start_end.get_num_elems() - 1; i += 2) {
        GKO_ASSERT_EQ(ranges_se_data[i], ranges_se_data[i + 1]);
        ranges.get_data()[i / 2 + 1] = ranges_se_data[i];
    }
    ranges.get_data()[ranges.get_num_elems() - 1] =
        ranges_se_data[ranges_start_end.get_num_elems() - 1];

    // move data to correct executor
    ranges.set_executor(exec);

    return Partition::build_from_contiguous(exec, ranges);
}


template <typename LocalIndexType>
std::unique_ptr<Partition<LocalIndexType>>
Partition<LocalIndexType>::build_uniformly(std::shared_ptr<const Executor> exec,
                                           comm_index_type num_parts,
                                           global_index_type global_size)
{
    global_index_type size_per_part = global_size / num_parts;
    global_index_type rest = global_size - (num_parts * size_per_part);

    Array<global_index_type> ranges(exec->get_master(), num_parts + 1);
    ranges.get_data()[0] = 0;
    for (comm_index_type pid = 0; pid < num_parts; ++pid) {
        ranges.get_data()[pid + 1] =
            ranges.get_data()[pid] + size_per_part + (rest-- > 0);
    }
    ranges.set_executor(exec);

    return Partition<LocalIndexType>::build_from_contiguous(exec, ranges);
}


template <typename LocalIndexType>
void Partition<LocalIndexType>::compute_range_ranks()
{
    auto exec = offsets_.get_executor();
    exec->run(partition::make_build_ranks(
        offsets_.get_const_data(), part_ids_.get_const_data(), get_num_ranges(),
        get_num_parts(), ranks_.get_data(), part_sizes_.get_data()));
}


template <typename LocalIndexType>
void Partition<LocalIndexType>::compute_block_gather_permutation(
    const bool recompute)
{
    if (block_gather_permutation_.get_num_elems() == 0 || recompute) {
        block_gather_permutation_.resize_and_reset(this->get_size());
        block_gather_permutation_.fill(-1);
        auto exec = block_gather_permutation_.get_executor();
        exec->run(partition::make_build_block_gathered_permute(
            this, block_gather_permutation_));
    }
}


template <typename LocalIndexType>
void Partition<LocalIndexType>::validate_data() const
{
    PolymorphicObject::validate_data();
    const auto exec = this->get_executor();
    // executors
    GKO_VALIDATION_CHECK(offsets_.get_executor() == exec);
    GKO_VALIDATION_CHECK(ranks_.get_executor() == exec);
    GKO_VALIDATION_CHECK(part_sizes_.get_executor() == exec);
    GKO_VALIDATION_CHECK(part_ids_.get_executor() == exec);
    // sizes
    const auto num_ranges = this->get_num_ranges();
    const auto num_parts = part_sizes_.get_num_elems();
    GKO_VALIDATION_CHECK(num_ranges >= 0);
    GKO_VALIDATION_CHECK(ranks_.get_num_elems() == num_ranges);
    GKO_VALIDATION_CHECK(part_ids_.get_num_elems() == num_ranges);
    GKO_VALIDATION_CHECK(part_sizes_.get_num_elems() == num_parts);
    // check range offsets: non-descending starting at 0
    Array<global_index_type> host_offsets(exec->get_master(), offsets_);
    const auto host_offset_ptr = host_offsets.get_const_data();
    GKO_VALIDATION_CHECK(host_offset_ptr[0] == 0);
    GKO_VALIDATION_CHECK_NAMED(
        "offsets need to be non-descending",
        std::is_sorted(host_offset_ptr, host_offset_ptr + (num_ranges + 1)));
    // check part IDs: in range [0, num_parts)
    Array<comm_index_type> host_part_ids(exec->get_master(), part_ids_);
    const auto host_part_id_ptr = host_part_ids.get_const_data();
    GKO_VALIDATION_CHECK_NAMED(
        "part IDs need to be in range",
        std::all_of(host_part_id_ptr, host_part_id_ptr + num_ranges,
                    [&](auto id) { return id >= 0 && id < num_parts; }));
    // check ranks and part sizes
    std::vector<global_index_type> partial_part_sizes(num_parts);
    Array<local_index_type> host_ranks(exec->get_master(), ranks_);
    Array<local_index_type> host_part_sizes(exec->get_master(), part_sizes_);
    const auto host_rank_ptr = host_ranks.get_const_data();
    const auto host_part_size_ptr = host_part_sizes.get_const_data();
    for (size_type i = 0; i < num_ranges; i++) {
        const auto part = host_part_id_ptr[i];
        const auto rank = host_rank_ptr[i];
        const auto range_size = host_offset_ptr[i + 1] - host_offset_ptr[i];
        GKO_VALIDATION_CHECK_NAMED("computed and stored range ranks must match",
                                   rank == partial_part_sizes[part]);
        partial_part_sizes[part] += range_size;
    }
    GKO_VALIDATION_CHECK_NAMED(
        "computed and stored part sizes must match",
        std::equal(partial_part_sizes.begin(), partial_part_sizes.end(),
                   host_part_size_ptr));
}


#define GKO_DECLARE_PARTITION(_type) class Partition<_type>
GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_PARTITION);


struct all_to_all_pattern {
    using comm_vector = std::vector<comm_index_type>;
    comm_vector send_sizes;
    comm_vector send_offsets;
    comm_vector recv_sizes;
    comm_vector recv_offsets;
};


template <typename LocalIndexType>
all_to_all_pattern build_communication_pattern(
    std::shared_ptr<mpi::communicator> from_comm,
    std::shared_ptr<Partition<LocalIndexType>> from_part,
    std::shared_ptr<Partition<LocalIndexType>> to_part,
    global_index_type* indices, size_type n)
{
    const auto from_n_parts = from_part->get_num_parts();

    const auto* to_ranges = to_part->get_const_range_bounds();
    const auto* to_pid = to_part->get_const_part_ids();
    const auto to_n_ranges = to_part->get_num_ranges();

    using comm_vector = std::vector<comm_index_type>;
    comm_vector send_sizes(from_n_parts);
    comm_vector send_offsets(from_n_parts + 1);
    comm_vector recv_sizes(from_n_parts);
    comm_vector recv_offsets(from_n_parts + 1);

    auto find_to_range = [&](global_index_type idx) {
        auto it =
            std::upper_bound(to_ranges + 1, to_ranges + to_n_ranges + 1, idx);
        return std::distance(to_ranges + 1, it);
    };

    for (size_type i = 0; i < n; ++i) {
        auto to_range = find_to_range(indices[i]);
        auto recv_pid = to_pid[to_range];
        send_sizes[recv_pid]++;
    }
    std::partial_sum(send_sizes.cbegin(), send_sizes.cend(),
                     send_offsets.begin() + 1);

    mpi::all_to_all(send_sizes.data(), 1, recv_sizes.data(), 1, from_comm);
    std::partial_sum(recv_sizes.cbegin(), recv_sizes.cend(),
                     recv_offsets.begin() + 1);

    return all_to_all_pattern{send_sizes, send_offsets, recv_sizes,
                              recv_offsets};
}


template <typename LocalIndexType>
Repartitioner<LocalIndexType>::Repartitioner(
    std::shared_ptr<const Executor> exec,
    std::shared_ptr<mpi::communicator> from_communicator,
    std::shared_ptr<Partition<LocalIndexType>> from_partition,
    std::shared_ptr<Partition<LocalIndexType>> to_partition)
    : exec_(std::move(exec)),
      from_partition_(std::move(from_partition)),
      to_partition_(std::move(to_partition)),
      from_comm_(std::move(from_communicator)),
      to_has_data_(false)
{
    const auto* old_ranges = from_partition_->get_const_range_bounds();
    const auto* old_pid = from_partition_->get_const_part_ids();
    const auto old_n_ranges = from_partition_->get_num_ranges();
    const auto old_n_parts = from_partition_->get_num_parts();

    const auto new_n_parts = to_partition_->get_num_parts();

    GKO_ASSERT(new_n_parts <= old_n_parts);
    GKO_ASSERT(from_partition_->get_size() == to_partition_->get_size());

    const auto rank = from_comm_->rank();
    to_has_data_ = rank < new_n_parts;

    if (new_n_parts < old_n_parts) {
        to_comm_ = mpi::communicator::create(from_comm_->get(), to_has_data_,
                                             from_comm_->rank());
    } else {
        to_comm_ = from_comm_;
    }

    std::vector<global_index_type> owned_global_idxs(
        from_partition_->get_part_size(rank));
    {
        size_type local_idx = 0;
        for (size_type range_idx = 0; range_idx < old_n_ranges; ++range_idx) {
            if (old_pid[range_idx] == rank) {
                for (global_index_type global_idx = old_ranges[range_idx];
                     global_idx < old_ranges[range_idx + 1]; ++global_idx) {
                    owned_global_idxs[local_idx++] = global_idx;
                }
            }
        }
    }
    auto pattern = build_communication_pattern(
        from_comm_, from_partition_, to_partition_, owned_global_idxs.data(),
        owned_global_idxs.size());

    default_send_sizes_ =
        std::make_shared<std::vector<comm_index_type>>(pattern.send_sizes);
    default_send_offsets_ =
        std::make_shared<std::vector<comm_index_type>>(pattern.send_offsets);
    default_recv_sizes_ =
        std::make_shared<std::vector<comm_index_type>>(pattern.recv_sizes);
    default_recv_offsets_ =
        std::make_shared<std::vector<comm_index_type>>(pattern.recv_offsets);
}

template <typename LocalIndexType>
template <typename ValueType>
void Repartitioner<LocalIndexType>::gather(
    const Vector<ValueType, LocalIndexType>* from,
    Vector<ValueType, LocalIndexType>* to)
{
    if (*(from->get_communicator()) != *from_comm_ ||
        *(to->get_communicator()) != *to_comm_) {
        throw GKO_MPI_ERROR(MPI_ERR_COMM);
    }
    // Todo: figure out if necessary to test parts for equality

    if (!is_ordered(from_partition_.get()) ||
        !is_ordered(to_partition_.get())) {
        GKO_NOT_IMPLEMENTED;
    }

    const auto* send_buffer = from->get_local()->get_const_values();
    auto* recv_buffer = to->get_local()->get_values();

    std::shared_ptr<std::vector<comm_index_type>> send_sizes;
    std::shared_ptr<std::vector<comm_index_type>> send_offsets;
    std::shared_ptr<std::vector<comm_index_type>> recv_sizes;
    std::shared_ptr<std::vector<comm_index_type>> recv_offsets;

    if (from->get_size()[1] == 1) {
        send_sizes = default_send_sizes_;
        send_offsets = default_send_offsets_;
        recv_sizes = default_recv_sizes_;
        recv_offsets = default_recv_offsets_;
    } else {
        GKO_NOT_IMPLEMENTED;
    }

    mpi::all_to_all(send_buffer, send_sizes->data(), send_offsets->data(),
                    recv_buffer, recv_sizes->data(), recv_offsets->data(), 1,
                    from_comm_);

    if (!to_has_data()) {
        *to = *Vector<ValueType, LocalIndexType>::create(
            to->get_executor(), to->get_communicator(), to->get_partition());
    }
}

#define GKO_DECLARE_REPETITIONER_GATHER(_value_type, _index_type) \
    void Repartitioner<_index_type>::gather<_value_type>(         \
        const Vector<_value_type, _index_type>* from,             \
        Vector<_value_type, _index_type>* to)
#define GKO_INSTANTIATE_REPETITIONER_GATHER_FOR_GIVEN_VALUE_TYPE(_value_type) \
    GKO_DECLARE_REPETITIONER_GATHER(_value_type, int32);                      \
    template GKO_DECLARE_REPETITIONER_GATHER(_value_type, int64)
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_INSTANTIATE_REPETITIONER_GATHER_FOR_GIVEN_VALUE_TYPE);


template <typename LocalIndexType>
template <typename ValueType>
void Repartitioner<LocalIndexType>::scatter(
    const Vector<ValueType, LocalIndexType>* to,
    Vector<ValueType, LocalIndexType>* from)
{
    if (*(from->get_communicator()) != *from_comm_ ||
        *(to->get_communicator()) != *to_comm_) {
        throw GKO_MPI_ERROR(MPI_ERR_COMM);
    }

    if (!is_ordered(from_partition_.get()) ||
        !is_ordered(to_partition_.get())) {
        GKO_NOT_IMPLEMENTED;
    }

    const auto* send_buffer = to->get_local()->get_const_values();
    auto* recv_buffer = from->get_local()->get_values();

    std::shared_ptr<std::vector<comm_index_type>> send_sizes;
    std::shared_ptr<std::vector<comm_index_type>> send_offsets;
    std::shared_ptr<std::vector<comm_index_type>> recv_sizes;
    std::shared_ptr<std::vector<comm_index_type>> recv_offsets;

    if (from->get_size()[1] == 1) {
        send_sizes = default_recv_sizes_;
        send_offsets = default_recv_offsets_;
        recv_sizes = default_send_sizes_;
        recv_offsets = default_send_offsets_;
    } else {
        GKO_NOT_IMPLEMENTED;
    }

    mpi::all_to_all(send_buffer, send_sizes->data(), send_offsets->data(),
                    recv_buffer, recv_sizes->data(), recv_offsets->data(), 1,
                    from_comm_);
}

#define GKO_DECLARE_REPETITIONER_SCATTER(_value_type, _index_type) \
    void Repartitioner<_index_type>::scatter<_value_type>(         \
        const Vector<_value_type, _index_type>* to,                \
        Vector<_value_type, _index_type>* from)
#define GKO_INSTANTIATE_REPETITIONER_SCATTER_FOR_GIVEN_VALUE_TYPE(_value_type) \
    GKO_DECLARE_REPETITIONER_SCATTER(_value_type, int32);                      \
    template GKO_DECLARE_REPETITIONER_SCATTER(_value_type, int64)
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_INSTANTIATE_REPETITIONER_SCATTER_FOR_GIVEN_VALUE_TYPE);


template <typename LocalIndexType>
template <typename ValueType>
void Repartitioner<LocalIndexType>::gather(
    const Matrix<ValueType, LocalIndexType>* from,
    Matrix<ValueType, LocalIndexType>* to)
{
    if (*(from->get_communicator()) != *from_comm_ ||
        *(to->get_communicator()) != *to_comm_) {
        throw GKO_MPI_ERROR(MPI_ERR_COMM);
    }

    using md_global = matrix_data<ValueType, global_index_type>;
    md_global local_data;
    from->write_local(local_data);

    const auto cur_local_nnz = local_data.nonzeros.size();
    std::vector<global_index_type> send_rows(cur_local_nnz);
    std::vector<global_index_type> send_cols(cur_local_nnz);
    std::vector<ValueType> send_values(cur_local_nnz);
    unpack_nonzeros(local_data.nonzeros.data(), local_data.nonzeros.size(),
                    send_rows.data(), send_cols.data(), send_values.data());

    auto pattern =
        build_communication_pattern(from_comm_, from_partition_, to_partition_,
                                    send_rows.data(), send_rows.size());

    const auto new_local_nnz = pattern.recv_offsets.back();
    std::vector<global_index_type> recv_rows(new_local_nnz);
    std::vector<global_index_type> recv_cols(new_local_nnz);
    std::vector<ValueType> recv_values(new_local_nnz);

    auto communicate = [&](const auto* send_buffer, auto* recv_buffer) {
        mpi::all_to_all(send_buffer, pattern.send_sizes.data(),
                        pattern.send_offsets.data(), recv_buffer,
                        pattern.recv_sizes.data(), pattern.recv_offsets.data(),
                        1, from_comm_);
    };
    communicate(send_rows.data(), recv_rows.data());
    communicate(send_cols.data(), recv_cols.data());
    communicate(send_values.data(), recv_values.data());


    md_global new_local_data(from->get_size());
    new_local_data.nonzeros.resize(new_local_nnz);
    pack_nonzeros(recv_rows.data(), recv_cols.data(), recv_values.data(),
                  new_local_nnz, new_local_data.nonzeros.data());
    new_local_data.ensure_row_major_order();

    auto tmp =
        Matrix<ValueType, LocalIndexType>::create(to->get_executor(), to_comm_);
    if (to_has_data_) {
        tmp->read_distributed(new_local_data, to_partition_);
    }
    tmp->move_to(to);
}

#define GKO_DECLARE_REPETITIONER_GATHER_MATRIX(_value_type, _index_type) \
    void Repartitioner<_index_type>::gather<_value_type>(                \
        const Matrix<_value_type, _index_type>* to,                      \
        Matrix<_value_type, _index_type>* from)
#define GKO_INSTANTIATE_REPETITIONER_GATHER_MATRIX_FOR_GIVEN_VALUE_TYPE( \
    _value_type)                                                         \
    GKO_DECLARE_REPETITIONER_GATHER_MATRIX(_value_type, int32);          \
    template GKO_DECLARE_REPETITIONER_GATHER_MATRIX(_value_type, int64)
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_INSTANTIATE_REPETITIONER_GATHER_MATRIX_FOR_GIVEN_VALUE_TYPE);

#define GKO_DECLARE_REPARTITIONER(_type) class Repartitioner<_type>
GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_REPARTITIONER);


template <typename LocalIndexType>
bool is_connected(const Partition<LocalIndexType>* partition)
{
    return partition->get_num_parts() == partition->get_num_ranges();
}

#define GKO_DECLARE_IS_CONNECTED(_type) \
    bool is_connected(const Partition<_type>* partition)
GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_IS_CONNECTED);


template <typename LocalIndexType>
bool is_ordered(const Partition<LocalIndexType>* partition)
{
    if (is_connected(partition)) {
        auto exec = partition->get_executor();
        bool is_ordered;
        exec->run(partition::make_is_ordered(partition, &is_ordered));
        return is_ordered;
    } else {
        return false;
    }
}

#define GKO_DECLARE_IS_ORDERED(_type) \
    bool is_ordered(const Partition<_type>* partition)
GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_IS_ORDERED);


}  // namespace distributed
}  // namespace gko
