/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2023, the Ginkgo authors
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


#include <fstream>
#include <ginkgo/core/base/mtx_io.hpp>
#include <ginkgo/core/base/precision_dispatch.hpp>
#include <ginkgo/core/distributed/vector.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/diagonal.hpp>

#include "core/distributed/matrix_kernels.hpp"


namespace gko {
namespace experimental {
namespace distributed {
namespace matrix {
namespace {


GKO_REGISTER_OPERATION(build_local_nonlocal,
                       distributed_matrix::build_local_nonlocal);


}  // namespace
}  // namespace matrix


namespace {


template <typename LocalIndexType, typename GlobalIndexType>
inline auto find_part(std::shared_ptr<gko::experimental::distributed::Partition<
                          LocalIndexType, GlobalIndexType>>
                          partition,
                      GlobalIndexType idx)
{
    auto range_bounds = partition->get_range_bounds();
    auto range_parts = partition->get_part_ids();
    auto num_ranges = partition->get_num_ranges();

    auto it =
        std::upper_bound(range_bounds + 1, range_bounds + num_ranges + 1, idx);
    auto range = std::distance(range_bounds + 1, it);
    return range_parts[range];
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
inline auto build_send_buffer(
    std::shared_ptr<const Executor> exec, mpi::communicator comm,
    const gko::matrix_data<ValueType, GlobalIndexType>& data,
    std::shared_ptr<gko::experimental::distributed::Partition<LocalIndexType,
                                                              GlobalIndexType>>
        part)
{
    using nonzero = gko::matrix_data_entry<ValueType, GlobalIndexType>;
    auto local_part = comm.rank();

    auto partition = part;
    if (exec != exec->get_master()) {
        partition = gko::clone(exec->get_master(), part);
    }

    auto range_bounds = partition->get_range_bounds();
    auto range_parts = partition->get_part_ids();
    auto num_ranges = partition->get_num_ranges();

    auto find_part = [&](GlobalIndexType idx) {
        auto it = std::upper_bound(range_bounds + 1,
                                   range_bounds + num_ranges + 1, idx);
        auto range = std::distance(range_bounds + 1, it);
        return range_parts[range];
    };

    std::vector<
        std::pair<gko::experimental::distributed::comm_index_type, nonzero>>
        send_buffer_local;
    for (size_t i = 0; i < data.nonzeros.size(); ++i) {
        auto entry = data.nonzeros[i];
        auto p_id = find_part(entry.row);
        if (p_id != local_part && p_id >= 0 && p_id < num_ranges &&
            entry.row >= 0 && entry.row < range_bounds[num_ranges]) {
            send_buffer_local.emplace_back(p_id, entry);
        }
    }
    std::sort(std::begin(send_buffer_local), std::end(send_buffer_local),
              [](const auto& a, const auto& b) { return a.first < b.first; });

    return send_buffer_local;
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType,
          typename DataType>
inline auto build_send_pattern(
    const std::vector<std::pair<gko::experimental::distributed::comm_index_type,
                                DataType>>& send_buffer,
    std::shared_ptr<gko::experimental::distributed::Partition<LocalIndexType,
                                                              GlobalIndexType>>
        partition)
{
    auto num_parts = partition->get_num_parts();
    std::vector<comm_index_type> send_sizes(num_parts, 0);
    std::vector<comm_index_type> send_offsets(num_parts + 1, 0);
    auto i = 0;
    for (auto& [p_id, entry] : send_buffer) {
        send_sizes[p_id]++;
        i++;
    }
    std::partial_sum(std::begin(send_sizes), std::end(send_sizes),
                     std::begin(send_offsets) + 1);

    return std::make_tuple(send_sizes, send_offsets);
}


template <typename LocalIndexType, typename GlobalIndexType>
inline auto build_receive_pattern(
    std::shared_ptr<const Executor> exec, mpi::communicator comm,
    const std::vector<gko::experimental::distributed::comm_index_type>&
        send_sizes,
    std::shared_ptr<gko::experimental::distributed::Partition<LocalIndexType,
                                                              GlobalIndexType>>
        partition)
{
    auto num_parts = partition->get_num_parts();

    std::vector<comm_index_type> recv_sizes(num_parts, 0);
    std::vector<comm_index_type> recv_offsets(num_parts + 1, 0);

    comm.all_to_all(exec, send_sizes.data(), 1, recv_sizes.data(), 1);
    std::partial_sum(std::begin(recv_sizes), std::end(recv_sizes),
                     std::begin(recv_offsets) + 1);

    return std::make_tuple(recv_sizes, recv_offsets);
}


template <typename ValueType, typename IndexType>
inline auto split_nonzero_entries(
    const std::vector<std::pair<gko::experimental::distributed::comm_index_type,
                                gko::matrix_data_entry<ValueType, IndexType>>>
        send_buffer)
{
    const auto size = send_buffer.size();
    std::vector<IndexType> row_buffer(size);
    std::vector<IndexType> col_buffer(size);
    std::vector<ValueType> val_buffer(size);

    for (size_t i = 0; i < size; ++i) {
        const auto& entry = send_buffer[i].second;
        row_buffer[i] = entry.row;
        col_buffer[i] = entry.column;
        val_buffer[i] = entry.value;
    }

    return std::make_tuple(row_buffer, col_buffer, val_buffer);
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
inline void communicate_overlap(
    std::shared_ptr<const Executor> exec, mpi::communicator comm,
    gko::matrix_data<ValueType, GlobalIndexType>& data,
    std::shared_ptr<gko::experimental::distributed::Partition<LocalIndexType,
                                                              GlobalIndexType>>
        partition)
{
    auto send_buffer = build_send_buffer(exec, comm, data, partition);

    // build send pattern
    auto [send_sizes, send_offsets] =
        build_send_pattern<ValueType, LocalIndexType, GlobalIndexType>(
            send_buffer, partition);

    // build receive pattern
    auto [recv_sizes, recv_offsets] =
        build_receive_pattern<LocalIndexType, GlobalIndexType>(
            exec, comm, send_sizes, partition);

    // split nonzero entries into buffers
    auto [send_row, send_col, send_val] =
        split_nonzero_entries<ValueType, GlobalIndexType>(send_buffer);

    // communicate buffers
    const auto size_recv_entries = recv_offsets.back();
    std::vector<GlobalIndexType> recv_row(size_recv_entries);
    std::vector<GlobalIndexType> recv_col(size_recv_entries);
    std::vector<ValueType> recv_val(size_recv_entries);

    auto communicate = [&](const auto* send_buffer, auto* recv_buffer) {
        comm.all_to_all_v(exec->get_master(), send_buffer, send_sizes.data(),
                          send_offsets.data(), recv_buffer, recv_sizes.data(),
                          recv_offsets.data());
    };

    communicate(send_row.data(), recv_row.data());
    communicate(send_col.data(), recv_col.data());
    communicate(send_val.data(), recv_val.data());

    // add new entries
    for (size_t i = 0; i < size_recv_entries; ++i) {
        data.nonzeros.emplace_back(recv_row[i], recv_col[i], recv_val[i]);
    }

    data.sum_duplicates();
}


}  // namespace


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
Matrix<ValueType, LocalIndexType, GlobalIndexType>::Matrix(
    std::shared_ptr<const Executor> exec, mpi::communicator comm)
    : Matrix(exec, comm, with_matrix_type<gko::matrix::Csr>())
{}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
Matrix<ValueType, LocalIndexType, GlobalIndexType>::Matrix(
    std::shared_ptr<const Executor> exec, mpi::communicator comm,
    const LinOp* local_matrix_type)
    : Matrix(exec, comm, local_matrix_type, local_matrix_type)
{}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
Matrix<ValueType, LocalIndexType, GlobalIndexType>::Matrix(
    std::shared_ptr<const Executor> exec, mpi::communicator comm,
    const LinOp* local_matrix_template, const LinOp* non_local_matrix_template)
    : EnableDistributedLinOp<
          Matrix<value_type, local_index_type, global_index_type>>{exec},
      DistributedBase{comm},
      send_offsets_(comm.size() + 1),
      send_sizes_(comm.size()),
      recv_offsets_(comm.size() + 1),
      recv_sizes_(comm.size()),
      gather_idxs_{exec},
      non_local_to_global_{exec},
      one_scalar_{},
      row_partition_{part_type::create(exec)},
      col_partition_{part_type::create(exec)},
      matrix_data_{exec},
      local_mtx_{local_matrix_template->clone(exec)},
      non_local_mtx_{non_local_matrix_template->clone(exec)}
{
    GKO_ASSERT(
        (dynamic_cast<ReadableFromMatrixData<ValueType, LocalIndexType>*>(
            local_mtx_.get())));
    GKO_ASSERT(
        (dynamic_cast<ReadableFromMatrixData<ValueType, LocalIndexType>*>(
            non_local_mtx_.get())));
    one_scalar_.init(exec, dim<2>{1, 1});
    one_scalar_->fill(one<value_type>());
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
void Matrix<ValueType, LocalIndexType, GlobalIndexType>::convert_to(
    Matrix<next_precision<value_type>, local_index_type, global_index_type>*
        result) const
{
    GKO_ASSERT(this->get_communicator().size() ==
               result->get_communicator().size());
    result->local_mtx_->copy_from(this->local_mtx_.get());
    result->non_local_mtx_->copy_from(this->non_local_mtx_.get());
    result->gather_idxs_ = this->gather_idxs_;
    result->send_offsets_ = this->send_offsets_;
    result->recv_offsets_ = this->recv_offsets_;
    result->recv_sizes_ = this->recv_sizes_;
    result->send_sizes_ = this->send_sizes_;
    // FIXME Add mixed prec copies to device_matrix_data
    // result->matrix_data_ = this->matrix_data_;
    result->row_partition_ = this->row_partition_;
    result->col_partition_ = this->col_partition_;
    result->non_local_to_global_ = this->non_local_to_global_;
    result->set_size(this->get_size());
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
void Matrix<ValueType, LocalIndexType, GlobalIndexType>::move_to(
    Matrix<next_precision<value_type>, local_index_type, global_index_type>*
        result)
{
    GKO_ASSERT(this->get_communicator().size() ==
               result->get_communicator().size());
    result->local_mtx_->move_from(this->local_mtx_.get());
    result->non_local_mtx_->move_from(this->non_local_mtx_.get());
    result->gather_idxs_ = std::move(this->gather_idxs_);
    result->send_offsets_ = std::move(this->send_offsets_);
    result->recv_offsets_ = std::move(this->recv_offsets_);
    result->recv_sizes_ = std::move(this->recv_sizes_);
    result->send_sizes_ = std::move(this->send_sizes_);
    // FIXME Add mixed prec copies to device_matrix_data
    // result->matrix_data_ = std::move(this->matrix_data_);
    result->row_partition_ = std::move(this->row_partition_);
    result->col_partition_ = std::move(this->col_partition_);
    result->non_local_to_global_ = std::move(this->non_local_to_global_);
    result->set_size(this->get_size());
    this->set_size({});
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
std::unique_ptr<gko::matrix::Diagonal<ValueType>>
Matrix<ValueType, LocalIndexType, GlobalIndexType>::extract_diagonal() const
{
    return gko::as<DiagonalExtractable<ValueType>>(this->get_local_matrix())
        ->extract_diagonal();
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
void Matrix<ValueType, LocalIndexType, GlobalIndexType>::read_distributed(
    const device_matrix_data<value_type, global_index_type>& data,
    const Partition<local_index_type, global_index_type>* row_partition,
    const Partition<local_index_type, global_index_type>* col_partition,
    bool add)
{
    using part_type =
        gko::experimental::distributed::Partition<local_index_type,
                                                  global_index_type>;
    const auto comm = this->get_communicator();
    GKO_ASSERT_EQ(data.get_size()[0], row_partition->get_size());
    GKO_ASSERT_EQ(data.get_size()[1], col_partition->get_size());
    GKO_ASSERT_EQ(comm.size(), row_partition->get_num_parts());
    GKO_ASSERT_EQ(comm.size(), col_partition->get_num_parts());
    auto exec = this->get_executor();
    auto local_part = comm.rank();
    this->row_partition_->copy_from(row_partition);
    this->col_partition_->copy_from(col_partition);
    this->matrix_data_ = data;

    auto dev_data = data;
    if (add) {
        auto host_data = this->matrix_data_.copy_to_host();
        communicate_overlap(exec, comm, host_data, this->row_partition_);
        dev_data =
            device_matrix_data<value_type, global_index_type>::create_from_host(
                this->get_executor(), host_data);
    }

    // set up LinOp sizes
    auto num_parts = static_cast<size_type>(row_partition->get_num_parts());
    auto global_num_rows = this->get_row_partition()->get_size();
    auto global_num_cols = col_partition->get_size();
    dim<2> global_dim{global_num_rows, global_num_cols};
    this->set_size(global_dim);

    // temporary storage for the output
    array<local_index_type> local_row_idxs{exec};
    array<local_index_type> local_col_idxs{exec};
    array<value_type> local_values{exec};
    array<local_index_type> non_local_row_idxs{exec};
    array<local_index_type> non_local_col_idxs{exec};
    array<value_type> non_local_values{exec};
    array<local_index_type> recv_gather_idxs{exec};
    array<comm_index_type> recv_sizes_array{exec, num_parts};

    // build local, non-local matrix data and communication structures
    exec->run(matrix::make_build_local_nonlocal(
        dev_data, make_temporary_clone(exec, row_partition).get(),
        make_temporary_clone(exec, col_partition).get(), local_part,
        local_row_idxs, local_col_idxs, local_values, non_local_row_idxs,
        non_local_col_idxs, non_local_values, recv_gather_idxs,
        recv_sizes_array, non_local_to_global_));

    // read the local matrix data
    const auto num_local_rows =
        static_cast<size_type>(row_partition->get_part_size(local_part));
    const auto num_local_cols =
        static_cast<size_type>(col_partition->get_part_size(local_part));
    const auto num_non_local_cols = non_local_to_global_.get_num_elems();
    device_matrix_data<value_type, local_index_type> local_data{
        exec, dim<2>{num_local_rows, num_local_cols}, std::move(local_row_idxs),
        std::move(local_col_idxs), std::move(local_values)};
    device_matrix_data<value_type, local_index_type> non_local_data{
        exec, dim<2>{num_local_rows, num_non_local_cols},
        std::move(non_local_row_idxs), std::move(non_local_col_idxs),
        std::move(non_local_values)};
    as<ReadableFromMatrixData<ValueType, LocalIndexType>>(this->local_mtx_)
        ->read(std::move(local_data));
    as<ReadableFromMatrixData<ValueType, LocalIndexType>>(this->non_local_mtx_)
        ->read(std::move(non_local_data));

    // exchange step 1: determine recv_sizes, send_sizes, send_offsets
    exec->get_master()->copy_from(exec.get(), num_parts,
                                  recv_sizes_array.get_const_data(),
                                  recv_sizes_.data());
    std::partial_sum(recv_sizes_.begin(), recv_sizes_.end(),
                     recv_offsets_.begin() + 1);
    comm.all_to_all(exec, recv_sizes_.data(), 1, send_sizes_.data(), 1);
    std::partial_sum(send_sizes_.begin(), send_sizes_.end(),
                     send_offsets_.begin() + 1);
    send_offsets_[0] = 0;
    recv_offsets_[0] = 0;

    // exchange step 2: exchange gather_idxs from receivers to senders
    auto use_host_buffer = mpi::requires_host_buffer(exec, comm);
    if (use_host_buffer) {
        recv_gather_idxs.set_executor(exec->get_master());
        gather_idxs_.clear();
        gather_idxs_.set_executor(exec->get_master());
    }
    gather_idxs_.resize_and_reset(send_offsets_.back());
    comm.all_to_all_v(use_host_buffer ? exec->get_master() : exec,
                      recv_gather_idxs.get_const_data(), recv_sizes_.data(),
                      recv_offsets_.data(), gather_idxs_.get_data(),
                      send_sizes_.data(), send_offsets_.data());
    if (use_host_buffer) {
        gather_idxs_.set_executor(exec);
    }
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
void Matrix<ValueType, LocalIndexType, GlobalIndexType>::read_distributed(
    const matrix_data<value_type, global_index_type>& data,
    const Partition<local_index_type, global_index_type>* row_partition,
    const Partition<local_index_type, global_index_type>* col_partition,
    bool add)
{
    this->read_distributed(
        device_matrix_data<value_type, global_index_type>::create_from_host(
            this->get_executor(), data),
        row_partition, col_partition, add);
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
void Matrix<ValueType, LocalIndexType, GlobalIndexType>::read_distributed(
    const matrix_data<ValueType, global_index_type>& data,
    const Partition<local_index_type, global_index_type>* partition, bool add)
{
    this->read_distributed(
        device_matrix_data<value_type, global_index_type>::create_from_host(
            this->get_executor(), data),
        partition, partition, add);
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
void Matrix<ValueType, LocalIndexType, GlobalIndexType>::read_distributed(
    const device_matrix_data<ValueType, GlobalIndexType>& data,
    const Partition<local_index_type, global_index_type>* partition, bool add)
{
    this->read_distributed(data, partition, partition, add);
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
mpi::request Matrix<ValueType, LocalIndexType, GlobalIndexType>::communicate(
    const local_vector_type* local_b) const
{
    auto exec = this->get_executor();
    const auto comm = this->get_communicator();
    auto num_cols = local_b->get_size()[1];
    auto send_size = send_offsets_.back();
    auto recv_size = recv_offsets_.back();
    auto send_dim = dim<2>{static_cast<size_type>(send_size), num_cols};
    auto recv_dim = dim<2>{static_cast<size_type>(recv_size), num_cols};
    recv_buffer_.init(exec, recv_dim);
    send_buffer_.init(exec, send_dim);

    local_b->row_gather(&gather_idxs_, send_buffer_.get());

    auto use_host_buffer = mpi::requires_host_buffer(exec, comm);
    if (use_host_buffer) {
        host_recv_buffer_.init(exec->get_master(), recv_dim);
        host_send_buffer_.init(exec->get_master(), send_dim);
        host_send_buffer_->copy_from(send_buffer_.get());
    }

    mpi::contiguous_type type(num_cols, mpi::type_impl<ValueType>::get_type());
    auto send_ptr = use_host_buffer ? host_send_buffer_->get_const_values()
                                    : send_buffer_->get_const_values();
    auto recv_ptr = use_host_buffer ? host_recv_buffer_->get_values()
                                    : recv_buffer_->get_values();
    exec->synchronize();
#ifdef GINKGO_FORCE_SPMV_BLOCKING_COMM
    comm.all_to_all_v(use_host_buffer ? exec->get_master() : exec, send_ptr,
                      send_sizes_.data(), send_offsets_.data(), type.get(),
                      recv_ptr, recv_sizes_.data(), recv_offsets_.data(),
                      type.get());
    return {};
#else
    return comm.i_all_to_all_v(
        use_host_buffer ? exec->get_master() : exec, send_ptr,
        send_sizes_.data(), send_offsets_.data(), type.get(), recv_ptr,
        recv_sizes_.data(), recv_offsets_.data(), type.get());
#endif
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
void Matrix<ValueType, LocalIndexType, GlobalIndexType>::apply_impl(
    const LinOp* b, LinOp* x) const
{
    distributed::precision_dispatch_real_complex<ValueType>(
        [this](const auto dense_b, auto dense_x) {
            auto x_exec = dense_x->get_executor();
            auto local_x = gko::matrix::Dense<ValueType>::create(
                x_exec, dense_x->get_local_vector()->get_size(),
                gko::make_array_view(
                    x_exec,
                    dense_x->get_local_vector()->get_num_stored_elements(),
                    dense_x->get_local_values()),
                dense_x->get_local_vector()->get_stride());

            auto comm = this->get_communicator();
            auto req = this->communicate(dense_b->get_local_vector());
            local_mtx_->apply(dense_b->get_local_vector(), local_x.get());
            req.wait();

            auto exec = this->get_executor();
            auto use_host_buffer = mpi::requires_host_buffer(exec, comm);
            if (use_host_buffer) {
                recv_buffer_->copy_from(host_recv_buffer_.get());
            }
            non_local_mtx_->apply(one_scalar_.get(), recv_buffer_.get(),
                                  one_scalar_.get(), local_x.get());
        },
        b, x);
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
void Matrix<ValueType, LocalIndexType, GlobalIndexType>::apply_impl(
    const LinOp* alpha, const LinOp* b, const LinOp* beta, LinOp* x) const
{
    distributed::precision_dispatch_real_complex<ValueType>(
        [this](const auto local_alpha, const auto dense_b,
               const auto local_beta, auto dense_x) {
            const auto x_exec = dense_x->get_executor();
            auto local_x = gko::matrix::Dense<ValueType>::create(
                x_exec, dense_x->get_local_vector()->get_size(),
                gko::make_array_view(
                    x_exec,
                    dense_x->get_local_vector()->get_num_stored_elements(),
                    dense_x->get_local_values()),
                dense_x->get_local_vector()->get_stride());

            auto comm = this->get_communicator();
            auto req = this->communicate(dense_b->get_local_vector());
            local_mtx_->apply(local_alpha, dense_b->get_local_vector(),
                              local_beta, local_x.get());
            req.wait();

            auto exec = this->get_executor();
            auto use_host_buffer = mpi::requires_host_buffer(exec, comm);
            if (use_host_buffer) {
                recv_buffer_->copy_from(host_recv_buffer_.get());
            }
            non_local_mtx_->apply(local_alpha, recv_buffer_.get(),
                                  one_scalar_.get(), local_x.get());
        },
        alpha, b, beta, x);
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
Matrix<ValueType, LocalIndexType, GlobalIndexType>::Matrix(const Matrix& other)
    : EnableDistributedLinOp<Matrix<value_type, local_index_type,
                                    global_index_type>>{other.get_executor()},
      DistributedBase{other.get_communicator()},
      matrix_data_{other.get_executor()}
{
    *this = other;
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
Matrix<ValueType, LocalIndexType, GlobalIndexType>::Matrix(
    Matrix&& other) noexcept
    : EnableDistributedLinOp<Matrix<value_type, local_index_type,
                                    global_index_type>>{other.get_executor()},
      DistributedBase{other.get_communicator()},
      matrix_data_{other.get_executor()}
{
    *this = std::move(other);
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
Matrix<ValueType, LocalIndexType, GlobalIndexType>&
Matrix<ValueType, LocalIndexType, GlobalIndexType>::operator=(
    const Matrix& other)
{
    if (this != &other) {
        GKO_ASSERT_EQ(other.get_communicator().size(),
                      this->get_communicator().size());
        this->set_size(other.get_size());
        local_mtx_->copy_from(other.local_mtx_.get());
        non_local_mtx_->copy_from(other.non_local_mtx_.get());
        gather_idxs_ = other.gather_idxs_;
        send_offsets_ = other.send_offsets_;
        recv_offsets_ = other.recv_offsets_;
        send_sizes_ = other.send_sizes_;
        recv_sizes_ = other.recv_sizes_;
        matrix_data_ = other.matrix_data_;
        row_partition_ = other.row_partition_;
        col_partition_ = other.col_partition_;
        non_local_to_global_ = other.non_local_to_global_;
        one_scalar_.init(this->get_executor(), dim<2>{1, 1});
        one_scalar_->fill(one<value_type>());
    }
    return *this;
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
Matrix<ValueType, LocalIndexType, GlobalIndexType>&
Matrix<ValueType, LocalIndexType, GlobalIndexType>::operator=(Matrix&& other)
{
    if (this != &other) {
        GKO_ASSERT_EQ(other.get_communicator().size(),
                      this->get_communicator().size());
        this->set_size(other.get_size());
        other.set_size({});
        local_mtx_->move_from(other.local_mtx_.get());
        non_local_mtx_->move_from(other.non_local_mtx_.get());
        gather_idxs_ = std::move(other.gather_idxs_);
        send_offsets_ = std::move(other.send_offsets_);
        recv_offsets_ = std::move(other.recv_offsets_);
        send_sizes_ = std::move(other.send_sizes_);
        recv_sizes_ = std::move(other.recv_sizes_);
        matrix_data_ = std::move(other.matrix_data_);
        row_partition_ = std::move(other.row_partition_);
        col_partition_ = std::move(other.col_partition_);
        non_local_to_global_ = std::move(other.non_local_to_global_);
        one_scalar_.init(this->get_executor(), dim<2>{1, 1});
        one_scalar_->fill(one<value_type>());
    }
    return *this;
}


#define GKO_DECLARE_DISTRIBUTED_MATRIX(ValueType, LocalIndexType, \
                                       GlobalIndexType)           \
    class Matrix<ValueType, LocalIndexType, GlobalIndexType>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_LOCAL_GLOBAL_INDEX_TYPE(
    GKO_DECLARE_DISTRIBUTED_MATRIX);


}  // namespace distributed
}  // namespace experimental
}  // namespace gko
