/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2022, the Ginkgo authors
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


#include <ginkgo/core/distributed/vector.hpp>


#include "core/distributed/matrix_kernels.hpp"


namespace gko {
namespace distributed {
namespace matrix {
namespace {


GKO_REGISTER_OPERATION(build_diag_offdiag,
                       distributed_matrix::build_diag_offdiag);


}  // namespace
}  // namespace matrix


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
Matrix<ValueType, LocalIndexType, GlobalIndexType>::Matrix(
    std::shared_ptr<const Executor> exec, mpi::communicator comm)
    : EnableLinOp<
          Matrix<value_type, local_index_type, global_index_type>>{exec},
      DistributedBase{comm},
      send_offsets_(comm.size() + 1),
      send_sizes_(comm.size()),
      recv_offsets_(comm.size() + 1),
      recv_sizes_(comm.size()),
      gather_idxs_{exec},
      local_to_global_ghost_{exec},
      one_scalar_{},
      diag_mtx_{local_matrix_type::create(exec)},
      offdiag_mtx_{local_matrix_type::create(exec)}
{
    one_scalar_.init(exec, dim<2>{1, 1});
    initialize<local_vector_type>({one<value_type>()}, exec)
        ->move_to(one_scalar_.get());
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
void Matrix<ValueType, LocalIndexType, GlobalIndexType>::convert_to(
    Matrix<next_precision<value_type>, local_index_type, global_index_type>*
        result) const
{
    GKO_ASSERT(this->get_communicator().size() ==
               result->get_communicator().size());
    result->diag_mtx_->copy_from(this->diag_mtx_.get());
    result->offdiag_mtx_->copy_from(this->offdiag_mtx_.get());
    result->gather_idxs_ = this->gather_idxs_;
    result->send_offsets_ = this->send_offsets_;
    result->recv_offsets_ = this->recv_offsets_;
    result->recv_sizes_ = this->recv_sizes_;
    result->send_sizes_ = this->send_sizes_;
    result->local_to_global_ghost_ = this->local_to_global_ghost_;
    result->set_size(this->get_size());
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
void Matrix<ValueType, LocalIndexType, GlobalIndexType>::move_to(
    Matrix<next_precision<value_type>, local_index_type, global_index_type>*
        result)
{
    convert_to(result);
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
void Matrix<ValueType, LocalIndexType, GlobalIndexType>::read_distributed(
    const device_matrix_data<value_type, global_index_type>& data,
    const Partition<local_index_type, global_index_type>* row_partition,
    const Partition<local_index_type, global_index_type>* col_partition)
{
    const auto comm = this->get_communicator();
    GKO_ASSERT_EQ(data.get_size()[0], row_partition->get_size());
    GKO_ASSERT_EQ(data.get_size()[1], col_partition->get_size());
    GKO_ASSERT_EQ(comm.size(), row_partition->get_num_parts());
    GKO_ASSERT_EQ(comm.size(), col_partition->get_num_parts());
    auto exec = this->get_executor();
    auto local_part = comm.rank();

    // set up LinOp sizes
    auto num_parts = static_cast<size_type>(row_partition->get_num_parts());
    auto global_num_rows = row_partition->get_size();
    auto global_num_cols = col_partition->get_size();
    dim<2> global_dim{global_num_rows, global_num_cols};
    this->set_size(global_dim);

    // temporary storage for the output
    device_matrix_data<value_type, local_index_type> diag_data{exec};
    device_matrix_data<value_type, local_index_type> offdiag_data{exec};
    Array<local_index_type> recv_gather_idxs{exec};
    Array<comm_index_type> recv_offsets_array{exec, num_parts + 1};

    // build diagonal, off-diagonal matrix and communication structures
    exec->run(matrix::make_build_diag_offdiag(
        data, make_temporary_clone(exec, row_partition).get(),
        make_temporary_clone(exec, col_partition).get(), local_part, diag_data,
        offdiag_data, recv_gather_idxs, recv_offsets_array.get_data(),
        local_to_global_ghost_));

    this->diag_mtx_->read(diag_data);
    this->offdiag_mtx_->read(offdiag_data);

    // exchange step 1: determine recv_sizes, send_sizes, send_offsets
    exec->get_master()->copy_from(exec.get(), num_parts + 1,
                                  recv_offsets_array.get_data(),
                                  recv_offsets_.data());
    std::adjacent_difference(recv_offsets_.begin() + 1, recv_offsets_.end(),
                             recv_sizes_.begin());
    comm.all_to_all(recv_sizes_.data(), 1, send_sizes_.data(), 1);
    std::partial_sum(send_sizes_.begin(), send_sizes_.end(),
                     send_offsets_.begin() + 1);
    send_offsets_[0] = 0;

    // exchange step 2: exchange gather_idxs from receivers to senders
    auto needs_host_buffer =
        exec->get_master() != exec && !gko::mpi::is_gpu_aware();
    if (needs_host_buffer) {
        recv_gather_idxs.set_executor(exec->get_master());
        gather_idxs_.clear();
        gather_idxs_.set_executor(exec->get_master());
    }
    gather_idxs_.resize_and_reset(send_offsets_.back());
    comm.all_to_all_v(recv_gather_idxs.get_const_data(), recv_sizes_.data(),
                      recv_offsets_.data(), gather_idxs_.get_data(),
                      send_sizes_.data(), send_offsets_.data());
    if (needs_host_buffer) {
        gather_idxs_.set_executor(exec);
    }
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
void Matrix<ValueType, LocalIndexType, GlobalIndexType>::read_distributed(
    const matrix_data<value_type, global_index_type>& data,
    const Partition<local_index_type, global_index_type>* row_partition,
    const Partition<local_index_type, global_index_type>* col_partition)
{
    this->read_distributed(
        device_matrix_data<value_type, global_index_type>::create_from_host(
            this->get_executor(), data),
        row_partition, col_partition);
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
void Matrix<ValueType, LocalIndexType, GlobalIndexType>::read_distributed(
    const matrix_data<ValueType, global_index_type>& data,
    const Partition<local_index_type, global_index_type>* partition)
{
    this->read_distributed(
        device_matrix_data<value_type, global_index_type>::create_from_host(
            this->get_executor(), data),
        partition, partition);
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
void Matrix<ValueType, LocalIndexType, GlobalIndexType>::read_distributed(
    const device_matrix_data<ValueType, GlobalIndexType>& data,
    const Partition<local_index_type, global_index_type>* partition)
{
    this->read_distributed(data, partition, partition);
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
    auto needs_host_buffer =
        exec->get_master() != exec && !gko::mpi::is_gpu_aware();
    if (needs_host_buffer) {
        host_recv_buffer_.init(exec->get_master(), recv_dim);
        host_send_buffer_.init(exec->get_master(), send_dim);
    }
    local_b->row_gather(&gather_idxs_, send_buffer_.get());
    mpi::contiguous_type type(num_cols, mpi::type_impl<ValueType>::get_type());
    if (needs_host_buffer) {
        host_send_buffer_->copy_from(send_buffer_.get());

        return comm.i_all_to_all_v(
            host_send_buffer_->get_const_values(), send_sizes_.data(),
            send_offsets_.data(), type.get(), host_recv_buffer_->get_values(),
            recv_sizes_.data(), recv_offsets_.data(), type.get());
    } else {
        return comm.i_all_to_all_v(
            send_buffer_->get_const_values(), send_sizes_.data(),
            send_offsets_.data(), type.get(), recv_buffer_->get_values(),
            recv_sizes_.data(), recv_offsets_.data(), type.get());
    }
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
void Matrix<ValueType, LocalIndexType, GlobalIndexType>::apply_impl(
    const LinOp* b, LinOp* x) const
{
    auto dense_b = as<global_vector_type>(b);
    auto exec = this->get_executor();
    auto dense_x = as<global_vector_type>(x);
    auto mutable_local_x = gko::matrix::Dense<ValueType>::create(
        x->get_executor(), dense_x->get_local_vector()->get_size(),
        gko::make_array_view(
            x->get_executor(),
            dense_x->get_local_vector()->get_num_stored_elements(),
            dense_x->get_local_values()),
        dense_x->get_local_vector()->get_stride());
    if (this->get_const_local_offdiag()->get_size()) {
        auto req = this->communicate(dense_b->get_local_vector());
        diag_mtx_->apply(dense_b->get_local_vector(), mutable_local_x.get());
        req.wait();
        auto needs_host_buffer =
            exec->get_master() != exec && !gko::mpi::is_gpu_aware();
        if (needs_host_buffer) {
            recv_buffer_->copy_from(host_recv_buffer_.get());
        }
        offdiag_mtx_->apply(one_scalar_.get(), recv_buffer_.get(),
                            one_scalar_.get(), mutable_local_x.get());
    } else {
        diag_mtx_->apply(dense_b->get_local_vector(), mutable_local_x.get());
    }
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
void Matrix<ValueType, LocalIndexType, GlobalIndexType>::apply_impl(
    const LinOp* alpha, const LinOp* b, const LinOp* beta, LinOp* x) const
{
    auto dense_b = as<global_vector_type>(b);
    auto dense_x = as<global_vector_type>(x);
    auto mutable_local_x = gko::matrix::Dense<ValueType>::create(
        x->get_executor(), dense_x->get_local_vector()->get_size(),
        gko::make_array_view(
            x->get_executor(),
            dense_x->get_local_vector()->get_num_stored_elements(),
            dense_x->get_local_values()),
        dense_x->get_local_vector()->get_stride());
    auto exec = this->get_executor();
    auto local_alpha = as<local_vector_type>(alpha);
    auto local_beta = as<local_vector_type>(beta);
    if (this->get_const_local_offdiag()->get_size()) {
        auto req = this->communicate(dense_b->get_local_vector());
        diag_mtx_->apply(local_alpha, dense_b->get_local_vector(), local_beta,
                         mutable_local_x.get());
        req.wait();
        auto needs_host_buffer =
            exec->get_master() != exec && !gko::mpi::is_gpu_aware();
        if (needs_host_buffer) {
            recv_buffer_->copy_from(host_recv_buffer_.get());
        }
        offdiag_mtx_->apply(local_alpha, recv_buffer_.get(), one_scalar_.get(),
                            mutable_local_x.get());
    } else {
        diag_mtx_->apply(local_alpha, dense_b->get_local_vector(), local_beta,
                         mutable_local_x.get());
    }
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
Matrix<ValueType, LocalIndexType, GlobalIndexType>::Matrix(const Matrix& other)
    : EnableLinOp<Matrix<value_type, local_index_type,
                         global_index_type>>{other.get_executor()},
      DistributedBase{other.get_communicator()}
{
    *this = other;
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
Matrix<ValueType, LocalIndexType, GlobalIndexType>::Matrix(
    Matrix&& other) noexcept
    : EnableLinOp<Matrix<value_type, local_index_type,
                         global_index_type>>{other.get_executor()},
      DistributedBase{other.get_communicator()}
{
    *this = std::move(other);
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
Matrix<ValueType, LocalIndexType, GlobalIndexType>&
Matrix<ValueType, LocalIndexType, GlobalIndexType>::operator=(
    const Matrix& other)
{
    if (this != &other) {
        this->set_size(other.get_size());
        other.diag_mtx_->convert_to(diag_mtx_.get());
        other.offdiag_mtx_->convert_to(offdiag_mtx_.get());
        gather_idxs_ = other.gather_idxs_;
        send_offsets_ = other.send_offsets_;
        recv_offsets_ = other.recv_offsets_;
        recv_sizes_ = other.recv_sizes_;
        send_sizes_ = other.send_sizes_;
        recv_sizes_ = other.recv_sizes_;
        local_to_global_ghost_ = other.local_to_global_ghost_;
        one_scalar_.init(this->get_executor(), dim<2>{1, 1});
        initialize<local_vector_type>({one<value_type>()}, this->get_executor())
            ->move_to(one_scalar_.get());
    }
    return *this;
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
Matrix<ValueType, LocalIndexType, GlobalIndexType>&
Matrix<ValueType, LocalIndexType, GlobalIndexType>::operator=(
    Matrix&& other) noexcept
{
    if (this != &other) {
        this->set_size(other.get_size());
        other.diag_mtx_->move_to(diag_mtx_.get());
        other.offdiag_mtx_->move_to(offdiag_mtx_.get());
        gather_idxs_ = std::move(other.gather_idxs_);
        send_offsets_ = std::move(other.send_offsets_);
        recv_offsets_ = std::move(other.recv_offsets_);
        recv_sizes_ = std::move(other.recv_sizes_);
        send_sizes_ = std::move(other.send_sizes_);
        recv_sizes_ = std::move(other.recv_sizes_);
        local_to_global_ghost_ = std::move(other.local_to_global_ghost_);
        one_scalar_.init(this->get_executor(), dim<2>{1, 1});
        initialize<local_vector_type>({one<value_type>()}, this->get_executor())
            ->move_to(one_scalar_.get());
    }
    return *this;
}


#define GKO_DECLARE_DISTRIBUTED_MATRIX(ValueType, LocalIndexType, \
                                       GlobalIndexType)           \
    class Matrix<ValueType, LocalIndexType, GlobalIndexType>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_LOCAL_GLOBAL_INDEX_TYPE(
    GKO_DECLARE_DISTRIBUTED_MATRIX);


}  // namespace distributed
}  // namespace gko
