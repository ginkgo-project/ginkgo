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
GKO_REGISTER_OPERATION(map_to_global_idxs,
                       distributed_matrix::map_to_global_idxs);


}  // namespace
}  // namespace matrix


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
Matrix<ValueType, LocalIndexType, GlobalIndexType>::Matrix(
    std::shared_ptr<const Executor> exec)
    : Matrix(exec, mpi::communicator(MPI_COMM_NULL))
{}


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
    Matrix<value_type, local_index_type, global_index_type>* result) const
{
    result->diag_mtx_->copy_from(this->diag_mtx_.get());
    result->offdiag_mtx_->copy_from(this->offdiag_mtx_.get());
    result->one_scalar_.init_from(this->one_scalar_.get());
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
    Matrix<value_type, local_index_type, global_index_type>* result)
{
    EnableLinOp<Matrix>::move_to(result);
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
void Matrix<ValueType, LocalIndexType, GlobalIndexType>::read_distributed(
    const matrix_data<ValueType, global_index_type>& data,
    const Partition<local_index_type, global_index_type>* partition)
{
    this->read_distributed(
        device_matrix_data<value_type, global_index_type>::create_from_host(
            this->get_executor(), data),
        partition);
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
void Matrix<ValueType, LocalIndexType, GlobalIndexType>::read_distributed(
    const device_matrix_data<ValueType, GlobalIndexType>& data,
    const Partition<local_index_type, global_index_type>* partition)
{
    const auto comm = this->get_communicator();
    GKO_ASSERT_IS_SQUARE_MATRIX(data.get_size());
    GKO_ASSERT_EQ(data.get_size()[0], partition->get_size());
    GKO_ASSERT_EQ(comm.size(), partition->get_num_parts());
    using nonzero_type = matrix_data_entry<value_type, local_index_type>;
    auto exec = this->get_executor();
    // TODO: after update move data to correct executor
    auto local_part = comm.rank();

    // set up LinOp sizes
    auto num_parts = static_cast<size_type>(partition->get_num_parts());
    auto global_size = partition->get_size();
    dim<2> global_dim{global_size, global_size};
    this->set_size(global_dim);

    // temporary storage for the output
    device_matrix_data<value_type, local_index_type> diag_data{exec};
    device_matrix_data<value_type, local_index_type> offdiag_data{exec};
    Array<local_index_type> recv_gather_idxs{exec};
    Array<comm_index_type> recv_offsets_array{exec, num_parts + 1};

    // build diagonal, off-diagonal matrix and communication structures
    exec->run(matrix::make_build_diag_offdiag(
        data, partition, local_part, diag_data, offdiag_data, recv_gather_idxs,
        recv_offsets_array.get_data(), local_to_global_ghost_));

    this->diag_mtx_->read(diag_data);
    this->offdiag_mtx_->read(offdiag_data);

    // exchange step 1: determine recv_sizes, send_sizes, send_offsets
    exec->get_master()->copy_from(exec.get(), num_parts + 1,
                                  recv_offsets_array.get_data(),
                                  recv_offsets_.data());
    // TODO clean this up a bit
    for (size_type i = 0; i < num_parts; i++) {
        recv_sizes_[i] = recv_offsets_[i + 1] - recv_offsets_[i];
    }
    comm.all_to_all(recv_sizes_.data(), 1, send_sizes_.data(), 1);
    std::partial_sum(send_sizes_.begin(), send_sizes_.end(),
                     send_offsets_.begin() + 1);
    send_offsets_[0] = 0;

    // exchange step 2: exchange gather_idxs from receivers to senders
    auto needs_host_buffer =
        exec->get_master() != exec /* || comm.is_gpu_aware() */;
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
    if (needs_host_buffer) {
        host_send_buffer_->copy_from(send_buffer_.get());
        return comm.i_all_to_all_v(host_send_buffer_->get_const_values(),
                                   send_sizes_.data(), send_offsets_.data(),
                                   host_recv_buffer_->get_values(),
                                   recv_sizes_.data(), recv_offsets_.data());
    } else {
        return comm.i_all_to_all_v(send_buffer_->get_const_values(),
                                   send_sizes_.data(), send_offsets_.data(),
                                   recv_buffer_->get_values(),
                                   recv_sizes_.data(), recv_offsets_.data());
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


#define GKO_DECLARE_DISTRIBUTED_MATRIX(ValueType, LocalIndexType, \
                                       GlobalIndexType)           \
    class Matrix<ValueType, LocalIndexType, GlobalIndexType>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_LOCAL_GLOBAL_INDEX_TYPE(
    GKO_DECLARE_DISTRIBUTED_MATRIX);


}  // namespace distributed
}  // namespace gko
