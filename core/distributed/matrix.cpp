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


#include <ginkgo/core/distributed/vector.hpp>


#include "core/distributed/matrix_kernels.hpp"


namespace gko {
namespace distributed {
namespace matrix {
GKO_REGISTER_OPERATION(build_diag_offdiag,
                       distributed_matrix::build_diag_offdiag);
}


template <typename ValueType, typename LocalIndexType>
Matrix<ValueType, LocalIndexType>::Matrix(
    std::shared_ptr<const Executor> exec,
    std::shared_ptr<mpi::communicator> comm)
    : EnableLinOp<Matrix<value_type, local_index_type>>{exec},
      DistributedBase{comm},
      send_offsets_(comm->size() + 1),
      send_sizes_(comm->size()),
      recv_offsets_(comm->size() + 1),
      recv_sizes_(comm->size()),
      gather_idxs_{exec},
      local_to_global_row{exec},
      local_to_global_offdiag_col{exec},
      one_scalar_{exec, dim<2>{1, 1}},
      diag_mtx_{exec},
      offdiag_mtx_{exec}
{
    auto one_val = one<ValueType>();
    exec->copy_from(exec->get_master().get(), 1, &one_val,
                    one_scalar_.get_values());
}


template <typename ValueType, typename LocalIndexType>
void Matrix<ValueType, LocalIndexType>::read_distributed(
    const matrix_data<ValueType, global_index_type> &data,
    std::shared_ptr<const Partition<LocalIndexType>> partition)
{
    this->read_distributed(
        Array<matrix_data_entry<ValueType, global_index_type>>::view(
            this->get_executor()->get_master(), data.nonzeros.size(),
            const_cast<matrix_data_entry<ValueType, global_index_type> *>(
                data.nonzeros.data())),
        data.size, partition);
}


template <typename ValueType, typename LocalIndexType>
void Matrix<ValueType, LocalIndexType>::read_distributed(
    const Array<matrix_data_entry<ValueType, global_index_type>> &data,
    dim<2> size, std::shared_ptr<const Partition<LocalIndexType>> partition)
{
    this->partition_ = partition;
    const auto comm = this->get_communicator();
    GKO_ASSERT_IS_SQUARE_MATRIX(size);
    GKO_ASSERT_EQ(size[0], partition->get_size());
    GKO_ASSERT_EQ(comm->size(), partition->get_num_parts());
    using nonzero_type = matrix_data_entry<ValueType, LocalIndexType>;
    auto exec = this->get_executor();
    auto local_data = make_temporary_clone(exec, &data);
    auto local_part = comm->rank();

    // set up LinOp sizes
    auto num_parts = static_cast<size_type>(partition->get_num_parts());
    auto global_size = partition->get_size();
    auto local_size =
        static_cast<size_type>(partition->get_part_size(local_part));
    dim<2> global_dim{global_size, global_size};
    dim<2> diag_dim{local_size, local_size};
    this->set_size(global_dim);

    // temporary storage for the output
    Array<nonzero_type> diag_data{exec};
    Array<nonzero_type> offdiag_data{exec};
    Array<local_index_type> recv_gather_idxs{exec};
    Array<comm_index_type> recv_offsets_array{exec, num_parts + 1};

    // build diagonal, off-diagonal matrix and communication structures
    exec->run(matrix::make_build_diag_offdiag(
        *local_data, partition.get(), local_part, diag_data, offdiag_data,
        recv_gather_idxs, recv_offsets_array.get_data(), local_to_global_row,
        local_to_global_offdiag_col, ValueType{}));

    dim<2> offdiag_dim{local_size, recv_gather_idxs.get_num_elems()};
    this->diag_mtx_.read(diag_data, diag_dim);
    this->offdiag_mtx_.read(offdiag_data, offdiag_dim);

    // exchange step 1: determine recv_sizes, send_sizes, send_offsets
    exec->get_master()->copy_from(exec.get(), num_parts + 1,
                                  recv_offsets_array.get_data(),
                                  recv_offsets_.data());
    // TODO clean this up a bit
    for (size_type i = 0; i < num_parts; i++) {
        recv_sizes_[i] = recv_offsets_[i + 1] - recv_offsets_[i];
    }
    mpi::all_to_all(recv_sizes_.data(), 1, send_sizes_.data(), 1,
                    this->get_communicator());
    std::partial_sum(send_sizes_.begin(), send_sizes_.end(),
                     send_offsets_.begin() + 1);
    send_offsets_[0] = 0;

    // exchange step 2: exchange gather_idxs from receivers to senders
    auto use_host_buffer =
        exec->get_master() != exec /* || comm.is_gpu_aware() */;
    if (use_host_buffer) {
        recv_gather_idxs.set_executor(exec->get_master());
        gather_idxs_.clear();
        gather_idxs_.set_executor(exec->get_master());
    }
    gather_idxs_.resize_and_reset(send_offsets_.back());
    mpi::all_to_all(recv_gather_idxs.get_const_data(), recv_sizes_.data(),
                    recv_offsets_.data(), gather_idxs_.get_data(),
                    send_sizes_.data(), send_offsets_.data(), 1,
                    this->get_communicator());
    if (use_host_buffer) {
        gather_idxs_.set_executor(exec);
    }
}


template <typename ValueType, typename LocalIndexType>
void Matrix<ValueType, LocalIndexType>::communicate(
    const LocalVec *local_b) const
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
    auto use_host_buffer =
        exec->get_master() != exec /* || comm->is_gpu_aware() */;
    if (use_host_buffer) {
        host_recv_buffer_.init(exec->get_master(), recv_dim);
        host_send_buffer_.init(exec->get_master(), send_dim);
    }
    local_b->row_gather(&gather_idxs_, send_buffer_.get());
    if (use_host_buffer) {
        host_send_buffer_->copy_from(send_buffer_.get());
        mpi::all_to_all(host_send_buffer_->get_const_values(),
                        send_sizes_.data(), send_offsets_.data(),
                        host_recv_buffer_->get_values(), recv_sizes_.data(),
                        recv_offsets_.data(), num_cols,
                        this->get_communicator());
        recv_buffer_->copy_from(host_recv_buffer_.get());
    } else {
        mpi::all_to_all(send_buffer_->get_const_values(), send_sizes_.data(),
                        send_offsets_.data(), recv_buffer_->get_values(),
                        recv_sizes_.data(), recv_offsets_.data(), num_cols,
                        this->get_communicator());
    }
}


template <typename ValueType, typename LocalIndexType>
void Matrix<ValueType, LocalIndexType>::apply_impl(const LinOp *b,
                                                   LinOp *x) const
{
    auto dense_b = as<GlobalVec>(b);
    auto dense_x = as<GlobalVec>(x);
    diag_mtx_.apply(dense_b->get_local(), dense_x->get_local());
    this->communicate(dense_b->get_local());
    offdiag_mtx_.apply(&one_scalar_, recv_buffer_.get(), &one_scalar_,
                       dense_x->get_local());
}


template <typename ValueType, typename LocalIndexType>
void Matrix<ValueType, LocalIndexType>::apply_impl(const LinOp *alpha,
                                                   const LinOp *b,
                                                   const LinOp *beta,
                                                   LinOp *x) const
{
    auto vec_b = as<GlobalVec>(b);
    auto vec_x = as<GlobalVec>(x);
    auto local_alpha = as<LocalVec>(alpha);
    auto local_beta = as<LocalVec>(beta);
    diag_mtx_.apply(local_alpha, vec_b->get_local(), local_beta,
                    vec_x->get_local());
    this->communicate(vec_b->get_local());
    offdiag_mtx_.apply(local_alpha, recv_buffer_.get(), &one_scalar_,
                       vec_x->get_local());
}


template <typename ValueType, typename LocalIndexType>
void Matrix<ValueType, LocalIndexType>::validate_data() const
{
    LinOp::validate_data();
    one_scalar_.validate_data();
    diag_mtx_.validate_data();
    offdiag_mtx_.validate_data();
    const auto exec = this->get_executor();
    const auto host_exec = exec->get_master();
    const auto comm = this->get_communicator();
    // executors
    GKO_VALIDATION_CHECK(one_scalar_.get_executor() == exec);
    GKO_VALIDATION_CHECK(diag_mtx_.get_executor() == exec);
    GKO_VALIDATION_CHECK(offdiag_mtx_.get_executor() == exec);
    GKO_VALIDATION_CHECK(gather_idxs_.get_executor() == exec);
    GKO_VALIDATION_CHECK(host_send_buffer_.get() == nullptr ||
                         host_send_buffer_->get_executor() == host_exec);
    GKO_VALIDATION_CHECK(host_recv_buffer_.get() == nullptr ||
                         host_recv_buffer_->get_executor() == host_exec);
    GKO_VALIDATION_CHECK(send_buffer_.get() == nullptr ||
                         send_buffer_->get_executor() == exec);
    GKO_VALIDATION_CHECK(recv_buffer_.get() == nullptr ||
                         recv_buffer_->get_executor() == exec);
    // sizes are matching
    const auto num_local_rows = diag_mtx_.get_size()[0];
    const auto num_offdiag_cols = offdiag_mtx_.get_size()[1];
    const auto num_gather_rows = gather_idxs_.get_num_elems();
    GKO_VALIDATION_CHECK(num_local_rows == diag_mtx_.get_size()[1]);
    GKO_VALIDATION_CHECK(num_local_rows == offdiag_mtx_.get_size()[0]);
    auto num_local_rows_sum = diag_mtx_.get_size()[0];
    mpi::all_reduce(&num_local_rows_sum, 1, mpi::op_type::sum,
                    this->get_communicator());
    GKO_VALIDATION_CHECK(num_local_rows_sum == this->get_size()[0]);
    const auto num_parts = comm->rank();
    GKO_VALIDATION_CHECK(num_parts == send_sizes_.size());
    GKO_VALIDATION_CHECK(num_parts == recv_sizes_.size());
    GKO_VALIDATION_CHECK(num_parts + 1 == send_offsets_.size());
    GKO_VALIDATION_CHECK(num_parts + 1 == recv_offsets_.size());
    // communication data structures are consistent
    auto send_copy = send_sizes_;
    auto recv_copy = recv_sizes_;
    for (comm_index_type i = 0; i < num_parts; i++) {
        GKO_VALIDATION_CHECK(send_sizes_[i] ==
                             send_offsets_[i + 1] - send_offsets_[i]);
        GKO_VALIDATION_CHECK(recv_sizes_[i] ==
                             recv_offsets_[i + 1] - recv_offsets_[i]);
    }
    mpi::all_to_all(send_copy.data(), 1, this->get_communicator());
    mpi::all_to_all(recv_copy.data(), 1, this->get_communicator());
    GKO_VALIDATION_CHECK(send_copy == recv_sizes_);
    GKO_VALIDATION_CHECK(recv_copy == send_sizes_);
    // gather indices are in bounds
    Array<local_index_type> host_gather_idxs(host_exec, gather_idxs_);
    const auto host_gather_idx_ptr = host_gather_idxs.get_const_data();
    GKO_VALIDATION_CHECK_NAMED(
        "gather indices need to be in range",
        std::all_of(
            host_gather_idx_ptr, host_gather_idx_ptr + num_gather_rows,
            [&](auto row) { return row >= 0 && row < num_local_rows; }));
}


#define GKO_DECLARE_DISTRIBUTED_MATRIX(ValueType, LocalIndexType) \
    class Matrix<ValueType, LocalIndexType>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_DISTRIBUTED_MATRIX);


}  // namespace distributed
}  // namespace gko
