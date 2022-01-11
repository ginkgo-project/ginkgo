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
GKO_REGISTER_OPERATION(map_to_global_idxs,
                       distributed_matrix::map_to_global_idxs);
}  // namespace matrix


template <typename ValueType, typename LocalIndexType>
Matrix<ValueType, LocalIndexType>::Matrix(
    std::shared_ptr<const Executor> exec, const gko::dim<2>& size,
    std::shared_ptr<mpi::communicator> comm)
    : EnableLinOp<Matrix<value_type, local_index_type>>{exec, size},
      DistributedBase{comm},
      send_offsets_(comm->size() + 1),
      send_sizes_(comm->size()),
      recv_offsets_(comm->size() + 1),
      recv_sizes_(comm->size()),
      gather_idxs_{exec},
      local_to_global_row{exec},
      local_to_global_offdiag_col{exec},
      one_scalar_{exec, dim<2>{1, 1}},
      diag_mtx_{LocalMtx::create(exec)},
      offdiag_mtx_{LocalMtx::create(exec)},
      local_mtx_blocks_{},
      serialized_local_mtx_{std::make_shared<serialized_mtx>(exec)}
{
    auto one_val = one<ValueType>();
    exec->copy_from(exec->get_master().get(), 1, &one_val,
                    one_scalar_.get_values());
}


template <typename ValueType, typename LocalIndexType>
Matrix<ValueType, LocalIndexType>::Matrix(
    std::shared_ptr<const Executor> exec, const gko::dim<2>& size,
    std::shared_ptr<const Partition<LocalIndexType>> partition,
    std::shared_ptr<mpi::communicator> comm)
    : EnableLinOp<Matrix<value_type, local_index_type>>{exec, size},
      DistributedBase{comm},
      partition_{partition},
      send_offsets_(comm->size() + 1),
      send_sizes_(comm->size()),
      recv_offsets_(comm->size() + 1),
      recv_sizes_(comm->size()),
      gather_idxs_{exec},
      local_to_global_row{exec},
      local_to_global_offdiag_col{exec},
      one_scalar_{exec, dim<2>{1, 1}},
      diag_mtx_{LocalMtx::create(
          exec, gko::dim<2>(partition->get_part_size(comm->rank())))},
      offdiag_mtx_{LocalMtx::create(
          exec, gko::dim<2>(partition->get_part_size(comm->rank()),
                            size[1] - partition->get_part_size(comm->rank())))},
      local_mtx_blocks_{},
      serialized_local_mtx_{std::make_shared<serialized_mtx>(exec)}
{
    auto one_val = one<ValueType>();
    exec->copy_from(exec->get_master().get(), 1, &one_val,
                    one_scalar_.get_values());
}


template <typename ValueType, typename LocalIndexType>
void Matrix<ValueType, LocalIndexType>::copy_communication_data(
    const Matrix<ValueType, LocalIndexType>* other)
{
    this->send_offsets_ = other->send_offsets_;
    this->send_sizes_ = other->send_sizes_;
    this->recv_offsets_ = other->recv_offsets_;
    this->recv_sizes_ = other->recv_sizes_;
    this->gather_idxs_ = other->gather_idxs_;
    this->local_to_global_row = other->local_to_global_row;
    this->local_to_global_offdiag_col = other->local_to_global_offdiag_col;
}


template <typename ValueType, typename LocalIndexType>
void Matrix<ValueType, LocalIndexType>::read_distributed(
    const matrix_data<ValueType, global_index_type>& data,
    std::shared_ptr<const Partition<LocalIndexType>> partition)
{
    this->read_distributed(
        Array<matrix_data_entry<ValueType, global_index_type>>::view(
            this->get_executor()->get_master(), data.nonzeros.size(),
            const_cast<matrix_data_entry<ValueType, global_index_type>*>(
                data.nonzeros.data())),
        data.size, partition);
}


template <typename ValueType, typename LocalIndexType>
void Matrix<ValueType, LocalIndexType>::read_distributed(
    const Array<matrix_data_entry<ValueType, global_index_type>>& data,
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
    this->diag_mtx_->read(diag_data, diag_dim);
    this->offdiag_mtx_->read(offdiag_data, offdiag_dim);

    // exchange step 1: determine recv_sizes, send_sizes, send_offsets
    exec->get_master()->copy_from(exec.get(), num_parts + 1,
                                  recv_offsets_array.get_data(),
                                  recv_offsets_.data());
    // TODO clean this up a bit
    for (size_type i = 0; i < num_parts; i++) {
        recv_sizes_[i] = recv_offsets_[i + 1] - recv_offsets_[i];
    }
    comm->all_to_all(recv_sizes_.data(), 1, send_sizes_.data(), 1);
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
    comm->all_to_all_v(recv_gather_idxs.get_const_data(), recv_sizes_.data(),
                       recv_offsets_.data(), gather_idxs_.get_data(),
                       send_sizes_.data(), send_offsets_.data());
    if (use_host_buffer) {
        gather_idxs_.set_executor(exec);
    }
    this->update_matrix_blocks();
    this->serialize_matrix_blocks();
}


template <typename ValueType, typename LocalIndexType>
void Matrix<ValueType, LocalIndexType>::init_local_mtx_blocks()
{
    const auto comm = this->get_communicator();
    const auto part = this->get_partition();
    auto num_parts = static_cast<size_type>(part->get_num_parts());
    auto global_size = part->get_size();
    auto local_rank = comm->rank();
    std::vector<size_type> local_size;
    for (auto i = 0; i < num_parts; ++i) {
        local_size.emplace_back(static_cast<size_type>(part->get_part_size(i)));
    }
    size_type col_st = 0;
    size_type col_end = 0;
    for (auto i = 0; i < num_parts; ++i) {
        if (i == local_rank) {
            this->local_mtx_blocks_.emplace_back(this->diag_mtx_);
        } else {
            col_end += local_size[i];
            auto offdiag_row_span = gko::span(0, local_size[local_rank]);
            auto offdiag_col_span = gko::span(col_st, col_end);
            this->local_mtx_blocks_.emplace_back(LocalMtx::create(
                this->get_executor(), gko::dim<2>(offdiag_row_span.length(),
                                                  offdiag_col_span.length())));

            col_st += local_size[i];
        }
    }
}


template <typename ValueType, typename LocalIndexType>
void Matrix<ValueType, LocalIndexType>::update_matrix_blocks()
{
    const auto comm = this->get_communicator();
    const auto part = this->get_partition();
    auto num_parts = static_cast<size_type>(part->get_num_parts());
    auto global_size = part->get_size();
    auto local_rank = comm->rank();
    std::vector<size_type> local_size;
    for (auto i = 0; i < num_parts; ++i) {
        local_size.emplace_back(static_cast<size_type>(part->get_part_size(i)));
    }
    size_type col_st = 0;
    size_type col_end = 0;
    for (auto i = 0; i < num_parts; ++i) {
        if (i == local_rank) {
            this->local_mtx_blocks_.emplace_back(this->diag_mtx_);
        } else {
            col_end += local_size[i];
            auto offdiag_row_span = gko::span(0, local_size[local_rank]);
            auto offdiag_col_span = gko::span(col_st, col_end);
            this->local_mtx_blocks_.emplace_back(
                this->offdiag_mtx_->create_submatrix(offdiag_row_span,
                                                     offdiag_col_span));
            col_st += local_size[i];
        }
    }
}


template <typename ValueType, typename LocalIndexType>
void Matrix<ValueType, LocalIndexType>::serialize_matrix_blocks()
{
    const auto exec = this->get_executor();
    const auto comm = this->get_communicator();
    const auto part = this->get_partition();
    auto num_parts = static_cast<size_type>(part->get_num_parts());
    auto global_size = part->get_size();
    auto local_rank = comm->rank();
    auto mat_blocks = this->local_mtx_blocks_;
    auto nnz_idx_count = 0;
    auto nnz_ptr_count = 0;
    // pre-compute Array sizes
    for (auto i = 0; i < mat_blocks.size(); ++i) {
        nnz_idx_count += mat_blocks[i]->get_num_stored_elements();
        nnz_ptr_count += mat_blocks[i]->get_size()[0] + 1;
    }
    this->serialized_local_mtx_->col_idxs =
        Array<local_index_type>(exec, nnz_idx_count);
    this->serialized_local_mtx_->values =
        Array<value_type>(exec, nnz_idx_count);
    this->serialized_local_mtx_->row_ptrs =
        Array<local_index_type>(exec, nnz_ptr_count);
    auto row_offset = 0;
    auto nnz_offset = 0;
    for (auto i = 0; i < mat_blocks.size(); ++i) {
        auto local_nnz_count = mat_blocks[i]->get_num_stored_elements();
        auto local_size = mat_blocks[i]->get_size();
        exec->copy(
            local_nnz_count, mat_blocks[i]->get_const_col_idxs(),
            this->serialized_local_mtx_->col_idxs.get_data() + nnz_offset);
        exec->copy(local_nnz_count, mat_blocks[i]->get_const_values(),
                   this->serialized_local_mtx_->values.get_data() + nnz_offset);
        exec->copy(
            local_size[0] + 1, mat_blocks[i]->get_const_row_ptrs(),
            this->serialized_local_mtx_->row_ptrs.get_data() + row_offset);
        row_offset += local_size[0] + 1;
        nnz_offset += local_nnz_count;
    }
    GKO_ASSERT(row_offset == nnz_ptr_count);
    GKO_ASSERT(nnz_offset == nnz_idx_count);
}


template <typename ValueType, typename LocalIndexType>
void Matrix<ValueType, LocalIndexType>::de_serialize_matrix_blocks(
    serialized_mtx& serialized_mtx,
    std::vector<std::shared_ptr<LocalMtx>>& mat_blocks) const
{
    const auto exec = this->get_executor();
    const auto comm = this->get_communicator();
    const auto part = this->get_partition();
    auto num_parts = static_cast<size_type>(part->get_num_parts());
    auto global_size = part->get_size();
    auto local_rank = comm->rank();
    auto row_offset = 0;
    auto nnz_offset = 0;
    for (auto i = 0; i < mat_blocks.size(); ++i) {
        auto local_nnz_count = mat_blocks[i]->get_num_stored_elements();
        auto local_size = mat_blocks[i]->get_size();
        exec->copy(local_nnz_count,
                   serialized_mtx.col_idxs.get_const_data() + nnz_offset,
                   mat_blocks[i]->get_col_idxs());
        exec->copy(local_nnz_count,
                   serialized_mtx.values.get_const_data() + nnz_offset,
                   mat_blocks[i]->get_values());
        exec->copy(local_size[0] + 1,
                   serialized_mtx.row_ptrs.get_const_data() + row_offset,
                   mat_blocks[i]->get_row_ptrs());
        row_offset += local_size[0] + 1;
        nnz_offset += local_nnz_count;
    }
}


template <typename ValueType, typename LocalIndexType>
std::vector<
    std::shared_ptr<typename Matrix<ValueType, LocalIndexType>::LocalMtx>>
Matrix<ValueType, LocalIndexType>::get_block_approx(
    const Overlap<size_type>& block_overlaps,
    const Array<size_type>& block_sizes)
{
    return std::vector<std::shared_ptr<LocalMtx>>{this->get_local_diag()};
}


template <typename ValueType, typename LocalIndexType>
std::vector<
    std::shared_ptr<const typename Matrix<ValueType, LocalIndexType>::LocalMtx>>
Matrix<ValueType, LocalIndexType>::get_block_approx(
    const Overlap<size_type>& block_overlaps,
    const Array<size_type>& block_sizes) const
{
    return std::vector<std::shared_ptr<const LocalMtx>>{this->get_local_diag()};
}


template <typename ValueType, typename LocalIndexType>
mpi::request Matrix<ValueType, LocalIndexType>::communicate(
    const LocalVec* local_b) const
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
        exec->get_master() != exec || !gko::mpi::is_gpu_aware();
    if (use_host_buffer) {
        host_recv_buffer_.init(exec->get_master(), recv_dim);
        host_send_buffer_.init(exec->get_master(), send_dim);
    }
    local_b->row_gather(&gather_idxs_, send_buffer_.get());
    mpi::request req;
    if (use_host_buffer) {
        // TODO Need to fix for multiple RHS
        host_send_buffer_->copy_from(send_buffer_.get());
        req = comm->i_all_to_all_v(host_send_buffer_->get_const_values(),
                                   send_sizes_.data(), send_offsets_.data(),
                                   host_recv_buffer_->get_values(),
                                   recv_sizes_.data(), recv_offsets_.data());
    } else {
        req = comm->i_all_to_all_v(send_buffer_->get_const_values(),
                                   send_sizes_.data(), send_offsets_.data(),
                                   recv_buffer_->get_values(),
                                   recv_sizes_.data(), recv_offsets_.data());
    }
    return req;
}


template <typename ValueType, typename LocalIndexType>
void Matrix<ValueType, LocalIndexType>::apply_impl(const LinOp* b,
                                                   LinOp* x) const
{
    auto comm = this->get_communicator();
    auto exec = this->get_executor();
    auto part = this->get_partition();
    auto num_parts = static_cast<size_type>(part->get_num_parts());
    auto global_size = part->get_size();
    using GlobalMat = Matrix<ValueType, LocalIndexType>;
    if (auto mat_b = dynamic_cast<const GlobalMat*>(b)) {
        // TODO: Move these prepare steps to a separate public function callable
        // by the user.
        auto mat_x = dynamic_cast<GlobalMat*>(x);
        // FIXME Pre-multiply (SpGEMM) local diagonal blocks and store in C
        // this->get_local_diag()->apply(mat_b->get_local_diag().get(),
        //                               mat_x->get_local_diag().get());
        // Create submatrices from off-diagonal blocks.
        const auto b_local_mtx_blocks = mat_b->get_local_mtx_blocks();
        mat_x->init_local_mtx_blocks();
        auto x_local_mtx_blocks = mat_x->get_local_mtx_blocks();
        // Sizes of the local diagonal and off-diagonal blocks
        std::vector<gko::dim<2>> local_sizes{};
        for (auto i = 0; i < comm->size(); ++i) {
            local_sizes.emplace_back(gko::dim<2>(
                part->get_part_size(comm->rank()), part->get_part_size(i)));
        }
        GKO_ASSERT(b_local_mtx_blocks.size() == comm->size());
        GKO_ASSERT(x_local_mtx_blocks.size() == comm->size());
        // GKO_ASSERT(comm->size() == 1024);
        std::vector<size_type> local_block_nnz{};
        for (auto i = 0; i < comm->size(); ++i) {
            local_block_nnz.emplace_back(
                b_local_mtx_blocks[i]->get_num_stored_elements());
        }
        const auto num_blocks = comm->size() * comm->size();
        std::vector<size_type> b_block_nnz{static_cast<size_type>(num_blocks),
                                           0};
        std::vector<int> disp{comm->size(), 0};
        for (auto i = 0; i < disp.size(); ++i) {
            disp[i] = i * comm->size();
        }
        std::vector<int> recv_counts{num_blocks, num_blocks};
        // Gather nnz counts of all blocks onto current rank
        comm->all_gather_v(local_block_nnz.data(), comm->size(),
                           b_block_nnz.data(), recv_counts.data(), disp.data());
        // Allocate/assign the sub matrices for the B matrix, which we receive.
        // TODO Not sure if this can be const LocalMtx, probably need a
        // const_cast for local_diag
        // FIX: Due to serialization simplifications, this is now a non-const
        // object.
        // TODO: Memory optimizations possible by using existing local diagonal
        // block, instead of creating a new object and copying the diag block.
        std::vector<std::shared_ptr<LocalMtx>> b_recv{
            static_cast<size_type>(comm->size() * comm->size()), nullptr};
        std::vector<size_type> b_cumul_col_nnz(comm->size(), 0);
        // Allocate the block b matrix.
        // Use column major for the blocks, because that makes it easier to loop
        // through when doing the local SpGEMMs
        for (auto j = 0; j < comm->size(); ++j) {
            for (auto i = 0; i < comm->size(); ++i) {
                b_recv[i + comm->size() * j] = LocalMtx::create(
                    exec, gko::dim<2>((local_sizes[j])[1], (local_sizes[i])[0]),
                    b_block_nnz[i * comm->size() + j]);
                b_cumul_col_nnz[j] += b_block_nnz[i * comm->size() + j];
            }
        }
        std::cout << "Here " << __LINE__ << std::endl;
        // The received data is also serialized, so allocate buffers to receive
        // in which we later will need to de-serialize.
        auto total_b_nnz_count = std::accumulate(
            b_block_nnz.data(), b_block_nnz.data() + num_blocks, size_type{0});
        serialized_mtx serialized_b_mtx = serialized_mtx{
            exec, total_b_nnz_count,
            static_cast<size_type>(comm->size() *
                                   (this->get_size()[0] + comm->size()))};
        // TODO: Compute a prefix sum buffer later to make this loop parallel.
        auto row_offset = 0;
        auto nnz_offset = 0;
        auto ser_recv_nnz_offset = 0;
        auto ser_recv_col_offset = 0;
        const auto ser_mtx = this->serialized_local_mtx_;
        // Communicate the b matrix. The input we get is in a block column major
        // format, so during de-serialization, we will need to fill that in
        // correctly
        for (auto i = 0; i < b_local_mtx_blocks.size(); ++i) {
            auto local_nnz_count =
                b_local_mtx_blocks[i]->get_num_stored_elements();
            auto local_size = b_local_mtx_blocks[i]->get_size();
            std::cout << "Here " << __LINE__ << " iter " << i << std::endl;
            comm->all_gather(
                ser_mtx->col_idxs.get_const_data() + nnz_offset,
                local_nnz_count,
                serialized_b_mtx.col_idxs.get_data() + ser_recv_nnz_offset,
                local_nnz_count);
            std::cout << "Here " << __LINE__ << " iter " << i << std::endl;
            comm->all_gather(
                ser_mtx->values.get_const_data() + nnz_offset, local_nnz_count,
                serialized_b_mtx.values.get_data() + ser_recv_nnz_offset,
                local_nnz_count);
            std::cout << "Here " << __LINE__ << " iter " << i << std::endl;
            comm->all_gather(
                ser_mtx->col_idxs.get_const_data() + row_offset,
                local_size[0] + 1,
                serialized_b_mtx.row_ptrs.get_data() + ser_recv_col_offset,
                local_size[0] + 1);
            std::cout << "Here " << __LINE__ << " iter " << i << std::endl;
            row_offset += local_size[0] + 1;
            nnz_offset += local_nnz_count;
            ser_recv_nnz_offset += b_cumul_col_nnz[i];
            ser_recv_col_offset += this->get_size()[0] + comm->size();
        }
        std::cout << "Here " << __LINE__ << std::endl;
        de_serialize_matrix_blocks(serialized_b_mtx, b_recv);
        std::cout << "Here " << __LINE__ << std::endl;
        auto one = gko::initialize<LocalVec>({1.0}, exec);
        // for (auto i = 0; i < comm->size(); ++i) {
        //     for (auto j = 0; j < comm->size(); ++j) {
        //         if (i == 0) {
        //             this->local_mtx_blocks_[i]->apply(
        //                 b_recv[i * comm->size() + j].get(),
        //                 x_local_mtx_blocks[j].get());
        //         } else {
        //             this->local_mtx_blocks_[i]->apply(
        //                 one.get(), b_recv[i * comm->size() + j].get(),
        //                 one.get(), x_local_mtx_blocks[j].get());
        //         }
        //     }
        // }
        std::cout << "Here " << __LINE__ << std::endl;
    } else {
        auto dense_b = as<GlobalVec>(b);
        auto dense_x = as<GlobalVec>(x);
        auto req = this->communicate(dense_b->get_local());
        diag_mtx_->apply(dense_b->get_local(), dense_x->get_local());
        req.wait();
        auto use_host_buffer =
            exec->get_master() != exec || !gko::mpi::is_gpu_aware();
        if (use_host_buffer) {
            recv_buffer_->copy_from(host_recv_buffer_.get());
        }
        offdiag_mtx_->apply(&one_scalar_, recv_buffer_.get(), &one_scalar_,
                            dense_x->get_local());
    }
    std::cout << "Here " << __LINE__ << std::endl;
}


template <typename ValueType, typename LocalIndexType>
void Matrix<ValueType, LocalIndexType>::apply_impl(const LinOp* alpha,
                                                   const LinOp* b,
                                                   const LinOp* beta,
                                                   LinOp* x) const
{
    auto vec_b = as<GlobalVec>(b);
    auto vec_x = as<GlobalVec>(x);
    auto exec = this->get_executor();
    auto local_alpha = as<LocalVec>(alpha);
    auto local_beta = as<LocalVec>(beta);
    auto req = this->communicate(vec_b->get_local());
    diag_mtx_->apply(local_alpha, vec_b->get_local(), local_beta,
                     vec_x->get_local());
    req.wait();
    auto use_host_buffer =
        exec->get_master() != exec || !gko::mpi::is_gpu_aware();
    if (use_host_buffer) {
        recv_buffer_->copy_from(host_recv_buffer_.get());
    }
    offdiag_mtx_->apply(local_alpha, recv_buffer_.get(), &one_scalar_,
                        vec_x->get_local());
}


template <typename ValueType, typename LocalIndexType>
void Matrix<ValueType, LocalIndexType>::validate_data() const
{
    LinOp::validate_data();
    one_scalar_.validate_data();
    diag_mtx_->validate_data();
    offdiag_mtx_->validate_data();
    const auto exec = this->get_executor();
    const auto host_exec = exec->get_master();
    const auto comm = this->get_communicator();
    // executors
    GKO_VALIDATION_CHECK(one_scalar_.get_executor() == exec);
    GKO_VALIDATION_CHECK(diag_mtx_->get_executor() == exec);
    GKO_VALIDATION_CHECK(offdiag_mtx_->get_executor() == exec);
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
    const auto num_local_rows = diag_mtx_->get_size()[0];
    const auto num_offdiag_cols = offdiag_mtx_->get_size()[1];
    const auto num_gather_rows = gather_idxs_.get_num_elems();
    GKO_VALIDATION_CHECK(num_local_rows == diag_mtx_->get_size()[1]);
    GKO_VALIDATION_CHECK(num_local_rows == offdiag_mtx_->get_size()[0]);
    auto num_local_rows_sum = diag_mtx_->get_size()[0];
    comm->all_reduce(&num_local_rows_sum, 1, MPI_SUM);
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
    comm->all_to_all(send_copy.data(), 1);
    comm->all_to_all(recv_copy.data(), 1);
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
