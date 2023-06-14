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


#include <ginkgo/core/base/precision_dispatch.hpp>
#include <ginkgo/core/distributed/vector.hpp>
#include <ginkgo/core/matrix/csr.hpp>


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


mpi::communicator setup_empty_neighborhood_communicator(mpi::communicator& comm)
{
    MPI_Comm graph;

    int source = comm.rank();
    int degree = 0;
    MPI_Dist_graph_create(comm.get(), 1, &source, &degree, nullptr,
                          MPI_UNWEIGHTED, MPI_INFO_NULL, false, &graph);

    return mpi::communicator{graph}.duplicate();
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
Matrix<ValueType, LocalIndexType, GlobalIndexType>::Matrix(
    std::shared_ptr<const Executor> exec, mpi::communicator comm)
    : Matrix(exec, comm, with_matrix_type<gko::matrix::Csr>())
{}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
Matrix<ValueType, LocalIndexType, GlobalIndexType>::Matrix(
    std::shared_ptr<const Executor> exec, mpi::communicator comm,
    ptr_param<const LinOp> local_matrix_template)
    : Matrix(exec, comm, local_matrix_template, local_matrix_template)
{}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
Matrix<ValueType, LocalIndexType, GlobalIndexType>::Matrix(
    std::shared_ptr<const Executor> exec, mpi::communicator comm,
    ptr_param<const LinOp> local_matrix_template,
    ptr_param<const LinOp> non_local_matrix_template)
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
    result->local_mtx_->copy_from(this->local_mtx_);
    result->non_local_mtx_->copy_from(this->non_local_mtx_);
    result->gather_idxs_ = this->gather_idxs_;
    result->send_offsets_ = this->send_offsets_;
    result->recv_offsets_ = this->recv_offsets_;
    result->recv_sizes_ = this->recv_sizes_;
    result->send_sizes_ = this->send_sizes_;
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
    result->local_mtx_->move_from(this->local_mtx_);
    result->non_local_mtx_->move_from(this->non_local_mtx_);
    result->gather_idxs_ = std::move(this->gather_idxs_);
    result->send_offsets_ = std::move(this->send_offsets_);
    result->recv_offsets_ = std::move(this->recv_offsets_);
    result->recv_sizes_ = std::move(this->recv_sizes_);
    result->send_sizes_ = std::move(this->send_sizes_);
    result->non_local_to_global_ = std::move(this->non_local_to_global_);
    result->set_size(this->get_size());
    this->set_size({});
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
void Matrix<ValueType, LocalIndexType,
            GlobalIndexType>::enable_neighborhood_communication()
{
    using distributed::comm_index_type;
    auto comm = this->get_communicator();
    // define only outgoing edges:
    auto source = comm.rank();
    // compute degree for own rank
    auto degree = static_cast<comm_index_type>(
        std::count_if(send_sizes_.begin(), send_sizes_.end(),
                      [](const auto v) { return v > 0; }));
    // destinations are ranks with send_size > 0
    std::vector<comm_index_type> destinations;
    std::vector<comm_index_type> weight;
    for (int r = 0; r < send_sizes_.size(); ++r) {
        if (send_sizes_[r] > 0) {
            destinations.push_back(r);
            weight.push_back(send_sizes_[r]);
        }
    }

    MPI_Comm graph;
    GKO_ASSERT_NO_MPI_ERRORS(MPI_Dist_graph_create(
        comm.get(), 1, &source, &degree, destinations.data(), weight.data(),
        MPI_INFO_NULL, true, &graph));
    neighbor_comm_ = mpi::communicator{graph}.duplicate();

    comm_index_type num_in_neighbors;
    comm_index_type num_out_neighbors;
    int weighted;
    GKO_ASSERT_NO_MPI_ERRORS(
        MPI_Dist_graph_neighbors_count(neighbor_comm_->get(), &num_in_neighbors,
                                       &num_out_neighbors, &weighted));

    std::vector<comm_index_type> out_neighbors(num_out_neighbors);
    std::vector<comm_index_type> in_neighbors(num_in_neighbors);
    std::vector<comm_index_type> out_weight(num_out_neighbors);
    std::vector<comm_index_type> in_weight(num_in_neighbors);
    GKO_ASSERT_NO_MPI_ERRORS(MPI_Dist_graph_neighbors(
        neighbor_comm_->get(), num_in_neighbors, in_neighbors.data(),
        in_weight.data(), num_out_neighbors, out_neighbors.data(),
        out_weight.data()));

    // compress communication info
    std::vector<comm_index_type> comp_send_offsets(num_out_neighbors + 1);
    std::vector<comm_index_type> comp_recv_offsets(num_in_neighbors + 1);

    bool reordered = comm.rank() != neighbor_comm_->rank();
    comm.all_reduce(this->get_executor(), &reordered, 1, MPI_LOR);

    if (comm.rank() == 0 && reordered) {
        std::cerr << "Reordered" << std::endl;
    }


    for (int r = 0; r < in_neighbors.size(); ++r) {
        comp_recv_offsets[r] = recv_offsets_[in_neighbors[r]];
    }
    comp_recv_offsets.back() = recv_offsets_.back();
    for (int r = 0; r < out_neighbors.size(); ++r) {
        comp_send_offsets[r] = send_offsets_[out_neighbors[r]];
    }
    comp_send_offsets.back() = send_offsets_.back();

    recv_sizes_ = std::move(in_weight);
    recv_offsets_ = std::move(comp_recv_offsets);
    send_sizes_ = std::move(out_weight);
    send_offsets_ = std::move(comp_send_offsets);
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
void Matrix<ValueType, LocalIndexType, GlobalIndexType>::read_distributed(
    const device_matrix_data<value_type, global_index_type>& data,
    ptr_param<const Partition<local_index_type, global_index_type>>
        row_partition,
    ptr_param<const Partition<local_index_type, global_index_type>>
        col_partition,
    bool disable_sparse_comm)
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
        data, make_temporary_clone(exec, row_partition).get(),
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

    // initialize send and recv data
    recv_sizes_.resize(comm.size());
    recv_offsets_.resize(comm.size() + 1);
    send_sizes_.resize(comm.size());
    send_offsets_.resize(comm.size() + 1);

    // exchange step 1: determine recv_sizes, send_sizes, send_offsets
    exec->get_master()->copy_from(
        exec, num_parts, recv_sizes_array.get_const_data(), recv_sizes_.data());
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

    if (disable_sparse_comm) {
        neighbor_comm_ = std::nullopt;
    } else {
        this->enable_neighborhood_communication();
    }
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
void Matrix<ValueType, LocalIndexType, GlobalIndexType>::read_distributed(
    const matrix_data<value_type, global_index_type>& data,
    ptr_param<const Partition<local_index_type, global_index_type>>
        row_partition,
    ptr_param<const Partition<local_index_type, global_index_type>>
        col_partition,
    bool disable_sparse_comm)
{
    this->read_distributed(
        device_matrix_data<value_type, global_index_type>::create_from_host(
            this->get_executor(), data),
        row_partition, col_partition, disable_sparse_comm);
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
void Matrix<ValueType, LocalIndexType, GlobalIndexType>::read_distributed(
    const matrix_data<ValueType, global_index_type>& data,
    ptr_param<const Partition<local_index_type, global_index_type>> partition,
    bool disable_sparse_comm)
{
    this->read_distributed(
        device_matrix_data<value_type, global_index_type>::create_from_host(
            this->get_executor(), data),
        partition, partition, disable_sparse_comm);
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
void Matrix<ValueType, LocalIndexType, GlobalIndexType>::read_distributed(
    const device_matrix_data<ValueType, GlobalIndexType>& data,
    ptr_param<const Partition<local_index_type, global_index_type>> partition,
    bool disable_sparse_comm)
{
    this->read_distributed(data, partition, partition, disable_sparse_comm);
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
    if (neighbor_comm_) {
        return neighbor_comm_->i_neighor_all_to_all_v(
            use_host_buffer ? exec->get_master() : exec, send_ptr,
            send_sizes_.data(), send_offsets_.data(), type.get(), recv_ptr,
            recv_sizes_.data(), recv_offsets_.data(), type.get());
    } else {
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

            auto req = this->communicate(dense_b->get_local_vector());
            local_mtx_->apply(dense_b->get_local_vector(), local_x);
            req.wait();

            auto exec = this->get_executor();
            auto comm = this->get_communicator();
            auto use_host_buffer = mpi::requires_host_buffer(exec, comm);
            if (use_host_buffer) {
                recv_buffer_->copy_from(host_recv_buffer_.get());
            }
            non_local_mtx_->apply(one_scalar_.get(), recv_buffer_.get(),
                                  one_scalar_.get(), local_x);
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

            auto req = this->communicate(dense_b->get_local_vector());
            local_mtx_->apply(local_alpha, dense_b->get_local_vector(),
                              local_beta, local_x);
            req.wait();

            auto exec = this->get_executor();
            auto comm = this->get_communicator();
            auto use_host_buffer = mpi::requires_host_buffer(exec, comm);
            if (use_host_buffer) {
                recv_buffer_->copy_from(host_recv_buffer_.get());
            }
            non_local_mtx_->apply(local_alpha, recv_buffer_.get(),
                                  one_scalar_.get(), local_x);
        },
        alpha, b, beta, x);
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
Matrix<ValueType, LocalIndexType, GlobalIndexType>::Matrix(const Matrix& other)
    : EnableDistributedLinOp<Matrix<value_type, local_index_type,
                                    global_index_type>>{other.get_executor()},
      DistributedBase{other.get_communicator()},
      neighbor_comm_(other.get_communicator())
{
    *this = other;
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
Matrix<ValueType, LocalIndexType, GlobalIndexType>::Matrix(
    Matrix&& other) noexcept
    : EnableDistributedLinOp<Matrix<value_type, local_index_type,
                                    global_index_type>>{other.get_executor()},
      DistributedBase{other.get_communicator()},
      neighbor_comm_(other.get_communicator())
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
        local_mtx_->copy_from(other.local_mtx_);
        non_local_mtx_->copy_from(other.non_local_mtx_);
        gather_idxs_ = other.gather_idxs_;
        send_offsets_ = other.send_offsets_;
        recv_offsets_ = other.recv_offsets_;
        send_sizes_ = other.send_sizes_;
        recv_sizes_ = other.recv_sizes_;
        non_local_to_global_ = other.non_local_to_global_;
        one_scalar_.init(this->get_executor(), dim<2>{1, 1});
        one_scalar_->fill(one<value_type>());
        if (other.neighbor_comm_) {
            neighbor_comm_ = other.neighbor_comm_->duplicate();
        } else {
            neighbor_comm_.reset();
        }
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
        local_mtx_->move_from(other.local_mtx_);
        non_local_mtx_->move_from(other.non_local_mtx_);
        gather_idxs_ = std::move(other.gather_idxs_);
        send_offsets_ = std::move(other.send_offsets_);
        recv_offsets_ = std::move(other.recv_offsets_);
        send_sizes_ = std::move(other.send_sizes_);
        recv_sizes_ = std::move(other.recv_sizes_);
        non_local_to_global_ = std::move(other.non_local_to_global_);
        one_scalar_.init(this->get_executor(), dim<2>{1, 1});
        one_scalar_->fill(one<value_type>());
        neighbor_comm_ = std::exchange(other.neighbor_comm_, std::nullopt);
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
