// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "ginkgo/core/distributed/matrix.hpp"

#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/precision_dispatch.hpp>
#include <ginkgo/core/distributed/assembly.hpp>
#include <ginkgo/core/distributed/vector.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/diagonal.hpp>

#include "core/components/prefix_sum_kernels.hpp"
#include "core/distributed/matrix_kernels.hpp"


namespace gko {
namespace experimental {
namespace distributed {
namespace matrix {
namespace {


GKO_REGISTER_OPERATION(separate_local_nonlocal,
                       distributed_matrix::separate_local_nonlocal);


}  // namespace
}  // namespace matrix


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
Matrix<ValueType, LocalIndexType, GlobalIndexType>::Matrix(
    std::shared_ptr<const Executor> exec, mpi::communicator comm)
    : Matrix(exec, comm,
             gko::matrix::Csr<ValueType, LocalIndexType>::create(exec),
             gko::matrix::Csr<ValueType, LocalIndexType>::create(exec))
{}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
Matrix<ValueType, LocalIndexType, GlobalIndexType>::Matrix(
    std::shared_ptr<const Executor> exec, mpi::communicator comm,
    ptr_param<const LinOp> local_matrix_template,
    ptr_param<const LinOp> non_local_matrix_template)
    : EnableLinOp<Matrix>{exec},
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
Matrix<ValueType, LocalIndexType, GlobalIndexType>::Matrix(
    std::shared_ptr<const Executor> exec, mpi::communicator comm, dim<2> size,
    std::shared_ptr<LinOp> local_linop)
    : EnableLinOp<Matrix>{exec},
      DistributedBase{comm},
      send_offsets_(comm.size() + 1),
      send_sizes_(comm.size()),
      recv_offsets_(comm.size() + 1),
      recv_sizes_(comm.size()),
      gather_idxs_{exec},
      non_local_to_global_{exec},
      one_scalar_{},
      non_local_mtx_(::gko::matrix::Coo<ValueType, LocalIndexType>::create(
          exec, dim<2>{local_linop->get_size()[0], 0}))
{
    this->set_size(size);
    one_scalar_.init(exec, dim<2>{1, 1});
    one_scalar_->fill(one<value_type>());
    local_mtx_ = std::move(local_linop);
}

template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
Matrix<ValueType, LocalIndexType, GlobalIndexType>::Matrix(
    std::shared_ptr<const Executor> exec, mpi::communicator comm, dim<2> size,
    std::shared_ptr<LinOp> local_linop, std::shared_ptr<LinOp> non_local_linop,
    std::vector<comm_index_type> recv_sizes,
    std::vector<comm_index_type> recv_offsets,
    array<local_index_type> recv_gather_idxs)
    : EnableLinOp<Matrix>{exec},
      DistributedBase{comm},
      send_offsets_(comm.size() + 1),
      send_sizes_(comm.size()),
      recv_offsets_(comm.size() + 1),
      recv_sizes_(comm.size()),
      gather_idxs_{exec},
      non_local_to_global_{exec},
      one_scalar_{}
{
    this->set_size(size);
    local_mtx_ = std::move(local_linop);
    non_local_mtx_ = std::move(non_local_linop);
    recv_offsets_ = std::move(recv_offsets);
    recv_sizes_ = std::move(recv_sizes);
    // build send information from recv copy
    // exchange step 1: determine recv_sizes, send_sizes, send_offsets
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

    one_scalar_.init(exec, dim<2>{1, 1});
    one_scalar_->fill(one<value_type>());
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
std::unique_ptr<Matrix<ValueType, LocalIndexType, GlobalIndexType>>
Matrix<ValueType, LocalIndexType, GlobalIndexType>::create(
    std::shared_ptr<const Executor> exec, mpi::communicator comm)
{
    return std::unique_ptr<Matrix>{new Matrix{exec, comm}};
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
std::unique_ptr<Matrix<ValueType, LocalIndexType, GlobalIndexType>>
Matrix<ValueType, LocalIndexType, GlobalIndexType>::create(
    std::shared_ptr<const Executor> exec, mpi::communicator comm,
    ptr_param<const LinOp> matrix_template)
{
    return create(exec, comm, matrix_template, matrix_template);
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
std::unique_ptr<Matrix<ValueType, LocalIndexType, GlobalIndexType>>
Matrix<ValueType, LocalIndexType, GlobalIndexType>::create(
    std::shared_ptr<const Executor> exec, mpi::communicator comm,
    ptr_param<const LinOp> local_matrix_template,
    ptr_param<const LinOp> non_local_matrix_template)
{
    return std::unique_ptr<Matrix>{new Matrix{exec, comm, local_matrix_template,
                                              non_local_matrix_template}};
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
std::unique_ptr<Matrix<ValueType, LocalIndexType, GlobalIndexType>>
Matrix<ValueType, LocalIndexType, GlobalIndexType>::create(
    std::shared_ptr<const Executor> exec, mpi::communicator comm, dim<2> size,
    std::shared_ptr<LinOp> local_linop)
{
    return std::unique_ptr<Matrix>{new Matrix{exec, comm, size, local_linop}};
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
std::unique_ptr<Matrix<ValueType, LocalIndexType, GlobalIndexType>>
Matrix<ValueType, LocalIndexType, GlobalIndexType>::create(
    std::shared_ptr<const Executor> exec, mpi::communicator comm, dim<2> size,
    std::shared_ptr<LinOp> local_linop, std::shared_ptr<LinOp> non_local_linop,
    std::vector<comm_index_type> recv_sizes,
    std::vector<comm_index_type> recv_offsets,
    array<local_index_type> recv_gather_idxs)
{
    return std::unique_ptr<Matrix>{new Matrix{
        exec, comm, size, local_linop, non_local_linop, std::move(recv_sizes),
        std::move(recv_offsets), std::move(recv_gather_idxs)}};
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
void Matrix<ValueType, LocalIndexType, GlobalIndexType>::convert_to(
    Matrix<next_precision_base<value_type>, local_index_type,
           global_index_type>* result) const
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
    Matrix<next_precision_base<value_type>, local_index_type,
           global_index_type>* result)
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
void Matrix<ValueType, LocalIndexType, GlobalIndexType>::read_distributed(
    const device_matrix_data<value_type, global_index_type>& data,
    std::shared_ptr<const Partition<local_index_type, global_index_type>>
        row_partition,
    std::shared_ptr<const Partition<local_index_type, global_index_type>>
        col_partition,
    assembly_mode assembly_type)
{
    const auto comm = this->get_communicator();
    GKO_ASSERT_EQ(data.get_size()[0], row_partition->get_size());
    GKO_ASSERT_EQ(data.get_size()[1], col_partition->get_size());
    GKO_ASSERT_EQ(comm.size(), row_partition->get_num_parts());
    GKO_ASSERT_EQ(comm.size(), col_partition->get_num_parts());
    auto exec = this->get_executor();
    auto local_part = comm.rank();
    auto use_host_buffer = mpi::requires_host_buffer(exec, comm);
    auto tmp_row_partition = make_temporary_clone(exec, row_partition);
    auto tmp_col_partition = make_temporary_clone(exec, col_partition);

    const device_matrix_data<value_type, global_index_type>* all_data_ptr =
        &data;
    device_matrix_data<value_type, global_index_type> assembled_data(exec);
    if (assembly_type == assembly_mode::communicate) {
        assembled_data = assemble_rows_from_neighbors<ValueType, LocalIndexType,
                                                      GlobalIndexType>(
            this->get_communicator(), data, row_partition);
        all_data_ptr = &assembled_data;
    }

    // set up LinOp sizes
    auto global_num_rows = row_partition->get_size();
    auto global_num_cols = col_partition->get_size();
    dim<2> global_dim{global_num_rows, global_num_cols};
    this->set_size(global_dim);

    // temporary storage for the output
    array<local_index_type> local_row_idxs{exec};
    array<local_index_type> local_col_idxs{exec};
    array<value_type> local_values{exec};
    array<local_index_type> non_local_row_idxs{exec};
    array<global_index_type> global_non_local_col_idxs{exec};
    array<value_type> non_local_values{exec};

    // separate input into local and non-local block
    // The rows and columns of the local block are mapped into local indexing,
    // as well as the rows of the non-local block. The columns of the non-local
    // block are still in global indices.
    exec->run(matrix::make_separate_local_nonlocal(
        *all_data_ptr, tmp_row_partition.get(), tmp_col_partition.get(),
        local_part, local_row_idxs, local_col_idxs, local_values,
        non_local_row_idxs, global_non_local_col_idxs, non_local_values));

    auto imap = index_map<local_index_type, global_index_type>(
        exec, col_partition, comm.rank(), global_non_local_col_idxs);

    auto non_local_col_idxs =
        imap.map_to_local(global_non_local_col_idxs, index_space::non_local);
    non_local_to_global_ =
        make_const_array_view(
            imap.get_executor(), imap.get_remote_global_idxs().get_size(),
            imap.get_remote_global_idxs().get_const_flat_data())
            .copy_to_array();

    // read the local matrix data
    const auto num_local_rows =
        static_cast<size_type>(row_partition->get_part_size(local_part));
    const auto num_local_cols =
        static_cast<size_type>(col_partition->get_part_size(local_part));
    device_matrix_data<value_type, local_index_type> local_data{
        exec, dim<2>{num_local_rows, num_local_cols}, std::move(local_row_idxs),
        std::move(local_col_idxs), std::move(local_values)};
    device_matrix_data<value_type, local_index_type> non_local_data{
        exec, dim<2>{num_local_rows, imap.get_remote_global_idxs().get_size()},
        std::move(non_local_row_idxs), std::move(non_local_col_idxs),
        std::move(non_local_values)};
    as<ReadableFromMatrixData<ValueType, LocalIndexType>>(this->local_mtx_)
        ->read(std::move(local_data));
    as<ReadableFromMatrixData<ValueType, LocalIndexType>>(this->non_local_mtx_)
        ->read(std::move(non_local_data));

    // exchange step 1: determine recv_sizes, send_sizes, send_offsets
    auto host_recv_targets =
        make_temporary_clone(exec->get_master(), &imap.get_remote_target_ids());
    auto host_offsets = make_temporary_clone(
        exec->get_master(), &imap.get_remote_global_idxs().get_offsets());
    auto compute_recv_sizes = [](const auto* recv_targets, size_type size,
                                 const auto* offsets, auto& recv_sizes) {
        for (size_type i = 0; i < size; ++i) {
            recv_sizes[recv_targets[i]] = offsets[i + 1] - offsets[i];
        }
    };
    std::fill(recv_sizes_.begin(), recv_sizes_.end(), 0);
    compute_recv_sizes(host_recv_targets->get_const_data(),
                       host_recv_targets->get_size(),
                       host_offsets->get_const_data(), recv_sizes_);
    std::partial_sum(recv_sizes_.begin(), recv_sizes_.end(),
                     recv_offsets_.begin() + 1);
    comm.all_to_all(exec, recv_sizes_.data(), 1, send_sizes_.data(), 1);
    std::partial_sum(send_sizes_.begin(), send_sizes_.end(),
                     send_offsets_.begin() + 1);
    send_offsets_[0] = 0;
    recv_offsets_[0] = 0;

    // exchange step 2: exchange gather_idxs from receivers to senders
    auto recv_gather_idxs =
        make_const_array_view(
            imap.get_executor(), imap.get_non_local_size(),
            imap.get_remote_local_idxs().get_const_flat_data())
            .copy_to_array();
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
    std::shared_ptr<const Partition<local_index_type, global_index_type>>
        row_partition,
    std::shared_ptr<const Partition<local_index_type, global_index_type>>
        col_partition,
    assembly_mode assembly_type)
{
    return this->read_distributed(
        device_matrix_data<value_type, global_index_type>::create_from_host(
            this->get_executor(), data),
        row_partition, col_partition, assembly_type);
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
void Matrix<ValueType, LocalIndexType, GlobalIndexType>::read_distributed(
    const matrix_data<ValueType, global_index_type>& data,
    std::shared_ptr<const Partition<local_index_type, global_index_type>>
        partition,
    assembly_mode assembly_type)
{
    return this->read_distributed(
        device_matrix_data<value_type, global_index_type>::create_from_host(
            this->get_executor(), data),
        partition, partition, assembly_type);
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
void Matrix<ValueType, LocalIndexType, GlobalIndexType>::read_distributed(
    const device_matrix_data<value_type, global_index_type>& data,
    std::shared_ptr<const Partition<local_index_type, global_index_type>>
        partition,
    assembly_mode assembly_type)
{
    return this->read_distributed(data, partition, partition, assembly_type);
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
mpi::request Matrix<ValueType, LocalIndexType, GlobalIndexType>::communicate(
    const local_vector_type* local_b) const
{
    // This function can never return early!
    // Even if the non-local part is empty, i.e. this process doesn't need
    // any data from other processes, the used MPI calls are collective
    // operations. They need to be called on all processes, even if a process
    // might not communicate any data.
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
            local_mtx_->apply(dense_b->get_local_vector(), local_x);
            req.wait();

            auto exec = this->get_executor();
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

            auto comm = this->get_communicator();
            auto req = this->communicate(dense_b->get_local_vector());
            local_mtx_->apply(local_alpha, dense_b->get_local_vector(),
                              local_beta, local_x);
            req.wait();

            auto exec = this->get_executor();
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
void Matrix<ValueType, LocalIndexType, GlobalIndexType>::col_scale(
    ptr_param<const global_vector_type> scaling_factors)
{
    GKO_ASSERT_CONFORMANT(this, scaling_factors.get());
    GKO_ASSERT_EQ(scaling_factors->get_size()[1], 1);
    auto exec = this->get_executor();
    auto comm = this->get_communicator();
    size_type n_local_cols = local_mtx_->get_size()[1];
    size_type n_non_local_cols = non_local_mtx_->get_size()[1];
    std::unique_ptr<global_vector_type> scaling_factors_single_stride;
    auto stride = scaling_factors->get_stride();
    if (stride != 1) {
        scaling_factors_single_stride = global_vector_type::create(exec, comm);
        scaling_factors_single_stride->copy_from(scaling_factors.get());
    }
    const auto scale_values =
        stride == 1 ? scaling_factors->get_const_local_values()
                    : scaling_factors_single_stride->get_const_local_values();
    const auto scale_diag = gko::matrix::Diagonal<ValueType>::create_const(
        exec, n_local_cols,
        make_const_array_view(exec, n_local_cols, scale_values));

    auto req = this->communicate(
        stride == 1 ? scaling_factors->get_local_vector()
                    : scaling_factors_single_stride->get_local_vector());
    scale_diag->rapply(local_mtx_, local_mtx_);
    req.wait();
    if (n_non_local_cols > 0) {
        auto use_host_buffer = mpi::requires_host_buffer(exec, comm);
        if (use_host_buffer) {
            recv_buffer_->copy_from(host_recv_buffer_.get());
        }
        const auto non_local_scale_diag =
            gko::matrix::Diagonal<ValueType>::create_const(
                exec, n_non_local_cols,
                make_const_array_view(exec, n_non_local_cols,
                                      recv_buffer_->get_const_values()));
        non_local_scale_diag->rapply(non_local_mtx_, non_local_mtx_);
    }
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
void Matrix<ValueType, LocalIndexType, GlobalIndexType>::row_scale(
    ptr_param<const global_vector_type> scaling_factors)
{
    GKO_ASSERT_EQUAL_ROWS(this, scaling_factors.get());
    GKO_ASSERT_EQ(scaling_factors->get_size()[1], 1);
    auto exec = this->get_executor();
    auto comm = this->get_communicator();
    size_type n_local_rows = local_mtx_->get_size()[0];
    std::unique_ptr<global_vector_type> scaling_factors_single_stride;
    auto stride = scaling_factors->get_stride();
    if (stride != 1) {
        scaling_factors_single_stride = global_vector_type::create(exec, comm);
        scaling_factors_single_stride->copy_from(scaling_factors.get());
    }
    const auto scale_values =
        stride == 1 ? scaling_factors->get_const_local_values()
                    : scaling_factors_single_stride->get_const_local_values();
    const auto scale_diag = gko::matrix::Diagonal<ValueType>::create_const(
        exec, n_local_rows,
        make_const_array_view(exec, n_local_rows, scale_values));

    scale_diag->apply(local_mtx_, local_mtx_);
    scale_diag->apply(non_local_mtx_, non_local_mtx_);
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
    }
    return *this;
}


#define GKO_DECLARE_DISTRIBUTED_MATRIX(ValueType, LocalIndexType, \
                                       GlobalIndexType)           \
    class Matrix<ValueType, LocalIndexType, GlobalIndexType>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_LOCAL_GLOBAL_INDEX_TYPE_BASE(
    GKO_DECLARE_DISTRIBUTED_MATRIX);


}  // namespace distributed
}  // namespace experimental
}  // namespace gko
