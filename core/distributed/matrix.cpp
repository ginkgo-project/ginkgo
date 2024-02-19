// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/distributed/matrix.hpp>


#include <ginkgo/core/base/precision_dispatch.hpp>
#include <ginkgo/core/distributed/neighborhood_communicator.hpp>
#include <ginkgo/core/distributed/vector.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/csr.hpp>


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
    : EnableDistributedLinOp<
          Matrix<value_type, local_index_type, global_index_type>>{exec},
      DistributedBase{comm},
      row_gatherer_{RowGatherer<LocalIndexType>::create(exec, comm)},
      imap_{exec},
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
    : EnableDistributedLinOp<
          Matrix<value_type, local_index_type, global_index_type>>{exec},
      DistributedBase{comm},
      row_gatherer_{RowGatherer<LocalIndexType>::create(exec, comm)},
      imap_{exec},
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
void Matrix<ValueType, LocalIndexType, GlobalIndexType>::convert_to(
    Matrix<next_precision<value_type>, local_index_type, global_index_type>*
        result) const
{
    GKO_ASSERT(this->get_communicator().size() ==
               result->get_communicator().size());
    result->local_mtx_->copy_from(this->local_mtx_);
    result->non_local_mtx_->copy_from(this->non_local_mtx_);
    result->row_gatherer_->copy_from(this->row_gatherer_);
    result->imap_ = this->imap_;
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
    result->row_gatherer_->move_from(this->row_gatherer_);
    result->imap_ = std::move(this->imap_);
    result->set_size(this->get_size());
    this->set_size({});
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
void Matrix<ValueType, LocalIndexType, GlobalIndexType>::read_distributed(
    const device_matrix_data<value_type, global_index_type>& data,
    std::shared_ptr<const Partition<local_index_type, global_index_type>>
        row_partition,
    std::shared_ptr<const Partition<local_index_type, global_index_type>>
        col_partition)
{
    const auto comm = this->get_communicator();
    GKO_ASSERT_EQ(data.get_size()[0], row_partition->get_size());
    GKO_ASSERT_EQ(data.get_size()[1], col_partition->get_size());
    GKO_ASSERT_EQ(comm.size(), row_partition->get_num_parts());
    GKO_ASSERT_EQ(comm.size(), col_partition->get_num_parts());
    auto exec = this->get_executor();
    auto local_part = comm.rank();

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
        data, make_temporary_clone(exec, row_partition).get(),
        make_temporary_clone(exec, col_partition).get(), local_part,
        local_row_idxs, local_col_idxs, local_values, non_local_row_idxs,
        global_non_local_col_idxs, non_local_values));

    auto imap = index_map<local_index_type, global_index_type>(
        exec, col_partition, comm.rank(), global_non_local_col_idxs);

    auto non_local_col_idxs =
        imap.map_to_local(global_non_local_col_idxs, index_space::non_local);

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

    row_gatherer_ = RowGatherer<local_index_type>::create(
        exec, std::make_shared<mpi::neighborhood_communicator>(comm, imap_),
        imap_);
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
void Matrix<ValueType, LocalIndexType, GlobalIndexType>::read_distributed(
    const matrix_data<value_type, global_index_type>& data,
    std::shared_ptr<const Partition<local_index_type, global_index_type>>
        row_partition,
    std::shared_ptr<const Partition<local_index_type, global_index_type>>
        col_partition)
{
    this->read_distributed(
        device_matrix_data<value_type, global_index_type>::create_from_host(
            this->get_executor(), data),
        row_partition, col_partition);
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
void Matrix<ValueType, LocalIndexType, GlobalIndexType>::read_distributed(
    const matrix_data<ValueType, global_index_type>& data,
    std::shared_ptr<const Partition<local_index_type, global_index_type>>
        partition)
{
    this->read_distributed(
        device_matrix_data<value_type, global_index_type>::create_from_host(
            this->get_executor(), data),
        partition, partition);
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
void Matrix<ValueType, LocalIndexType, GlobalIndexType>::read_distributed(
    const device_matrix_data<ValueType, GlobalIndexType>& data,
    std::shared_ptr<const Partition<local_index_type, global_index_type>>
        partition)
{
    this->read_distributed(data, partition, partition);
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

            auto exec = this->get_executor();
            auto comm = this->get_communicator();
            auto recv_dim =
                dim<2>{static_cast<size_type>(
                           row_gatherer_->get_collective_communicator()
                               ->get_recv_size()),
                       dense_b->get_size()[1]};
            auto recv_exec = mpi::requires_host_buffer(exec, comm)
                                 ? exec->get_master()
                                 : exec;
            recv_buffer_.init(recv_exec, recv_dim);
            auto req =
                this->row_gatherer_->apply_async(dense_b, recv_buffer_.get());
            local_mtx_->apply(dense_b->get_local_vector(), local_x);
            req.wait();

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

            auto exec = this->get_executor();
            auto comm = this->get_communicator();
            auto recv_dim =
                dim<2>{static_cast<size_type>(
                           row_gatherer_->get_collective_communicator()
                               ->get_recv_size()),
                       dense_b->get_size()[1]};
            auto recv_exec = mpi::requires_host_buffer(exec, comm)
                                 ? exec->get_master()
                                 : exec;
            recv_buffer_.init(recv_exec, recv_dim);
            auto req =
                this->row_gatherer_->apply_async(dense_b, recv_buffer_.get());
            local_mtx_->apply(local_alpha, dense_b->get_local_vector(),
                              local_beta, local_x);
            req.wait();

            non_local_mtx_->apply(local_alpha, recv_buffer_.get(),
                                  one_scalar_.get(), local_x);
        },
        alpha, b, beta, x);
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
std::shared_ptr<LinOp>
Matrix<ValueType, LocalIndexType,
       GlobalIndexType>::get_overlapping_local_matrix(size_type overlap)
{
    using Csr = gko::matrix::Csr<ValueType, LocalIndexType>;
    auto exec = this->get_executor();
    auto host_local =
        make_temporary_clone(exec->get_master(), as<Csr>(local_mtx_));
    auto host_non_local =
        make_temporary_clone(exec->get_master(), as<Csr>(non_local_mtx_));

    auto& send_idxs = spcomm_.get_send_idxs<LocalIndexType>();
    auto host_send_idxs = make_temporary_clone(exec->get_master(), &send_idxs);
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
Matrix<ValueType, LocalIndexType, GlobalIndexType>::Matrix(const Matrix& other)
    : EnableDistributedLinOp<Matrix<value_type, local_index_type,
                                    global_index_type>>{other.get_executor()},
      DistributedBase{other.get_communicator()},
      row_gatherer_{RowGatherer<LocalIndexType>::create(
          other.get_executor(), other.get_communicator())},
      imap_{other.get_executor()}
{
    *this = other;
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
Matrix<ValueType, LocalIndexType, GlobalIndexType>::Matrix(
    Matrix&& other) noexcept
    : EnableDistributedLinOp<Matrix<value_type, local_index_type,
                                    global_index_type>>{other.get_executor()},
      DistributedBase{other.get_communicator()},
      row_gatherer_{RowGatherer<LocalIndexType>::create(
          other.get_executor(), other.get_communicator())},
      imap_{other.get_executor()}
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
        row_gatherer_->copy_from(other.row_gatherer_);
        imap_ = other.imap_;
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
        row_gatherer_->move_from(other.row_gatherer_);
        imap_ = std::move(other.imap_);
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
