// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/distributed/matrix.hpp>


#include <set>


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
GKO_REGISTER_OPERATION(separate_local_nonlocal,
                       distributed_matrix::separate_local_nonlocal);


}  // namespace
}  // namespace matrix


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
      one_scalar_{},
      local_mtx_{local_matrix_template->clone(exec)},
      non_local_mtx_{non_local_matrix_template->clone(exec)},
      sparse_comm_(sparse_communicator::create(comm))
{
    GKO_ASSERT(
        (dynamic_cast<ReadableFromMatrixData<ValueType, LocalIndexType>*>(
            local_mtx_.get())));
    GKO_ASSERT(
        (dynamic_cast<ReadableFromMatrixData<ValueType, LocalIndexType>*>(
            non_local_mtx_.get())));
    one_scalar_.init(exec, dim<2>{1, 1});
    one_scalar_->fill(one<value_type>());
    recv_buffer_.init(exec, {}, nullptr);
    send_buffer_.init(exec, {}, nullptr);
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
Matrix<ValueType, LocalIndexType, GlobalIndexType>::Matrix(
    std::shared_ptr<const Executor> exec,
    std::shared_ptr<const sparse_communicator> sparse_comm,
    std::unique_ptr<LinOp> local_matrix,
    std::unique_ptr<LinOp> non_local_matrix)
    : EnableDistributedLinOp<
          Matrix<value_type, local_index_type, global_index_type>>{exec},
      DistributedBase{sparse_comm->get_communicator()},
      one_scalar_{},
      local_mtx_{std::move(local_matrix)},
      non_local_mtx_{std::move(non_local_matrix)},
      sparse_comm_(sparse_comm)
{
    // auto recv_idxs =
    //     sparse_comm->get_partition<GlobalIndexType>()->get_recv_indices();
    // GKO_ASSERT(collection::get_size(recv_idxs.idxs) ==
    //            non_local_mtx_->get_size()[1]);
    recv_buffer_.init(exec, {}, nullptr);
    send_buffer_.init(exec, {}, nullptr);
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
    result->sparse_comm_ = this->sparse_comm_;
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
    result->sparse_comm_ = std::move(this->sparse_comm_);
    result->set_size(this->get_size());
    this->set_size({});
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
void Matrix<ValueType, LocalIndexType, GlobalIndexType>::read_distributed(
    const device_matrix_data<value_type, local_index_type>& local_data,
    const device_matrix_data<value_type, local_index_type>& non_local_data,
    std::shared_ptr<const sparse_communicator> sparse_comm)
{
    sparse_comm_ = std::move(sparse_comm);
    const auto comm = sparse_comm_->get_communicator();
    auto exec = this->get_executor();
    // this is a partition of the column space
    // auto part = sparse_comm_->get_<LocalIndexType>();
    // GKO_ASSERT_EQUAL_ROWS(local_data.get_size(), non_local_data.get_size());
    // GKO_ASSERT_EQ(local_data.get_size()[1], part->get_local_end());
    // GKO_ASSERT_EQ(non_local_data.get_size()[1],
    //               collection::get_size(part->get_recv_indices().idxs));

    as<ReadableFromMatrixData<ValueType, LocalIndexType>>(local_mtx_)
        ->read(std::move(local_data));
    as<ReadableFromMatrixData<ValueType, LocalIndexType>>(non_local_mtx_)
        ->read(std::move(non_local_data));

    auto num_rows = local_mtx_->get_size()[0];
    auto num_cols = local_mtx_->get_size()[1];
    comm.all_reduce(exec, &num_rows, 1, MPI_SUM);
    comm.all_reduce(exec, &num_cols, 1, MPI_SUM);
    this->set_size({num_rows, num_cols});
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
index_map<LocalIndexType, GlobalIndexType>
Matrix<ValueType, LocalIndexType, GlobalIndexType>::read_distributed(
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
    array<global_index_type> global_non_local_col_idxs{exec};
    array<value_type> non_local_values{exec};
    array<local_index_type> recv_gather_idxs{exec};
    array<comm_index_type> recv_sizes_array{exec, num_parts};

    // build local, non-local matrix data and communication structures
    // exec->run(matrix::make_build_local_nonlocal(
    //     data, make_temporary_clone(exec, row_partition).get(),
    //     make_temporary_clone(exec, col_partition).get(), local_part,
    //     local_row_idxs, local_col_idxs, local_values, non_local_row_idxs,
    //     non_local_col_idxs, non_local_values, recv_gather_idxs,
    //     recv_sizes_array, non_local_to_global_));

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
        exec, comm, col_partition, global_non_local_col_idxs);

    auto non_local_col_idxs = imap.get_non_local(global_non_local_col_idxs);

    // read the local matrix data
    const auto num_local_rows =
        static_cast<size_type>(row_partition->get_part_size(local_part));
    const auto num_local_cols =
        static_cast<size_type>(col_partition->get_part_size(local_part));
    device_matrix_data<value_type, local_index_type> local_data{
        exec, dim<2>{num_local_rows, num_local_cols}, std::move(local_row_idxs),
        std::move(local_col_idxs), std::move(local_values)};
    device_matrix_data<value_type, local_index_type> non_local_data{
        exec,
        dim<2>{num_local_rows,
               imap.get_remote_global_idxs().get_flat().get_size()},
        std::move(non_local_row_idxs), std::move(non_local_col_idxs),
        std::move(non_local_values)};
    as<ReadableFromMatrixData<ValueType, LocalIndexType>>(this->local_mtx_)
        ->read(std::move(local_data));
    as<ReadableFromMatrixData<ValueType, LocalIndexType>>(this->non_local_mtx_)
        ->read(std::move(non_local_data));

    sparse_comm_ = sparse_communicator::create(comm, imap);

    return imap;
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
index_map<LocalIndexType, GlobalIndexType>
Matrix<ValueType, LocalIndexType, GlobalIndexType>::read_distributed(
    const matrix_data<value_type, global_index_type>& data,
    std::shared_ptr<const Partition<local_index_type, global_index_type>>
        row_partition,
    std::shared_ptr<const Partition<local_index_type, global_index_type>>
        col_partition)
{
    return this->read_distributed(
        device_matrix_data<value_type, global_index_type>::create_from_host(
            this->get_executor(), data),
        row_partition, col_partition);
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
index_map<LocalIndexType, GlobalIndexType>
Matrix<ValueType, LocalIndexType, GlobalIndexType>::read_distributed(
    const matrix_data<ValueType, global_index_type>& data,
    std::shared_ptr<const Partition<local_index_type, global_index_type>>
        partition)
{
    return this->read_distributed(
        device_matrix_data<value_type, global_index_type>::create_from_host(
            this->get_executor(), data),
        partition, partition);
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
index_map<LocalIndexType, GlobalIndexType>
Matrix<ValueType, LocalIndexType, GlobalIndexType>::read_distributed(
    const device_matrix_data<ValueType, GlobalIndexType>& data,
    std::shared_ptr<const Partition<local_index_type, global_index_type>>
        partition)
{
    return this->read_distributed(data, partition, partition);
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

            auto exec = this->get_executor();
            auto use_host_buffer = mpi::requires_host_buffer(exec, comm);

            auto req = [&] {
                if (use_host_buffer) {
                    return sparse_comm_->communicate(
                        dense_b->get_local_vector(), host_send_buffer_,
                        host_recv_buffer_);
                } else {
                    return sparse_comm_->communicate(
                        dense_b->get_local_vector(), send_buffer_,
                        recv_buffer_);
                }
            }();
            local_mtx_->apply(dense_b->get_local_vector(), local_x);
            req.wait();

            if (use_host_buffer) {
                recv_buffer_.init_from(host_recv_buffer_.get());
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

            auto exec = this->get_executor();
            auto use_host_buffer = mpi::requires_host_buffer(exec, comm);

            auto req = [&] {
                if (use_host_buffer) {
                    return sparse_comm_->communicate(
                        dense_b->get_local_vector(), host_send_buffer_,
                        host_recv_buffer_);
                } else {
                    return sparse_comm_->communicate(
                        dense_b->get_local_vector(), send_buffer_,
                        recv_buffer_);
                }
            }();
            local_mtx_->apply(local_alpha, dense_b->get_local_vector(),
                              local_beta, local_x);
            req.wait();

            if (use_host_buffer) {
                recv_buffer_.init_from(host_recv_buffer_.get());
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
      DistributedBase{other.get_communicator()}
{
    *this = other;
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
Matrix<ValueType, LocalIndexType, GlobalIndexType>::Matrix(
    Matrix&& other) noexcept
    : EnableDistributedLinOp<Matrix<value_type, local_index_type,
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
        sparse_comm_ = other.sparse_comm_;
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
        sparse_comm_ = std::move(other.sparse_comm_);
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
