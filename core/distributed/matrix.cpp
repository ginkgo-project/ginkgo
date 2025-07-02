// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "ginkgo/core/distributed/matrix.hpp"

#include <utility>

#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/precision_dispatch.hpp>
#include <ginkgo/core/distributed/assembly.hpp>
#include <ginkgo/core/distributed/neighborhood_communicator.hpp>
#include <ginkgo/core/distributed/partition_helpers.hpp>
#include <ginkgo/core/distributed/vector.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/diagonal.hpp>

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
    : Matrix(exec, RowGatherer<LocalIndexType>::create(exec, comm),
             gko::matrix::Csr<ValueType, LocalIndexType>::create(exec),
             gko::matrix::Csr<ValueType, LocalIndexType>::create(exec))
{}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
Matrix<ValueType, LocalIndexType, GlobalIndexType>::Matrix(
    std::shared_ptr<const Executor> exec,
    std::shared_ptr<const RowGatherer<LocalIndexType>> row_gather_template,
    ptr_param<const LinOp> local_matrix_template,
    ptr_param<const LinOp> non_local_matrix_template)
    : EnableLinOp<Matrix>{exec},
      DistributedBase{row_gather_template->get_communicator()},
      row_gatherer_{row_gather_template->clone(exec)},
      imap_{exec},
      one_scalar_{exec, 1.0},
      local_mtx_{local_matrix_template->clone(exec)},
      non_local_mtx_{non_local_matrix_template->clone(exec)}
{
    GKO_ASSERT(
        (dynamic_cast<ReadableFromMatrixData<ValueType, LocalIndexType>*>(
            local_mtx_.get())));
    GKO_ASSERT(
        (dynamic_cast<ReadableFromMatrixData<ValueType, LocalIndexType>*>(
            non_local_mtx_.get())));
}

template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
Matrix<ValueType, LocalIndexType, GlobalIndexType>::Matrix(
    std::shared_ptr<const Executor> exec, mpi::communicator comm, dim<2> size,
    std::shared_ptr<LinOp> local_linop)
    : EnableLinOp<Matrix>{exec},
      DistributedBase{comm},
      row_gatherer_{RowGatherer<LocalIndexType>::create(
          exec, mpi::detail::create_default_collective_communicator(comm))},
      imap_{exec},
      one_scalar_{exec, 1.0},
      non_local_mtx_(::gko::matrix::Coo<ValueType, LocalIndexType>::create(
          exec, dim<2>{local_linop->get_size()[0], 0}))
{
    this->set_size(size);
    local_mtx_ = std::move(local_linop);
}

template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
Matrix<ValueType, LocalIndexType, GlobalIndexType>::Matrix(
    std::shared_ptr<const Executor> exec, mpi::communicator comm,
    index_map<LocalIndexType, GlobalIndexType> imap,
    std::shared_ptr<LinOp> local_linop, std::shared_ptr<LinOp> non_local_linop)
    : EnableLinOp<Matrix>{exec},
      DistributedBase{comm},
      row_gatherer_(RowGatherer<LocalIndexType>::create(
          exec,
          mpi::detail::create_default_collective_communicator(comm)
              ->create_with_same_type(comm, &imap),
          imap)),
      imap_(std::move(imap)),
      one_scalar_{exec, 1.0}
{
    this->set_size({imap_.get_global_size(), imap_.get_global_size()});
    local_mtx_ = std::move(local_linop);
    non_local_mtx_ = std::move(non_local_linop);
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
    return std::unique_ptr<Matrix>{new Matrix{
        exec,
        RowGatherer<LocalIndexType>::create(
            exec, mpi::detail::create_default_collective_communicator(comm)),
        local_matrix_template, non_local_matrix_template}};
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
    array<comm_index_type> part_ids(exec->get_master(), comm.size());
    std::iota(part_ids.get_data(), part_ids.get_data() + part_ids.get_size(),
              0);
    auto contiguous_partition =
        share(build_partition_from_local_size<LocalIndexType, GlobalIndexType>(
            exec, comm, local_linop->get_size()[0]));
    array<global_index_type> global_recv_gather_idxs(
        exec, recv_gather_idxs.get_size());
    for (int rank = 0; rank < comm.size(); ++rank) {
        if (recv_sizes[rank] > 0) {
            auto map = index_map<LocalIndexType, GlobalIndexType>(
                exec, contiguous_partition, rank, array<GlobalIndexType>{exec});
            auto local_view = make_array_view(
                exec, recv_sizes[rank],
                recv_gather_idxs.get_data() + recv_offsets[rank]);
            auto global_idxs =
                map.map_to_global(local_view, index_space::local);
            exec->copy(recv_sizes[rank], global_idxs.get_const_data(),
                       global_recv_gather_idxs.get_data() + recv_offsets[rank]);
        }
    }

    return Matrix::create(
        exec, comm,
        index_map<LocalIndexType, GlobalIndexType>(
            exec, contiguous_partition, comm.rank(), global_recv_gather_idxs),
        std::move(local_linop), std::move(non_local_linop));
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
std::unique_ptr<Matrix<ValueType, LocalIndexType, GlobalIndexType>>
Matrix<ValueType, LocalIndexType, GlobalIndexType>::create(
    std::shared_ptr<const Executor> exec, mpi::communicator comm,
    index_map<LocalIndexType, GlobalIndexType> imap,
    std::shared_ptr<LinOp> local_linop, std::shared_ptr<LinOp> non_local_linop)
{
    return std::unique_ptr<Matrix>{
        new Matrix{std::move(exec), comm, std::move(imap),
                   std::move(local_linop), std::move(non_local_linop)}};
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


#if GINKGO_ENABLE_HALF || GINKGO_ENABLE_BFLOAT16
template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
void Matrix<ValueType, LocalIndexType, GlobalIndexType>::convert_to(
    Matrix<next_precision<value_type, 2>, local_index_type, global_index_type>*
        result) const
{
    GKO_ASSERT(this->get_communicator().size() ==
               result->get_communicator().size());
    result->local_mtx_->copy_from(this->local_mtx_.get());
    result->non_local_mtx_->copy_from(this->non_local_mtx_.get());
    result->row_gatherer_->copy_from(this->row_gatherer_);
    result->imap_ = this->imap_;
    result->set_size(this->get_size());
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
void Matrix<ValueType, LocalIndexType, GlobalIndexType>::move_to(
    Matrix<next_precision<value_type, 2>, local_index_type, global_index_type>*
        result)
{
    GKO_ASSERT(this->get_communicator().size() ==
               result->get_communicator().size());
    result->local_mtx_->move_from(this->local_mtx_.get());
    result->non_local_mtx_->move_from(this->non_local_mtx_.get());
    result->row_gatherer_->move_from(this->row_gatherer_);
    result->imap_ = std::move(this->imap_);
    result->set_size(this->get_size());
    this->set_size({});
}
#endif


#if GINKGO_ENABLE_HALF && GINKGO_ENABLE_BFLOAT16
template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
void Matrix<ValueType, LocalIndexType, GlobalIndexType>::convert_to(
    Matrix<next_precision<value_type, 3>, local_index_type, global_index_type>*
        result) const
{
    GKO_ASSERT(this->get_communicator().size() ==
               result->get_communicator().size());
    result->local_mtx_->copy_from(this->local_mtx_.get());
    result->non_local_mtx_->copy_from(this->non_local_mtx_.get());
    result->row_gatherer_->copy_from(this->row_gatherer_);
    result->imap_ = this->imap_;
    result->set_size(this->get_size());
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
void Matrix<ValueType, LocalIndexType, GlobalIndexType>::move_to(
    Matrix<next_precision<value_type, 3>, local_index_type, global_index_type>*
        result)
{
    GKO_ASSERT(this->get_communicator().size() ==
               result->get_communicator().size());
    result->local_mtx_->move_from(this->local_mtx_.get());
    result->non_local_mtx_->move_from(this->non_local_mtx_.get());
    result->row_gatherer_->move_from(this->row_gatherer_);
    result->imap_ = std::move(this->imap_);
    result->set_size(this->get_size());
    this->set_size({});
}
#endif


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

    imap_ = index_map<local_index_type, global_index_type>(
        exec, col_partition, comm.rank(), global_non_local_col_idxs);

    auto non_local_col_idxs =
        imap_.map_to_local(global_non_local_col_idxs, index_space::non_local);

    // read the local matrix data
    const auto num_local_rows =
        static_cast<size_type>(row_partition->get_part_size(local_part));
    const auto num_local_cols =
        static_cast<size_type>(col_partition->get_part_size(local_part));
    device_matrix_data<value_type, local_index_type> local_data{
        exec, dim<2>{num_local_rows, num_local_cols}, std::move(local_row_idxs),
        std::move(local_col_idxs), std::move(local_values)};
    device_matrix_data<value_type, local_index_type> non_local_data{
        exec, dim<2>{num_local_rows, imap_.get_remote_global_idxs().get_size()},
        std::move(non_local_row_idxs), std::move(non_local_col_idxs),
        std::move(non_local_values)};
    as<ReadableFromMatrixData<ValueType, LocalIndexType>>(this->local_mtx_)
        ->read(std::move(local_data));
    as<ReadableFromMatrixData<ValueType, LocalIndexType>>(this->non_local_mtx_)
        ->read(std::move(non_local_data));

    row_gatherer_ = RowGatherer<LocalIndexType>::create(
        row_gatherer_->get_executor(),
        row_gatherer_->get_collective_communicator()->create_with_same_type(
            comm, &imap_),
        imap_);
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


template <typename ValueType, typename LocalIndexType>
std::pair<std::shared_ptr<Vector<ValueType>>,
          std::shared_ptr<Vector<ValueType>>>
init_recv_buffers(std::shared_ptr<const Executor> exec,
                  const RowGatherer<LocalIndexType>* row_gatherer,
                  size_type num_cols, const detail::GenericVectorCache& buffer,
                  const detail::GenericVectorCache& host_buffer)
{
    auto comm =
        row_gatherer->get_collective_communicator()->get_base_communicator();
    auto global_recv_dim =
        dim<2>{static_cast<size_type>(row_gatherer->get_size()[0]), num_cols};
    auto local_recv_dim = dim<2>{
        static_cast<size_type>(
            row_gatherer->get_collective_communicator()->get_recv_size()),
        num_cols};

    auto vector = buffer.template get<ValueType>(exec, comm, global_recv_dim,
                                                 local_recv_dim);
    auto host_vector = host_buffer.template get<ValueType>(
        exec->get_master(), comm, global_recv_dim, local_recv_dim);
    return std::make_pair(vector, host_vector);
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
void Matrix<ValueType, LocalIndexType, GlobalIndexType>::apply_impl(
    const LinOp* b, LinOp* x) const
{
    distributed::mixed_precision_dispatch_real_complex<ValueType>(
        [this](const auto dense_b, auto dense_x) {
            using x_value_type =
                typename std::decay_t<decltype(*dense_x)>::value_type;
            using b_value_type =
                typename std::decay_t<decltype(*dense_b)>::value_type;
            auto x_exec = dense_x->get_executor();
            auto local_x = gko::matrix::Dense<x_value_type>::create(
                x_exec, dense_x->get_local_vector()->get_size(),
                gko::make_array_view(
                    x_exec,
                    dense_x->get_local_vector()->get_num_stored_elements(),
                    dense_x->get_local_values()),
                dense_x->get_local_vector()->get_stride());

            auto exec = this->get_executor();
            auto comm = this->get_communicator();
            auto [recv_vector, host_recv_vector] =
                init_recv_buffers<b_value_type>(
                    exec, row_gatherer_.get(), dense_b->get_size()[1],
                    recv_buffer_, host_recv_buffer_);
            auto recv_ptr = mpi::requires_host_buffer(exec, comm)
                                ? host_recv_vector.get()
                                : recv_vector.get();
            auto ev = this->row_gatherer_->apply_prepare(dense_b, recv_ptr);
            local_mtx_->apply(dense_b->get_local_vector(), local_x);
            auto req =
                this->row_gatherer_->apply_finalize(dense_b, recv_ptr, ev);
            req.wait();

            if (recv_ptr != recv_vector.get()) {
                recv_vector->copy_from(host_recv_vector);
            }
            if (auto coo = std::dynamic_pointer_cast<
                    const ::gko::matrix::Coo<ValueType, LocalIndexType>>(
                    non_local_mtx_)) {
                coo->apply2(recv_vector->get_local_vector(), local_x);
            } else {
                non_local_mtx_->apply(
                    one_scalar_.template get<ValueType>().get(),
                    recv_vector->get_local_vector(),
                    one_scalar_.template get<x_value_type>().get(), local_x);
            }
        },
        b, x);
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
void Matrix<ValueType, LocalIndexType, GlobalIndexType>::apply_impl(
    const LinOp* alpha, const LinOp* b, const LinOp* beta, LinOp* x) const
{
    distributed::mixed_precision_dispatch_real_complex<ValueType>(
        [this, alpha, beta](const auto dense_b, auto dense_x) {
            using x_value_type =
                typename std::decay_t<decltype(*dense_x)>::value_type;
            using b_value_type =
                typename std::decay_t<decltype(*dense_b)>::value_type;
            const auto x_exec = dense_x->get_executor();
            auto local_alpha = gko::make_temporary_conversion<ValueType>(alpha);
            auto local_beta =
                gko::make_temporary_conversion<x_value_type>(beta);
            auto local_x = gko::matrix::Dense<x_value_type>::create(
                x_exec, dense_x->get_local_vector()->get_size(),
                gko::make_array_view(
                    x_exec,
                    dense_x->get_local_vector()->get_num_stored_elements(),
                    dense_x->get_local_values()),
                dense_x->get_local_vector()->get_stride());

            auto exec = this->get_executor();
            auto comm = this->get_communicator();
            auto [recv_vector, host_recv_vector] =
                init_recv_buffers<b_value_type>(
                    exec, row_gatherer_.get(), dense_b->get_size()[1],
                    recv_buffer_, host_recv_buffer_);
            auto recv_ptr = mpi::requires_host_buffer(exec, comm)
                                ? host_recv_vector.get()
                                : recv_vector.get();
            auto req = this->row_gatherer_->apply_async(dense_b, recv_ptr);
            local_mtx_->apply(local_alpha.get(), dense_b->get_local_vector(),
                              local_beta.get(), local_x);
            req.wait();

            if (recv_ptr != recv_vector.get()) {
                recv_vector->copy_from(host_recv_vector);
            }
            if (auto coo = std::dynamic_pointer_cast<
                    const ::gko::matrix::Coo<ValueType, LocalIndexType>>(
                    non_local_mtx_)) {
                coo->apply2(local_alpha.get(), recv_vector->get_local_vector(),
                            local_x);
            } else {
                non_local_mtx_->apply(
                    local_alpha.get(), recv_vector->get_local_vector(),
                    one_scalar_.template get<x_value_type>().get(), local_x);
            }
        },
        b, x);
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
    auto scaling_stride = scaling_factors->get_stride();
    if (scaling_stride != 1) {
        scaling_factors_single_stride = global_vector_type::create(exec, comm);
        scaling_factors_single_stride->copy_from(scaling_factors.get());
    }
    const global_vector_type* scaling_factors_ptr =
        scaling_stride == 1 ? scaling_factors.get()
                            : scaling_factors_single_stride.get();
    const auto scale_diag = gko::matrix::Diagonal<ValueType>::create_const(
        exec, n_local_cols,
        make_const_array_view(exec, n_local_cols,
                              scaling_factors_ptr->get_const_local_values()));

    auto [recv_vector, host_recv_vector] = init_recv_buffers<ValueType>(
        exec, row_gatherer_.get(), scaling_factors->get_size()[1], recv_buffer_,
        host_recv_buffer_);
    auto recv_ptr = mpi::requires_host_buffer(exec, comm)
                        ? host_recv_vector.get()
                        : recv_vector.get();

    auto req = row_gatherer_->apply_async(scaling_factors_ptr, recv_ptr);
    scale_diag->rapply(local_mtx_, local_mtx_);
    req.wait();
    if (n_non_local_cols > 0) {
        if (recv_ptr != recv_vector.get()) {
            recv_vector->copy_from(host_recv_vector);
        }
        const auto non_local_scale_diag =
            gko::matrix::Diagonal<ValueType>::create_const(
                exec, n_non_local_cols,
                make_const_array_view(exec, n_non_local_cols,
                                      recv_vector->get_const_local_values()));
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
      DistributedBase{other.get_communicator()},
      row_gatherer_{RowGatherer<LocalIndexType>::create(
          other.get_executor(), other.get_communicator())},
      imap_(other.get_executor()),
      one_scalar_(other.get_executor(), 1.0)
{
    *this = other;
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
Matrix<ValueType, LocalIndexType, GlobalIndexType>::Matrix(
    Matrix&& other) noexcept
    : EnableLinOp<Matrix<value_type, local_index_type,
                         global_index_type>>{other.get_executor()},
      DistributedBase{other.get_communicator()},
      row_gatherer_{RowGatherer<LocalIndexType>::create(
          other.get_executor(), other.get_communicator())},
      imap_(other.get_executor()),
      one_scalar_(other.get_executor(), 1.0)
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
