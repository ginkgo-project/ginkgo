// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "ginkgo/core/distributed/dd_matrix.hpp"

#include <ginkgo/core/base/precision_dispatch.hpp>
#include <ginkgo/core/distributed/vector.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/diagonal.hpp>

#include "core/components/fill_array_kernels.hpp"
#include "core/components/prefix_sum_kernels.hpp"
#include "core/distributed/dd_matrix_kernels.hpp"
#include "ginkgo/core/base/exception_helpers.hpp"

namespace gko {
namespace experimental {
namespace distributed {
namespace dd_matrix {
namespace {


GKO_REGISTER_OPERATION(filter_non_owning_idxs,
                       distributed_dd_matrix::filter_non_owning_idxs);
GKO_REGISTER_OPERATION(fill_seq_array, components::fill_seq_array);
GKO_REGISTER_OPERATION(prefix_sum_nonnegative,
                       components::prefix_sum_nonnegative);


}  // namespace
}  // namespace dd_matrix


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
DdMatrix<ValueType, LocalIndexType, GlobalIndexType>::DdMatrix(
    std::shared_ptr<const Executor> exec, mpi::communicator comm)
    : DdMatrix(exec, comm,
               gko::matrix::Csr<ValueType, LocalIndexType>::create(exec))
{}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
DdMatrix<ValueType, LocalIndexType, GlobalIndexType>::DdMatrix(
    std::shared_ptr<const Executor> exec, mpi::communicator comm,
    ptr_param<const LinOp> matrix_template)
    : EnableLinOp<
          DdMatrix<value_type, local_index_type, global_index_type>>{exec},
      DistributedBase{comm},
      local_mtx_{matrix_template->clone(exec)},
      restriction_{global_matrix_type::create(exec, comm)},
      prolongation_{global_matrix_type::create(exec, comm)}
{
    GKO_ASSERT(
        (dynamic_cast<ReadableFromMatrixData<ValueType, LocalIndexType>*>(
            local_mtx_.get())));
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
std::unique_ptr<DdMatrix<ValueType, LocalIndexType, GlobalIndexType>>
DdMatrix<ValueType, LocalIndexType, GlobalIndexType>::create(
    std::shared_ptr<const Executor> exec, mpi::communicator comm)
{
    return std::unique_ptr<DdMatrix>{new DdMatrix{exec, comm}};
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
std::unique_ptr<DdMatrix<ValueType, LocalIndexType, GlobalIndexType>>
DdMatrix<ValueType, LocalIndexType, GlobalIndexType>::create(
    std::shared_ptr<const Executor> exec, mpi::communicator comm,
    ptr_param<const LinOp> matrix_template)
{
    return std::unique_ptr<DdMatrix>{new DdMatrix{exec, comm, matrix_template}};
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
void DdMatrix<ValueType, LocalIndexType, GlobalIndexType>::convert_to(
    DdMatrix<next_precision_base<value_type>, local_index_type,
             global_index_type>* result) const
{
    GKO_ASSERT(this->get_communicator().size() ==
               result->get_communicator().size());
    result->local_mtx_->copy_from(this->local_mtx_);
    result->restriction_->copy_from(this->restriction_);
    result->prolongation_->copy_from(this->prolongation_);
    result->set_size(this->get_size());
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
void DdMatrix<ValueType, LocalIndexType, GlobalIndexType>::move_to(
    DdMatrix<next_precision_base<value_type>, local_index_type,
             global_index_type>* result)
{
    GKO_ASSERT(this->get_communicator().size() ==
               result->get_communicator().size());
    result->local_mtx_->move_from(this->local_mtx_);
    result->restriction_->move_from(this->restriction_);
    result->prolongation_->move_from(this->prolongation_);
    result->set_size(this->get_size());
    this->set_size({});
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
void DdMatrix<ValueType, LocalIndexType, GlobalIndexType>::read_distributed(
    const device_matrix_data<value_type, global_index_type>& data,
    std::shared_ptr<const Partition<local_index_type, global_index_type>>
        partition)
{
    const auto comm = this->get_communicator();
    GKO_ASSERT_EQ(data.get_size()[0], partition->get_size());
    GKO_ASSERT_EQ(data.get_size()[1], partition->get_size());
    GKO_ASSERT_EQ(comm.size(), partition->get_num_parts());
    auto exec = this->get_executor();
    auto local_part = comm.rank();
    auto use_host_buffer = mpi::requires_host_buffer(exec, comm);
    auto tmp_partition = make_temporary_clone(exec, partition);

    // set up LinOp sizes
    auto global_num_rows = partition->get_size();
    dim<2> global_dim{global_num_rows, global_num_rows};
    this->set_size(global_dim);

    size_type num_parts = comm.size();
    array<GlobalIndexType> non_owning_row_idxs{exec};
    array<GlobalIndexType> non_owning_col_idxs{exec};
    device_matrix_data<value_type, global_index_type> data_copy{exec, data};
    auto arrays = data_copy.empty_out();

    exec->run(dd_matrix::make_filter_non_owning_idxs(
        data, make_temporary_clone(exec, partition).get(),
        make_temporary_clone(exec, partition).get(), local_part,
        non_owning_row_idxs, non_owning_col_idxs));

    auto map = gko::experimental::distributed::index_map<LocalIndexType,
                                                         GlobalIndexType>(
        exec, partition, local_part, non_owning_row_idxs);

    GlobalIndexType local_num_rows =
        map.get_local_size() + map.get_non_local_size();
    auto local_col_idxs = map.map_to_local(
        arrays.col_idxs, gko::experimental::distributed::index_space::combined);
    auto local_row_idxs = map.map_to_local(
        arrays.row_idxs, gko::experimental::distributed::index_space::combined);

    // Construct the local diagonal block.
    device_matrix_data<value_type, local_index_type> local_data{
        exec,
        dim<2>{static_cast<size_type>(local_num_rows),
               static_cast<size_type>(local_num_rows)},
        local_row_idxs, local_col_idxs, arrays.values};
    local_data.sort_row_major();
    as<ReadableFromMatrixData<ValueType, LocalIndexType>>(this->local_mtx_)
        ->read(std::move(local_data));

    // Gather local sizes from all ranks and build the partition in the enriched
    // space.
    array<GlobalIndexType> range_bounds{
        use_host_buffer ? exec->get_master() : exec, num_parts + 1};
    comm.all_gather(exec, &local_num_rows, 1, range_bounds.get_data(), 1);
    range_bounds.set_executor(exec);
    exec->run(dd_matrix::make_prefix_sum_nonnegative(range_bounds.get_data(),
                                                     num_parts + 1));
    auto large_partition =
        share(Partition<LocalIndexType, GlobalIndexType>::build_from_contiguous(
            exec, range_bounds));

    // Build the restricion and prolongation operators.
    array<GlobalIndexType> remote_idxs{exec, 0};
    auto enriched_map =
        gko::experimental::distributed::index_map<LocalIndexType,
                                                  GlobalIndexType>(
            exec, large_partition, local_part, remote_idxs);
    array<LocalIndexType> local_idxs{exec,
                                     static_cast<size_type>(local_num_rows)};
    exec->run(dd_matrix::make_fill_seq_array(
        local_idxs.get_data(), static_cast<size_type>(local_num_rows)));
    auto restrict_col_idxs =
        map.map_to_global(local_idxs, index_space::combined);
    auto restrict_row_idxs =
        enriched_map.map_to_global(local_idxs, index_space::combined);
    array<ValueType> restrict_values{exec,
                                     static_cast<size_type>(local_num_rows)};
    restrict_values.fill(one<ValueType>());
    device_matrix_data<ValueType, GlobalIndexType> restrict_data{
        exec, dim<2>{large_partition->get_size(), partition->get_size()},
        std::move(restrict_row_idxs), std::move(restrict_col_idxs),
        std::move(restrict_values)};
    restriction_ =
        Matrix<ValueType, LocalIndexType, GlobalIndexType>::create(exec, comm);
    restriction_->read_distributed(restrict_data, large_partition, partition);
    auto prolongate_col_idxs =
        enriched_map.map_to_global(local_idxs, index_space::combined);
    auto prolongate_row_idxs =
        map.map_to_global(local_idxs, index_space::combined);
    array<ValueType> prolongate_values{exec,
                                       static_cast<size_type>(local_num_rows)};
    prolongate_values.fill(one<ValueType>());
    device_matrix_data<ValueType, GlobalIndexType> prolongate_data{
        exec, dim<2>{partition->get_size(), large_partition->get_size()},
        std::move(prolongate_row_idxs), std::move(prolongate_col_idxs),
        std::move(prolongate_values)};
    prolongation_ =
        Matrix<ValueType, LocalIndexType, GlobalIndexType>::create(exec, comm);
    prolongation_->read_distributed(prolongate_data, partition, large_partition,
                                    assembly_mode::communicate);
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
void DdMatrix<ValueType, LocalIndexType, GlobalIndexType>::read_distributed(
    const matrix_data<ValueType, global_index_type>& data,
    std::shared_ptr<const Partition<local_index_type, global_index_type>>
        partition)
{
    return this->read_distributed(
        device_matrix_data<value_type, global_index_type>::create_from_host(
            this->get_executor(), data),
        partition);
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
void DdMatrix<ValueType, LocalIndexType, GlobalIndexType>::apply_impl(
    const LinOp* b, LinOp* x) const
{
    auto exec = this->get_executor();
    auto comm = this->get_communicator();
    const auto nrhs = x->get_size()[1];
    dim<2> global_buffer_size{restriction_->get_size()[0], nrhs};
    dim<2> local_buffer_size{local_mtx_->get_size()[0], nrhs};
    lhs_buffer_.init(exec, comm, global_buffer_size, local_buffer_size);
    rhs_buffer_.init(exec, comm, global_buffer_size, local_buffer_size);
    distributed::precision_dispatch_real_complex<ValueType>(
        [this](const auto dense_b, auto dense_x) {
            auto exec = this->get_executor();
            restriction_->apply(dense_b, lhs_buffer_.get());

            auto local_b = gko::matrix::Dense<ValueType>::create(
                exec, lhs_buffer_->get_local_vector()->get_size(),
                gko::make_array_view(
                    exec,
                    lhs_buffer_->get_local_vector()->get_num_stored_elements(),
                    lhs_buffer_->get_local_values()),
                lhs_buffer_->get_local_vector()->get_stride());
            auto local_x = gko::matrix::Dense<ValueType>::create(
                exec, rhs_buffer_->get_local_vector()->get_size(),
                gko::make_array_view(
                    exec,
                    rhs_buffer_->get_local_vector()->get_num_stored_elements(),
                    rhs_buffer_->get_local_values()),
                rhs_buffer_->get_local_vector()->get_stride());

            local_mtx_->apply(local_b, local_x);

            prolongation_->apply(rhs_buffer_.get(), dense_x);
        },
        b, x);
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
void DdMatrix<ValueType, LocalIndexType, GlobalIndexType>::apply_impl(
    const LinOp* alpha, const LinOp* b, const LinOp* beta, LinOp* x) const
{
    auto exec = this->get_executor();
    auto comm = this->get_communicator();
    const auto nrhs = x->get_size()[1];
    dim<2> global_buffer_size{restriction_->get_size()[0], nrhs};
    dim<2> local_buffer_size{local_mtx_->get_size()[0], nrhs};
    lhs_buffer_.init(exec, comm, global_buffer_size, local_buffer_size);
    rhs_buffer_.init(exec, comm, global_buffer_size, local_buffer_size);
    distributed::precision_dispatch_real_complex<ValueType>(
        [this](const auto local_alpha, const auto dense_b,
               const auto local_beta, auto dense_x) {
            auto exec = this->get_executor();
            restriction_->apply(dense_b, lhs_buffer_.get());

            auto local_b = gko::matrix::Dense<ValueType>::create(
                exec, lhs_buffer_->get_local_vector()->get_size(),
                gko::make_array_view(
                    exec,
                    lhs_buffer_->get_local_vector()->get_num_stored_elements(),
                    lhs_buffer_->get_local_values()),
                lhs_buffer_->get_local_vector()->get_stride());
            auto local_x = gko::matrix::Dense<ValueType>::create(
                exec, rhs_buffer_->get_local_vector()->get_size(),
                gko::make_array_view(
                    exec,
                    rhs_buffer_->get_local_vector()->get_num_stored_elements(),
                    rhs_buffer_->get_local_values()),
                rhs_buffer_->get_local_vector()->get_stride());

            local_mtx_->apply(local_b, local_x);

            prolongation_->apply(local_alpha, rhs_buffer_.get(), local_beta,
                                 dense_x);
        },
        alpha, b, beta, x);
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
void DdMatrix<ValueType, LocalIndexType, GlobalIndexType>::col_scale(
    ptr_param<const global_vector_type> scaling_factors)
{
    GKO_ASSERT_CONFORMANT(this, scaling_factors.get());
    GKO_ASSERT_EQ(scaling_factors->get_size()[1], 1);
    auto exec = this->get_executor();
    auto comm = this->get_communicator();
    dim<2> global_buffer_size{restriction_->get_size()[0], 1u};
    dim<2> local_buffer_size{local_mtx_->get_size()[0], 1u};
    lhs_buffer_.init(exec, comm, global_buffer_size, local_buffer_size);
    size_type n_local_cols = local_mtx_->get_size()[1];
    restriction_->apply(scaling_factors, lhs_buffer_.get());
    const auto scale_diag = gko::matrix::Diagonal<ValueType>::create_const(
        exec, n_local_cols,
        make_const_array_view(exec, n_local_cols,
                              lhs_buffer_->get_const_local_values()));
    scale_diag->rapply(local_mtx_, local_mtx_);
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
void DdMatrix<ValueType, LocalIndexType, GlobalIndexType>::row_scale(
    ptr_param<const global_vector_type> scaling_factors)
{
    GKO_ASSERT_EQUAL_ROWS(this, scaling_factors.get());
    GKO_ASSERT_EQ(scaling_factors->get_size()[1], 1);
    auto exec = this->get_executor();
    auto comm = this->get_communicator();
    dim<2> global_buffer_size{restriction_->get_size()[0], 1u};
    dim<2> local_buffer_size{local_mtx_->get_size()[0], 1u};
    lhs_buffer_.init(exec, comm, global_buffer_size, local_buffer_size);
    size_type n_local_cols = local_mtx_->get_size()[1];
    restriction_->apply(scaling_factors, lhs_buffer_.get());
    const auto scale_diag = gko::matrix::Diagonal<ValueType>::create_const(
        exec, n_local_cols,
        make_const_array_view(exec, n_local_cols,
                              lhs_buffer_->get_const_local_values()));
    scale_diag->apply(local_mtx_, local_mtx_);
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
DdMatrix<ValueType, LocalIndexType, GlobalIndexType>::DdMatrix(
    const DdMatrix& other)
    : EnableLinOp<DdMatrix<value_type, local_index_type,
                           global_index_type>>{other.get_executor()},
      DistributedBase{other.get_communicator()}
{
    *this = other;
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
DdMatrix<ValueType, LocalIndexType, GlobalIndexType>::DdMatrix(
    DdMatrix&& other) noexcept
    : EnableLinOp<DdMatrix<value_type, local_index_type,
                           global_index_type>>{other.get_executor()},
      DistributedBase{other.get_communicator()}
{
    *this = std::move(other);
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
DdMatrix<ValueType, LocalIndexType, GlobalIndexType>&
DdMatrix<ValueType, LocalIndexType, GlobalIndexType>::operator=(
    const DdMatrix& other)
{
    if (this != &other) {
        GKO_ASSERT_EQ(other.get_communicator().size(),
                      this->get_communicator().size());
        this->set_size(other.get_size());
        local_mtx_->copy_from(other.local_mtx_);
        restriction_->copy_from(other.restriction_);
        prolongation_->copy_from(other.prolongation_);
    }
    return *this;
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
DdMatrix<ValueType, LocalIndexType, GlobalIndexType>&
DdMatrix<ValueType, LocalIndexType, GlobalIndexType>::operator=(
    DdMatrix&& other)
{
    if (this != &other) {
        GKO_ASSERT_EQ(other.get_communicator().size(),
                      this->get_communicator().size());
        this->set_size(other.get_size());
        other.set_size({});
        local_mtx_->move_from(other.local_mtx_);
        restriction_->move_from(other.restriction_);
        prolongation_->move_from(other.prolongation_);
    }
    return *this;
}


#define GKO_DECLARE_DISTRIBUTED_DD_MATRIX(ValueType, LocalIndexType, \
                                          GlobalIndexType)           \
    class DdMatrix<ValueType, LocalIndexType, GlobalIndexType>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_LOCAL_GLOBAL_INDEX_TYPE_BASE(
    GKO_DECLARE_DISTRIBUTED_DD_MATRIX);


}  // namespace distributed
}  // namespace experimental
}  // namespace gko
