// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "ginkgo/core/distributed/dd_matrix.hpp"

#include <ginkgo/core/base/precision_dispatch.hpp>
#include <ginkgo/core/distributed/vector.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
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
      prolongation_{global_matrix_type::create(exec, comm)},
      map_{exec}
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
    result->map_ = map_;
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
    result->map_ = map_;
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

    map_ = gko::experimental::distributed::index_map<LocalIndexType,
                                                     GlobalIndexType>(
        exec, partition, local_part, non_owning_row_idxs);

    GlobalIndexType full_local_num_rows =
        map_.get_local_size() + map_.get_non_local_size();
    auto local_col_idxs = map_.map_to_local(
        arrays.col_idxs, gko::experimental::distributed::index_space::combined);
    auto local_row_idxs = map_.map_to_local(
        arrays.row_idxs, gko::experimental::distributed::index_space::combined);

    // Active local DOFs = those referenced by this rank's entries; DOFs
    // without local contributions are excluded from the local matrix and
    // the broken space.
    device_matrix_data<value_type, local_index_type> full_local_data{
        exec,
        dim<2>{static_cast<size_type>(full_local_num_rows),
               static_cast<size_type>(full_local_num_rows)},
        local_row_idxs, local_col_idxs, arrays.values};
    full_local_data.sort_row_major();
    auto host_data = full_local_data.copy_to_host();
    std::vector<bool> row_active(full_local_num_rows, false);
    std::vector<bool> col_active(full_local_num_rows, false);
    for (auto entry : host_data.nonzeros) {
        row_active[entry.row] = true;
        col_active[entry.column] = true;
    }
    array<LocalIndexType> host_active{exec->get_master()};
    array<LocalIndexType> old_to_new{exec->get_master(),
                                     static_cast<size_type>(
                                         full_local_num_rows)};
    {
        std::vector<LocalIndexType> active;
        active.reserve(full_local_num_rows);
        for (GlobalIndexType i = 0; i < full_local_num_rows; i++) {
            old_to_new.get_data()[i] = invalid_index<LocalIndexType>();
            if (row_active[i] || col_active[i]) {
                old_to_new.get_data()[i] =
                    static_cast<LocalIndexType>(active.size());
                active.push_back(static_cast<LocalIndexType>(i));
            }
        }
        host_active = array<LocalIndexType>(exec->get_master(), active.begin(),
                                            active.end());
    }
    GlobalIndexType local_num_rows = host_active.get_size();

    // Remap entries into the compressed active numbering.
    for (auto& entry : host_data.nonzeros) {
        entry.row = old_to_new.get_const_data()[entry.row];
        entry.column = old_to_new.get_const_data()[entry.column];
    }
    // DOFs that appear only as columns (structurally nonsymmetric input)
    // still have empty rows: keep the legacy unit-diagonal padding for
    // exactly those, so local solvers remain well defined. For structurally
    // symmetric data this loop adds nothing.
    array<ValueType> prolongate_values{exec->get_master(),
                                       static_cast<size_type>(local_num_rows)};
    prolongate_values.fill(one<ValueType>());
    for (GlobalIndexType i = 0; i < full_local_num_rows; i++) {
        if (col_active[i] && !row_active[i]) {
            auto new_idx = old_to_new.get_const_data()[i];
            host_data.nonzeros.emplace_back(new_idx, new_idx,
                                            one<ValueType>());
            prolongate_values.get_data()[new_idx] = zero<ValueType>();
        }
    }
    host_data.size = dim<2>{static_cast<size_type>(local_num_rows),
                            static_cast<size_type>(local_num_rows)};
    host_data.sort_row_major();
    as<ReadableFromMatrixData<ValueType, LocalIndexType>>(this->local_mtx_)
        ->read(std::move(host_data));

    active_idxs_ = host_active;
    active_idxs_.set_executor(exec);

    // Gather ACTIVE local sizes from all ranks and build the partition in the
    // enriched (broken) space.
    array<GlobalIndexType> range_bounds{exec->get_master(), num_parts + 1};
    comm.all_gather(exec->get_master(), &local_num_rows, 1,
                    range_bounds.get_data(), 1);
    range_bounds.set_executor(exec);
    exec->run(dd_matrix::make_prefix_sum_nonnegative(range_bounds.get_data(),
                                                     num_parts + 1));
    auto large_partition =
        share(Partition<LocalIndexType, GlobalIndexType>::build_from_contiguous(
            exec, range_bounds));

    // Build the restriction and prolongation operators over the active DOFs.
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
        map_.map_to_global(active_idxs_, index_space::combined);
    auto restrict_row_idxs =
        enriched_map.map_to_global(local_idxs, index_space::combined);
    array<ValueType> restrict_values{exec,
                                     static_cast<size_type>(local_num_rows)};
    auto prolongate_col_idxs =
        enriched_map.map_to_global(local_idxs, index_space::combined);
    auto prolongate_row_idxs =
        map_.map_to_global(active_idxs_, index_space::combined);
    restrict_values.fill(one<ValueType>());
    prolongate_values.set_executor(exec);

    device_matrix_data<ValueType, GlobalIndexType> restrict_data{
        exec, dim<2>{large_partition->get_size(), partition->get_size()},
        std::move(restrict_row_idxs), std::move(restrict_col_idxs),
        std::move(restrict_values)};
    restrict_data.remove_zeros();
    restrict_data.sort_row_major();
    restriction_ =
        Matrix<ValueType, LocalIndexType, GlobalIndexType>::create(exec, comm);
    restriction_->read_distributed(restrict_data, large_partition, partition);
    device_matrix_data<ValueType, GlobalIndexType> prolongate_data{
        exec, dim<2>{partition->get_size(), large_partition->get_size()},
        std::move(prolongate_row_idxs), std::move(prolongate_col_idxs),
        std::move(prolongate_values)};
    prolongate_data.remove_zeros();
    prolongate_data.sort_row_major();
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
void DdMatrix<ValueType, LocalIndexType, GlobalIndexType>::set_null_space(
    std::shared_ptr<const global_vector_type> null_space)
{
    if (null_space) {
        GKO_ASSERT_EQUAL_ROWS(this, null_space.get());
    }
    null_space_ = std::move(null_space);
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
void DdMatrix<ValueType, LocalIndexType, GlobalIndexType>::
    set_constant_null_space(
        std::shared_ptr<const Partition<local_index_type, global_index_type>>
            partition)
{
    auto exec = this->get_executor();
    auto comm = this->get_communicator();
    auto global_size = this->get_size()[0];
    auto local_size =
        static_cast<size_type>(partition->get_part_size(comm.rank()));
    auto local_vec =
        gko::matrix::Dense<ValueType>::create(exec, dim<2>{local_size, 1});
    // Fill with 1/sqrt(n) so the null-space vector is normalized.
    local_vec->fill(one<ValueType>() /
                    sqrt(static_cast<remove_complex<ValueType>>(global_size)));
    null_space_ = global_vector_type::create(exec, comm, dim<2>{global_size, 1},
                                             std::move(local_vec));
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
void DdMatrix<ValueType, LocalIndexType, GlobalIndexType>::remove_null_space(
    ptr_param<global_vector_type> vec) const
{
    if (!null_space_) {
        return;
    }
    auto exec = this->get_executor();
    GKO_ASSERT_EQUAL_COLS(null_space_.get(), vec.get());
    const auto num_nsp = null_space_->get_size()[1];
    // dot = null_space^T * vec  (1 x num_nsp, column-wise dot products)
    nsp_dot_buffer_.init(exec, dim<2>{1, num_nsp});
    null_space_->compute_dot(vec.get(), nsp_dot_buffer_.get());
    // vec = vec - dot * null_space
    vec->sub_scaled(nsp_dot_buffer_.get(), null_space_);
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
DdMatrix<ValueType, LocalIndexType, GlobalIndexType>::DdMatrix(
    const DdMatrix& other)
    : EnableLinOp<DdMatrix<value_type, local_index_type,
                           global_index_type>>{other.get_executor()},
      DistributedBase{other.get_communicator()},
      map_{other.get_executor()}
{
    *this = other;
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
DdMatrix<ValueType, LocalIndexType, GlobalIndexType>::DdMatrix(
    DdMatrix&& other) noexcept
    : EnableLinOp<DdMatrix<value_type, local_index_type,
                           global_index_type>>{other.get_executor()},
      DistributedBase{other.get_communicator()},
      map_{other.get_executor()}
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
        map_ = other.map_;
        null_space_ = other.null_space_;
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
        map_ = other.map_;
        null_space_ = std::move(other.null_space_);
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
