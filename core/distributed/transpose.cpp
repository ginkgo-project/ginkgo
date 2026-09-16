// SPDX-FileCopyrightText: 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/device_matrix_data.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/mpi.hpp>
#include <ginkgo/core/distributed/index_map.hpp>
#include <ginkgo/core/distributed/partition.hpp>

#include "ginkgo/core/distributed/matrix.hpp"


namespace gko {
namespace experimental {
namespace distributed {


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
void Matrix<ValueType, LocalIndexType, GlobalIndexType>::transpose(
    ptr_param<Matrix> result) const
{
    auto* result_ptr = result.get();

    auto exec = this->get_executor();
    auto comm = this->get_communicator();
    auto rank = comm.rank();

    // Both partitions have to be known; read_distributed sets them.
    if (this->get_row_partition() == nullptr) {
        GKO_INVALID_STATE(
            "distributed transpose requires a row partition, which is only "
            "set when the matrix is filled by read_distributed");
    }
    auto row_partition = this->get_row_partition();
    auto col_partition = this->imap_.get_partition();
    if (col_partition == nullptr) {
        GKO_INVALID_STATE(
            "distributed transpose requires a column partition, which is only "
            "set when the matrix is filled by read_distributed");
    }

    if (result_ptr->get_executor() != exec) {
        GKO_INVALID_STATE(
            "distributed transpose requires the result to be on the same "
            "executor as this matrix");
    }
    auto result_comm = result_ptr->get_communicator();
    if (!(comm.is_identical(result_comm) || comm.is_congruent(result_comm))) {
        GKO_INVALID_STATE(
            "distributed transpose requires the result to use the same "
            "communicator as this matrix");
    }

    device_matrix_data<ValueType, LocalIndexType> diag_data{exec};
    device_matrix_data<ValueType, LocalIndexType> off_diag_data{exec};
    as<WritableToMatrixData<ValueType, LocalIndexType>>(this->get_diag_matrix())
        ->write(diag_data);
    as<WritableToMatrixData<ValueType, LocalIndexType>>(
        this->get_off_diag_matrix())
        ->write(off_diag_data);

    const auto diag_nnz = diag_data.get_num_stored_elements();
    const auto off_diag_nnz = off_diag_data.get_num_stored_elements();
    const auto total_nnz = diag_nnz + off_diag_nnz;

    // Rows to global indices. A matrix has no non-local rows, so an index map
    // over the row partition needs no remote indices.
    array<LocalIndexType> local_rows{exec, total_nnz};
    exec->copy_from(exec, diag_nnz, diag_data.get_const_row_idxs(),
                    local_rows.get_data());
    exec->copy_from(exec, off_diag_nnz, off_diag_data.get_const_row_idxs(),
                    local_rows.get_data() + diag_nnz);
    const index_map<LocalIndexType, GlobalIndexType> row_imap{
        exec, row_partition, rank, array<GlobalIndexType>{exec}};
    const auto global_rows =
        row_imap.map_to_global(local_rows, index_space::local);

    // Columns to global indices: the diagonal block holds owned columns, the
    // off-diagonal block non-local ones.
    const auto diag_cols = gko::detail::array_const_cast(
        make_const_array_view(exec, diag_nnz, diag_data.get_const_col_idxs()));
    const auto off_diag_cols =
        gko::detail::array_const_cast(make_const_array_view(
            exec, off_diag_nnz, off_diag_data.get_const_col_idxs()));
    const auto global_diag_cols =
        this->imap_.map_to_global(diag_cols, index_space::local);
    const auto global_off_diag_cols =
        this->imap_.map_to_global(off_diag_cols, index_space::non_local);

    // Entry (i, j) becomes (j, i): result rows are this matrix's columns.
    array<GlobalIndexType> t_row_idxs{exec, total_nnz};
    array<GlobalIndexType> t_col_idxs{exec, total_nnz};
    array<ValueType> t_values{exec, total_nnz};
    exec->copy_from(exec, diag_nnz, global_diag_cols.get_const_data(),
                    t_row_idxs.get_data());
    exec->copy_from(exec, off_diag_nnz, global_off_diag_cols.get_const_data(),
                    t_row_idxs.get_data() + diag_nnz);
    exec->copy_from(exec, total_nnz, global_rows.get_const_data(),
                    t_col_idxs.get_data());
    exec->copy_from(exec, diag_nnz, diag_data.get_const_values(),
                    t_values.get_data());
    exec->copy_from(exec, off_diag_nnz, off_diag_data.get_const_values(),
                    t_values.get_data() + diag_nnz);

    device_matrix_data<ValueType, GlobalIndexType> transposed_data{
        exec, dim<2>{this->get_size()[1], this->get_size()[0]},
        std::move(t_row_idxs), std::move(t_col_idxs), std::move(t_values)};

    // Swapped partitions make this matrix's columns the result's rows. The
    // communicating assembly sends each entry to the rank owning its row, and
    // read_distributed builds the index map, halo and row gatherer.
    result_ptr->read_distributed(transposed_data, col_partition, row_partition,
                                 assembly_mode::communicate);
}


#define GKO_DECLARE_DISTRIBUTED_MATRIX_TRANSPOSE(ValueType, LocalIndexType, \
                                                 GlobalIndexType)           \
    void Matrix<ValueType, LocalIndexType, GlobalIndexType>::transpose(     \
        ptr_param<Matrix> result) const
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_LOCAL_GLOBAL_INDEX_TYPE(
    GKO_DECLARE_DISTRIBUTED_MATRIX_TRANSPOSE);


}  // namespace distributed
}  // namespace experimental
}  // namespace gko
