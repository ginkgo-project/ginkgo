// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/matrix/sellp_kernels.hpp"


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/components/prefix_sum_kernels.hpp"


namespace gko {
namespace kernels {
namespace reference {
/**
 * @brief The SELL-P matrix format namespace.
 * @ref Sellp
 * @ingroup sellp
 */
namespace sellp {


template <typename ValueType, typename IndexType>
void spmv(std::shared_ptr<const ReferenceExecutor> exec,
          const matrix::Sellp<ValueType, IndexType>* a,
          const matrix::Dense<ValueType>* b, matrix::Dense<ValueType>* c)
{
    auto col_idxs = a->get_const_col_idxs();
    auto slice_lengths = a->get_const_slice_lengths();
    auto slice_sets = a->get_const_slice_sets();
    auto slice_size = a->get_slice_size();
    auto slice_num = ceildiv(a->get_size()[0] + slice_size - 1, slice_size);
    for (size_type slice = 0; slice < slice_num; slice++) {
        for (size_type row = 0; row < slice_size; row++) {
            size_type global_row = slice * slice_size + row;
            if (global_row >= a->get_size()[0]) {
                break;
            }
            for (size_type j = 0; j < c->get_size()[1]; j++) {
                c->at(global_row, j) = zero<ValueType>();
            }
            for (size_type i = 0; i < slice_lengths[slice]; i++) {
                auto val = a->val_at(row, slice_sets[slice], i);
                auto col = a->col_at(row, slice_sets[slice], i);
                if (col != invalid_index<IndexType>()) {
                    for (size_type j = 0; j < c->get_size()[1]; j++) {
                        c->at(global_row, j) += val * b->at(col, j);
                    }
                }
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_SELLP_SPMV_KERNEL);


template <typename ValueType, typename IndexType>
void advanced_spmv(std::shared_ptr<const ReferenceExecutor> exec,
                   const matrix::Dense<ValueType>* alpha,
                   const matrix::Sellp<ValueType, IndexType>* a,
                   const matrix::Dense<ValueType>* b,
                   const matrix::Dense<ValueType>* beta,
                   matrix::Dense<ValueType>* c)
{
    auto vals = a->get_const_values();
    auto col_idxs = a->get_const_col_idxs();
    auto slice_lengths = a->get_const_slice_lengths();
    auto slice_sets = a->get_const_slice_sets();
    auto slice_size = a->get_slice_size();
    auto slice_num = ceildiv(a->get_size()[0] + slice_size - 1, slice_size);
    auto valpha = alpha->at(0, 0);
    auto vbeta = beta->at(0, 0);
    for (size_type slice = 0; slice < slice_num; slice++) {
        for (size_type row = 0; row < slice_size; row++) {
            size_type global_row = slice * slice_size + row;
            if (global_row >= a->get_size()[0]) {
                break;
            }
            for (size_type j = 0; j < c->get_size()[1]; j++) {
                c->at(global_row, j) *= vbeta;
            }
            for (size_type i = 0; i < slice_lengths[slice]; i++) {
                auto val = a->val_at(row, slice_sets[slice], i);
                auto col = a->col_at(row, slice_sets[slice], i);
                if (col != invalid_index<IndexType>()) {
                    for (size_type j = 0; j < c->get_size()[1]; j++) {
                        c->at(global_row, j) += valpha * val * b->at(col, j);
                    }
                }
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_SELLP_ADVANCED_SPMV_KERNEL);


template <typename IndexType>
void compute_slice_sets(std::shared_ptr<const DefaultExecutor> exec,
                        const array<IndexType>& row_ptrs, size_type slice_size,
                        size_type stride_factor, size_type* slice_sets,
                        size_type* slice_lengths)
{
    const auto num_rows = row_ptrs.get_size() - 1;
    const auto num_slices = ceildiv(num_rows, slice_size);
    const auto row_ptrs_ptr = row_ptrs.get_const_data();
    for (size_type slice = 0; slice < num_slices; slice++) {
        size_type slice_length = 0;
        for (size_type local_row = 0; local_row < slice_size; local_row++) {
            const auto row = slice * slice_size + local_row;
            const auto row_length =
                row < num_rows ? row_ptrs_ptr[row + 1] - row_ptrs_ptr[row]
                               : IndexType{};
            slice_length = std::max<size_type>(
                slice_length,
                ceildiv(row_length, stride_factor) * stride_factor);
        }
        slice_lengths[slice] = slice_length;
    }
    exec->copy(num_slices, slice_lengths, slice_sets);
    components::prefix_sum_nonnegative(exec, slice_sets, num_slices + 1);
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(
    GKO_DECLARE_SELLP_COMPUTE_SLICE_SETS_KERNEL);


template <typename ValueType, typename IndexType>
void fill_in_matrix_data(std::shared_ptr<const DefaultExecutor> exec,
                         const device_matrix_data<ValueType, IndexType>& data,
                         const int64* row_ptrs,
                         matrix::Sellp<ValueType, IndexType>* output)
{
    const auto slice_size = output->get_slice_size();
    const auto slice_sets = output->get_const_slice_sets();
    const auto cols = output->get_col_idxs();
    const auto vals = output->get_values();
    for (size_type row = 0; row < output->get_size()[0]; row++) {
        const auto row_begin = row_ptrs[row];
        const auto row_end = row_ptrs[row + 1];
        const auto row_nnz = row_end - row_begin;
        const auto slice = row / slice_size;
        const auto local_row = row % slice_size;
        const auto slice_begin = slice_sets[slice];
        const auto slice_end = slice_sets[slice + 1];
        const auto slice_length = slice_end - slice_begin;
        auto out_idx = slice_begin * slice_size + local_row;
        for (auto i = row_begin; i < row_end; i++) {
            cols[out_idx] = data.get_const_col_idxs()[i];
            vals[out_idx] = data.get_const_values()[i];
            out_idx += slice_size;
        }
        for (auto i = row_nnz; i < slice_length; i++) {
            cols[out_idx] = invalid_index<IndexType>();
            vals[out_idx] = zero<ValueType>();
            out_idx += slice_size;
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_SELLP_FILL_IN_MATRIX_DATA_KERNEL);


template <typename ValueType, typename IndexType>
void fill_in_dense(std::shared_ptr<const ReferenceExecutor> exec,
                   const matrix::Sellp<ValueType, IndexType>* source,
                   matrix::Dense<ValueType>* result)
{
    auto num_rows = source->get_size()[0];
    auto num_cols = source->get_size()[1];
    auto vals = source->get_const_values();
    auto col_idxs = source->get_const_col_idxs();
    auto slice_lengths = source->get_const_slice_lengths();
    auto slice_sets = source->get_const_slice_sets();
    auto slice_size = source->get_slice_size();
    auto slice_num =
        ceildiv(source->get_size()[0] + slice_size - 1, slice_size);
    for (size_type slice = 0; slice < slice_num; slice++) {
        for (size_type row = 0; row < slice_size; row++) {
            size_type global_row = slice * slice_size + row;
            if (global_row >= num_rows) {
                break;
            }
            for (size_type i = slice_sets[slice]; i < slice_sets[slice + 1];
                 i++) {
                const auto col = col_idxs[row + i * slice_size];
                if (col != invalid_index<IndexType>()) {
                    result->at(global_row, col) = vals[row + i * slice_size];
                }
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_SELLP_FILL_IN_DENSE_KERNEL);


template <typename ValueType, typename IndexType>
void count_nonzeros_per_row(std::shared_ptr<const DefaultExecutor> exec,
                            const matrix::Sellp<ValueType, IndexType>* source,
                            IndexType* result)
{
    auto num_rows = source->get_size()[0];
    auto slice_size = source->get_slice_size();
    auto slice_num = ceildiv(num_rows, slice_size);

    const auto vals = source->get_const_values();
    const auto slice_lengths = source->get_const_slice_lengths();
    const auto slice_sets = source->get_const_slice_sets();
    const auto col_idxs = source->get_const_col_idxs();

    for (size_type slice = 0; slice < slice_num; slice++) {
        for (size_type row = 0; row < slice_size; row++) {
            auto global_row = slice * slice_size + row;
            if (global_row >= num_rows) {
                break;
            }
            IndexType row_nnz{};
            for (size_type sellp_ind = slice_sets[slice] * slice_size + row;
                 sellp_ind < slice_sets[slice + 1] * slice_size + row;
                 sellp_ind += slice_size) {
                row_nnz +=
                    col_idxs[sellp_ind] != invalid_index<IndexType>() ? 1 : 0;
            }
            result[global_row] = row_nnz;
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_SELLP_COUNT_NONZEROS_PER_ROW_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_csr(std::shared_ptr<const ReferenceExecutor> exec,
                    const matrix::Sellp<ValueType, IndexType>* source,
                    matrix::Csr<ValueType, IndexType>* result)
{
    auto num_rows = source->get_size()[0];
    auto slice_size = source->get_slice_size();
    auto slice_num = ceildiv(num_rows, slice_size);

    const auto source_vals = source->get_const_values();
    const auto source_slice_lengths = source->get_const_slice_lengths();
    const auto source_slice_sets = source->get_const_slice_sets();
    const auto source_col_idxs = source->get_const_col_idxs();

    auto result_vals = result->get_values();
    auto result_row_ptrs = result->get_row_ptrs();
    auto result_col_idxs = result->get_col_idxs();

    size_type cur_ptr = 0;

    for (size_type slice = 0; slice < slice_num; slice++) {
        for (size_type row = 0; row < slice_size; row++) {
            auto global_row = slice * slice_size + row;
            if (global_row >= num_rows) {
                break;
            }
            result_row_ptrs[global_row] = cur_ptr;
            for (size_type sellp_ind =
                     source_slice_sets[slice] * slice_size + row;
                 sellp_ind < source_slice_sets[slice + 1] * slice_size + row;
                 sellp_ind += slice_size) {
                if (source_col_idxs[sellp_ind] != invalid_index<IndexType>()) {
                    result_vals[cur_ptr] = source_vals[sellp_ind];
                    result_col_idxs[cur_ptr] = source_col_idxs[sellp_ind];
                    cur_ptr++;
                }
            }
        }
    }
    result_row_ptrs[num_rows] = cur_ptr;
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_SELLP_CONVERT_TO_CSR_KERNEL);


template <typename ValueType, typename IndexType>
void extract_diagonal(std::shared_ptr<const ReferenceExecutor> exec,
                      const matrix::Sellp<ValueType, IndexType>* orig,
                      matrix::Diagonal<ValueType>* diag)
{
    const auto diag_size = diag->get_size()[0];
    const auto slice_size = orig->get_slice_size();
    const auto slice_num = ceildiv(orig->get_size()[0], slice_size);

    const auto orig_values = orig->get_const_values();
    const auto orig_slice_sets = orig->get_const_slice_sets();
    const auto orig_slice_lengths = orig->get_const_slice_lengths();
    const auto orig_col_idxs = orig->get_const_col_idxs();
    auto diag_values = diag->get_values();

    for (size_type slice = 0; slice < slice_num; slice++) {
        for (size_type row = 0; row < slice_size; row++) {
            auto global_row = slice_size * slice + row;
            if (global_row >= diag_size) {
                break;
            }
            for (size_type i = 0; i < orig_slice_lengths[slice]; i++) {
                if (orig->col_at(row, orig_slice_sets[slice], i) ==
                    global_row) {
                    diag_values[global_row] =
                        orig->val_at(row, orig_slice_sets[slice], i);
                    break;
                }
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_SELLP_EXTRACT_DIAGONAL_KERNEL);


}  // namespace sellp
}  // namespace reference
}  // namespace kernels
}  // namespace gko
