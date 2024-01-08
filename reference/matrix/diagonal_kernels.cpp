// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/matrix/diagonal_kernels.hpp"


#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>


namespace gko {
namespace kernels {
namespace reference {
/**
 * @brief The Diagonal matrix format namespace.
 *
 * @ingroup diagonal
 */
namespace diagonal {


template <typename ValueType>
void apply_to_dense(std::shared_ptr<const ReferenceExecutor> exec,
                    const matrix::Diagonal<ValueType>* a,
                    const matrix::Dense<ValueType>* b,
                    matrix::Dense<ValueType>* c, bool inverse)
{
    const auto diag_values = a->get_const_values();
    for (size_type row = 0; row < a->get_size()[0]; row++) {
        const auto scal =
            inverse ? one<ValueType>() / diag_values[row] : diag_values[row];
        for (size_type col = 0; col < b->get_size()[1]; col++) {
            c->at(row, col) = b->at(row, col) * scal;
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_DIAGONAL_APPLY_TO_DENSE_KERNEL);


template <typename ValueType>
void right_apply_to_dense(std::shared_ptr<const ReferenceExecutor> exec,
                          const matrix::Diagonal<ValueType>* a,
                          const matrix::Dense<ValueType>* b,
                          matrix::Dense<ValueType>* c)
{
    const auto diag_values = a->get_const_values();
    for (size_type row = 0; row < b->get_size()[0]; row++) {
        for (size_type col = 0; col < a->get_size()[1]; col++) {
            c->at(row, col) = b->at(row, col) * diag_values[col];
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_DIAGONAL_RIGHT_APPLY_TO_DENSE_KERNEL);


template <typename ValueType, typename IndexType>
void apply_to_csr(std::shared_ptr<const ReferenceExecutor> exec,
                  const matrix::Diagonal<ValueType>* a,
                  const matrix::Csr<ValueType, IndexType>* b,
                  matrix::Csr<ValueType, IndexType>* c, bool inverse)
{
    const auto diag_values = a->get_const_values();
    c->copy_from(b);
    auto csr_values = c->get_values();
    const auto csr_row_ptrs = c->get_const_row_ptrs();

    for (size_type row = 0; row < c->get_size()[0]; row++) {
        const auto scal =
            inverse ? one<ValueType>() / diag_values[row] : diag_values[row];
        for (size_type idx = csr_row_ptrs[row]; idx < csr_row_ptrs[row + 1];
             idx++) {
            csr_values[idx] *= scal;
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DIAGONAL_APPLY_TO_CSR_KERNEL);


template <typename ValueType, typename IndexType>
void right_apply_to_csr(std::shared_ptr<const ReferenceExecutor> exec,
                        const matrix::Diagonal<ValueType>* a,
                        const matrix::Csr<ValueType, IndexType>* b,
                        matrix::Csr<ValueType, IndexType>* c)
{
    const auto diag_values = a->get_const_values();
    c->copy_from(b);
    auto csr_values = c->get_values();
    const auto csr_row_ptrs = c->get_const_row_ptrs();
    const auto csr_col_idxs = c->get_const_col_idxs();

    for (size_type row = 0; row < c->get_size()[0]; row++) {
        for (size_type idx = csr_row_ptrs[row]; idx < csr_row_ptrs[row + 1];
             idx++) {
            csr_values[idx] *= diag_values[csr_col_idxs[idx]];
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DIAGONAL_RIGHT_APPLY_TO_CSR_KERNEL);


template <typename ValueType, typename IndexType>
void fill_in_matrix_data(std::shared_ptr<const DefaultExecutor> exec,
                         const device_matrix_data<ValueType, IndexType>& data,
                         matrix::Diagonal<ValueType>* output)
{
    for (size_type i = 0; i < data.get_num_stored_elements(); i++) {
        const auto row = data.get_const_row_idxs()[i];
        if (row == data.get_const_col_idxs()[i]) {
            output->get_values()[row] = data.get_const_values()[i];
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DIAGONAL_FILL_IN_MATRIX_DATA_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_csr(std::shared_ptr<const ReferenceExecutor> exec,
                    const matrix::Diagonal<ValueType>* source,
                    matrix::Csr<ValueType, IndexType>* result)
{
    const auto size = source->get_size()[0];
    auto row_ptrs = result->get_row_ptrs();
    auto col_idxs = result->get_col_idxs();
    auto csr_values = result->get_values();
    const auto diag_values = source->get_const_values();

    for (size_type i = 0; i < size; i++) {
        row_ptrs[i] = i;
        col_idxs[i] = i;
        csr_values[i] = diag_values[i];
    }
    row_ptrs[size] = size;
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DIAGONAL_CONVERT_TO_CSR_KERNEL);


template <typename ValueType>
void conj_transpose(std::shared_ptr<const ReferenceExecutor> exec,
                    const matrix::Diagonal<ValueType>* orig,
                    matrix::Diagonal<ValueType>* trans)
{
    const auto size = orig->get_size()[0];
    const auto orig_values = orig->get_const_values();
    auto trans_values = trans->get_values();

    for (size_type i = 0; i < size; i++) {
        trans_values[i] = conj(orig_values[i]);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_DIAGONAL_CONJ_TRANSPOSE_KERNEL);


}  // namespace diagonal
}  // namespace reference
}  // namespace kernels
}  // namespace gko
