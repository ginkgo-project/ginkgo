// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/preconditioner/jacobi_kernels.hpp"

#include <ginkgo/core/base/math.hpp>

#include "common/unified/base/kernel_launch.hpp"


namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {
/**
 * @brief The Jacobi preconditioner namespace.
 *
 * @ingroup jacobi
 */
namespace jacobi {


template <typename ValueType>
void scalar_conj(std::shared_ptr<const DefaultExecutor> exec,
                 const array<ValueType>& diag, array<ValueType>& conj_diag)
{
    run_kernel(
        exec,
        [] GKO_KERNEL(auto elem, auto diag, auto conj_diag) {
            conj_diag[elem] = conj(diag[elem]);
        },
        diag.get_size(), diag, conj_diag);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_JACOBI_SCALAR_CONJ_KERNEL);


template <typename ValueType>
void invert_diagonal(std::shared_ptr<const DefaultExecutor> exec,
                     const array<ValueType>& diag, array<ValueType>& inv_diag)
{
    run_kernel(
        exec,
        [] GKO_KERNEL(auto elem, auto diag, auto inv_diag) {
            // if the diagonal is zero, we use 1 for in the inverted result.
            inv_diag[elem] = is_zero(diag[elem]) ? one(diag[elem])
                                                 : one(diag[elem]) / diag[elem];
        },
        diag.get_size(), diag, inv_diag);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_JACOBI_INVERT_DIAGONAL_KERNEL);


template <typename ValueType>
void scalar_apply(std::shared_ptr<const DefaultExecutor> exec,
                  const array<ValueType>& diag,
                  const matrix::Dense<ValueType>* alpha,
                  const matrix::Dense<ValueType>* b,
                  const matrix::Dense<ValueType>* beta,
                  matrix::Dense<ValueType>* x)
{
    if (alpha->get_size()[1] > 1) {
        run_kernel(
            exec,
            [] GKO_KERNEL(auto row, auto col, auto diag, auto alpha, auto b,
                          auto beta, auto x) {
                x(row, col) = beta[col] * x(row, col) +
                              alpha[col] * b(row, col) * diag[row];
            },
            x->get_size(), diag, alpha->get_const_values(), b,
            beta->get_const_values(), x);
    } else {
        run_kernel(
            exec,
            [] GKO_KERNEL(auto row, auto col, auto diag, auto alpha, auto b,
                          auto beta, auto x) {
                x(row, col) =
                    beta[0] * x(row, col) + alpha[0] * b(row, col) * diag[row];
            },
            x->get_size(), diag, alpha->get_const_values(), b,
            beta->get_const_values(), x);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_JACOBI_SCALAR_APPLY_KERNEL);


template <typename ValueType>
void simple_scalar_apply(std::shared_ptr<const DefaultExecutor> exec,
                         const array<ValueType>& diag,
                         const matrix::Dense<ValueType>* b,
                         matrix::Dense<ValueType>* x)
{
    run_kernel(
        exec,
        [] GKO_KERNEL(auto row, auto col, auto diag, auto b, auto x) {
            x(row, col) = b(row, col) * diag[row];
        },
        x->get_size(), diag, b, x);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_JACOBI_SIMPLE_SCALAR_APPLY_KERNEL);


template <typename ValueType>
void scalar_convert_to_dense(std::shared_ptr<const DefaultExecutor> exec,
                             const array<ValueType>& blocks,
                             matrix::Dense<ValueType>* result)
{
    run_kernel(
        exec,
        [] GKO_KERNEL(auto row, auto col, auto diag, auto result) {
            result(row, col) = zero(diag[row]);
            if (row == col) {
                result(row, col) = diag[row];
            }
        },
        result->get_size(), blocks, result);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_JACOBI_SCALAR_CONVERT_TO_DENSE_KERNEL);


template <typename ValueType, typename IndexType>
void scalar_l1(std::shared_ptr<const DefaultExecutor> exec,
               const matrix::Csr<ValueType, IndexType>* csr,
               matrix::Diagonal<ValueType>* diag)
{
    run_kernel(
        exec,
        [] GKO_KERNEL(auto row, auto row_ptrs, auto col_idxs, auto vals,
                      auto diag) {
            auto off_diag = zero(vals[0]);
            for (auto i = row_ptrs[row]; i < row_ptrs[row + 1]; i++) {
                if (col_idxs[i] == row) {
                    continue;
                }
                off_diag += abs(vals[i]);
            }
            // TODO: It is unclear effect when this applies to negative diagonal
            // value. The reference paper only discusses the positive diagonal
            // value.
            diag[row] += off_diag;
        },
        csr->get_size()[0], csr->get_const_row_ptrs(),
        csr->get_const_col_idxs(), csr->get_const_values(), diag->get_values());
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_JACOBI_SCALAR_L1_KERNEL);


template <typename ValueType, typename IndexType>
void block_l1(std::shared_ptr<const DefaultExecutor> exec, size_type num_blocks,
              const array<IndexType>& block_ptrs,
              matrix::Csr<ValueType, IndexType>* csr)
{
    // Note: there are two another possible ways to do it.
    // 1. allocate block_num * max_block_size -> have enough thread for rows in
    // block, and run the process if the threads runs on a valid row.
    // 2. allocate thread per row -> get the block first, and use it as
    // diagonal/off-diagonal condition.
    run_kernel(
        exec,
        [] GKO_KERNEL(auto block_id, auto block_ptrs, auto row_ptrs,
                      auto col_idxs, auto vals) {
            auto start = block_ptrs[block_id];
            auto end = block_ptrs[block_id + 1];
            for (auto row = start; row < end; row++) {
                auto off_diag = zero(vals[0]);
                IndexType diag_idx = -1;
                for (auto i = row_ptrs[row]; i < row_ptrs[row + 1]; i++) {
                    auto col = col_idxs[i];
                    if (col >= start && col < end) {
                        if (col == row) {
                            diag_idx = i;
                        }
                        continue;
                    }
                    off_diag += abs(vals[i]);
                }
                // TODO: It is unclear effect when this applies to negative
                // diagonal value. The reference paper only discusses the
                // positive diagonal value.
                vals[diag_idx] += off_diag;
            }
        },
        num_blocks, block_ptrs.get_const_data(), csr->get_const_row_ptrs(),
        csr->get_const_col_idxs(), csr->get_values());
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_JACOBI_BLOCK_L1_KERNEL);


}  // namespace jacobi
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko
