// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/matrix/dense_kernels.hpp"


#include <ginkgo/core/base/device_matrix_data.hpp>
#include <ginkgo/core/base/math.hpp>


#include "common/unified/base/kernel_launch.hpp"
#include "common/unified/base/kernel_launch_reduction.hpp"
#include "core/base/array_access.hpp"
#include "core/base/mixed_precision_types.hpp"
#include "core/components/prefix_sum_kernels.hpp"


namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {
/**
 * @brief The Dense matrix format namespace.
 *
 * @ingroup dense
 */
namespace dense {


template <typename InValueType, typename OutValueType>
void copy(std::shared_ptr<const DefaultExecutor> exec,
          const matrix::Dense<InValueType>* input,
          matrix::Dense<OutValueType>* output)
{
    run_kernel(
        exec,
        [] GKO_KERNEL(auto row, auto col, auto input, auto output) {
            output(row, col) = input(row, col);
        },
        input->get_size(), input, output);
}


template <typename ValueType>
void fill(std::shared_ptr<const DefaultExecutor> exec,
          matrix::Dense<ValueType>* mat, ValueType value)
{
    run_kernel(
        exec,
        [] GKO_KERNEL(auto row, auto col, auto mat, auto value) {
            mat(row, col) = value;
        },
        mat->get_size(), mat, value);
}


template <typename ValueType, typename IndexType>
void fill_in_matrix_data(std::shared_ptr<const DefaultExecutor> exec,
                         const device_matrix_data<ValueType, IndexType>& data,
                         matrix::Dense<ValueType>* output)
{
    run_kernel(
        exec,
        [] GKO_KERNEL(auto i, auto row, auto col, auto val, auto output) {
            output(row[i], col[i]) = val[i];
        },
        data.get_num_stored_elements(), data.get_const_row_idxs(),
        data.get_const_col_idxs(), data.get_const_values(), output);
}


template <typename ValueType, typename ScalarType>
void scale(std::shared_ptr<const DefaultExecutor> exec,
           const matrix::Dense<ScalarType>* alpha, matrix::Dense<ValueType>* x)
{
    if (alpha->get_size()[1] > 1) {
        run_kernel(
            exec,
            [] GKO_KERNEL(auto row, auto col, auto alpha, auto x) {
                x(row, col) *= alpha[col];
            },
            x->get_size(), alpha->get_const_values(), x);
    } else {
        run_kernel(
            exec,
            [] GKO_KERNEL(auto row, auto col, auto alpha, auto x) {
                x(row, col) *= alpha[0];
            },
            x->get_size(), alpha->get_const_values(), x);
    }
}


template <typename ValueType, typename ScalarType>
void inv_scale(std::shared_ptr<const DefaultExecutor> exec,
               const matrix::Dense<ScalarType>* alpha,
               matrix::Dense<ValueType>* x)
{
    if (alpha->get_size()[1] > 1) {
        run_kernel(
            exec,
            [] GKO_KERNEL(auto row, auto col, auto alpha, auto x) {
                x(row, col) /= alpha[col];
            },
            x->get_size(), alpha->get_const_values(), x);
    } else {
        run_kernel(
            exec,
            [] GKO_KERNEL(auto row, auto col, auto alpha, auto x) {
                x(row, col) /= alpha[0];
            },
            x->get_size(), alpha->get_const_values(), x);
    }
}


template <typename ValueType, typename ScalarType>
void add_scaled(std::shared_ptr<const DefaultExecutor> exec,
                const matrix::Dense<ScalarType>* alpha,
                const matrix::Dense<ValueType>* x, matrix::Dense<ValueType>* y)
{
    if (alpha->get_size()[1] > 1) {
        run_kernel(
            exec,
            [] GKO_KERNEL(auto row, auto col, auto alpha, auto x, auto y) {
                y(row, col) += alpha[col] * x(row, col);
            },
            x->get_size(), alpha->get_const_values(), x, y);
    } else {
        run_kernel(
            exec,
            [] GKO_KERNEL(auto row, auto col, auto alpha, auto x, auto y) {
                y(row, col) += alpha[0] * x(row, col);
            },
            x->get_size(), alpha->get_const_values(), x, y);
    }
}


template <typename ValueType, typename ScalarType>
void sub_scaled(std::shared_ptr<const DefaultExecutor> exec,
                const matrix::Dense<ScalarType>* alpha,
                const matrix::Dense<ValueType>* x, matrix::Dense<ValueType>* y)
{
    if (alpha->get_size()[1] > 1) {
        run_kernel(
            exec,
            [] GKO_KERNEL(auto row, auto col, auto alpha, auto x, auto y) {
                y(row, col) -= alpha[col] * x(row, col);
            },
            x->get_size(), alpha->get_const_values(), x, y);
    } else {
        run_kernel(
            exec,
            [] GKO_KERNEL(auto row, auto col, auto alpha, auto x, auto y) {
                y(row, col) -= alpha[0] * x(row, col);
            },
            x->get_size(), alpha->get_const_values(), x, y);
    }
}


template <typename ValueType>
void add_scaled_diag(std::shared_ptr<const DefaultExecutor> exec,
                     const matrix::Dense<ValueType>* alpha,
                     const matrix::Diagonal<ValueType>* x,
                     matrix::Dense<ValueType>* y)
{
    const auto diag_values = x->get_const_values();
    run_kernel(
        exec,
        [] GKO_KERNEL(auto i, auto alpha, auto diag, auto y) {
            y(i, i) += alpha[0] * diag[i];
        },
        x->get_size()[0], alpha->get_const_values(), x->get_const_values(), y);
}


template <typename ValueType>
void sub_scaled_diag(std::shared_ptr<const DefaultExecutor> exec,
                     const matrix::Dense<ValueType>* alpha,
                     const matrix::Diagonal<ValueType>* x,
                     matrix::Dense<ValueType>* y)
{
    const auto diag_values = x->get_const_values();
    run_kernel(
        exec,
        [] GKO_KERNEL(auto i, auto alpha, auto diag, auto y) {
            y(i, i) -= alpha[0] * diag[i];
        },
        x->get_size()[0], alpha->get_const_values(), x->get_const_values(), y);
}


template <typename ValueType>
void compute_dot(std::shared_ptr<const DefaultExecutor> exec,
                 const matrix::Dense<ValueType>* x,
                 const matrix::Dense<ValueType>* y,
                 matrix::Dense<ValueType>* result, array<char>& tmp)
{
    run_kernel_col_reduction_cached(
        exec,
        [] GKO_KERNEL(auto i, auto j, auto x, auto y) {
            return x(i, j) * y(i, j);
        },
        GKO_KERNEL_REDUCE_SUM(ValueType), result->get_values(), x->get_size(),
        tmp, x, y);
}


template <typename ValueType>
void compute_conj_dot(std::shared_ptr<const DefaultExecutor> exec,
                      const matrix::Dense<ValueType>* x,
                      const matrix::Dense<ValueType>* y,
                      matrix::Dense<ValueType>* result, array<char>& tmp)
{
    run_kernel_col_reduction_cached(
        exec,
        [] GKO_KERNEL(auto i, auto j, auto x, auto y) {
            return conj(x(i, j)) * y(i, j);
        },
        GKO_KERNEL_REDUCE_SUM(ValueType), result->get_values(), x->get_size(),
        tmp, x, y);
}


template <typename ValueType>
void compute_norm2(std::shared_ptr<const DefaultExecutor> exec,
                   const matrix::Dense<ValueType>* x,
                   matrix::Dense<remove_complex<ValueType>>* result,
                   array<char>& tmp)
{
    run_kernel_col_reduction_cached(
        exec,
        [] GKO_KERNEL(auto i, auto j, auto x) { return squared_norm(x(i, j)); },
        [] GKO_KERNEL(auto a, auto b) { return a + b; },
        [] GKO_KERNEL(auto a) { return sqrt(a); }, remove_complex<ValueType>{},
        result->get_values(), x->get_size(), tmp, x);
}

template <typename ValueType>
void compute_norm1(std::shared_ptr<const DefaultExecutor> exec,
                   const matrix::Dense<ValueType>* x,
                   matrix::Dense<remove_complex<ValueType>>* result,
                   array<char>& tmp)
{
    run_kernel_col_reduction_cached(
        exec, [] GKO_KERNEL(auto i, auto j, auto x) { return abs(x(i, j)); },
        GKO_KERNEL_REDUCE_SUM(remove_complex<ValueType>), result->get_values(),
        x->get_size(), tmp, x);
}


template <typename ValueType>
void compute_mean(std::shared_ptr<const DefaultExecutor> exec,
                  const matrix::Dense<ValueType>* x,
                  matrix::Dense<ValueType>* result, array<char>& tmp)
{
    using ValueType_nc = gko::remove_complex<ValueType>;
    run_kernel_col_reduction_cached(
        exec,
        [] GKO_KERNEL(auto i, auto j, auto x, auto inv_total_size) {
            return x(i, j) * inv_total_size;
        },
        GKO_KERNEL_REDUCE_SUM(ValueType), result->get_values(), x->get_size(),
        tmp, x, ValueType_nc{1.} / x->get_size()[0]);
}


template <typename ValueType>
void compute_max_nnz_per_row(std::shared_ptr<const DefaultExecutor> exec,
                             const matrix::Dense<ValueType>* source,
                             size_type& result)
{
    array<size_type> partial{exec, source->get_size()[0] + 1};
    count_nonzeros_per_row(exec, source, partial.get_data());
    run_kernel_reduction(
        exec, [] GKO_KERNEL(auto i, auto partial) { return partial[i]; },
        GKO_KERNEL_REDUCE_MAX(size_type),
        partial.get_data() + source->get_size()[0], source->get_size()[0],
        partial);
    result = get_element(partial, source->get_size()[0]);
}


template <typename ValueType>
void compute_slice_sets(std::shared_ptr<const DefaultExecutor> exec,
                        const matrix::Dense<ValueType>* source,
                        size_type slice_size, size_type stride_factor,
                        size_type* slice_sets, size_type* slice_lengths)
{
    const auto num_rows = source->get_size()[0];
    array<size_type> row_nnz{exec, num_rows};
    count_nonzeros_per_row(exec, source, row_nnz.get_data());
    const auto num_slices =
        static_cast<size_type>(ceildiv(num_rows, slice_size));
    run_kernel_row_reduction(
        exec,
        [] GKO_KERNEL(auto slice, auto local_row, auto row_nnz, auto slice_size,
                      auto stride_factor, auto num_rows) {
            const auto row = slice * slice_size + local_row;
            return row < num_rows ? static_cast<size_type>(
                                        ceildiv(row_nnz[row], stride_factor) *
                                        stride_factor)
                                  : size_type{};
        },
        GKO_KERNEL_REDUCE_MAX(size_type), slice_lengths, 1,
        gko::dim<2>{num_slices, slice_size}, row_nnz, slice_size, stride_factor,
        num_rows);
    exec->copy(num_slices, slice_lengths, slice_sets);
    components::prefix_sum_nonnegative(exec, slice_sets, num_slices + 1);
}


template <typename ValueType, typename IndexType>
void count_nonzeros_per_row(std::shared_ptr<const DefaultExecutor> exec,
                            const matrix::Dense<ValueType>* mtx,
                            IndexType* result)
{
    run_kernel_row_reduction(
        exec,
        [] GKO_KERNEL(auto i, auto j, auto mtx) {
            return is_nonzero(mtx(i, j)) ? 1 : 0;
        },
        GKO_KERNEL_REDUCE_SUM(IndexType), result, 1, mtx->get_size(), mtx);
}


template <typename ValueType>
void compute_squared_norm2(std::shared_ptr<const DefaultExecutor> exec,
                           const matrix::Dense<ValueType>* x,
                           matrix::Dense<remove_complex<ValueType>>* result,
                           array<char>& tmp)
{
    run_kernel_col_reduction_cached(
        exec,
        [] GKO_KERNEL(auto i, auto j, auto x) { return squared_norm(x(i, j)); },
        GKO_KERNEL_REDUCE_SUM(remove_complex<ValueType>), result->get_values(),
        x->get_size(), tmp, x);
}


template <typename ValueType>
void compute_sqrt(std::shared_ptr<const DefaultExecutor> exec,
                  matrix::Dense<ValueType>* x)
{
    run_kernel(
        exec,
        [] GKO_KERNEL(auto row, auto col, auto x) {
            x(row, col) = sqrt(x(row, col));
        },
        x->get_size(), x);
}


template <typename ValueType, typename IndexType>
void symm_permute(std::shared_ptr<const DefaultExecutor> exec,
                  const IndexType* permutation_indices,
                  const matrix::Dense<ValueType>* orig,
                  matrix::Dense<ValueType>* permuted)
{
    run_kernel(
        exec,
        [] GKO_KERNEL(auto row, auto col, auto orig, auto perm, auto permuted) {
            permuted(row, col) = orig(perm[row], perm[col]);
        },
        orig->get_size(), orig, permutation_indices, permuted);
}


template <typename ValueType, typename IndexType>
void inv_symm_permute(std::shared_ptr<const DefaultExecutor> exec,
                      const IndexType* permutation_indices,
                      const matrix::Dense<ValueType>* orig,
                      matrix::Dense<ValueType>* permuted)
{
    run_kernel(
        exec,
        [] GKO_KERNEL(auto row, auto col, auto orig, auto perm, auto permuted) {
            permuted(perm[row], perm[col]) = orig(row, col);
        },
        orig->get_size(), orig, permutation_indices, permuted);
}


template <typename ValueType, typename IndexType>
void nonsymm_permute(std::shared_ptr<const DefaultExecutor> exec,
                     const IndexType* row_permutation_indices,
                     const IndexType* column_permutation_indices,
                     const matrix::Dense<ValueType>* orig,
                     matrix::Dense<ValueType>* permuted)
{
    run_kernel(
        exec,
        [] GKO_KERNEL(auto row, auto col, auto orig, auto row_perm,
                      auto col_perm, auto permuted) {
            permuted(row, col) = orig(row_perm[row], col_perm[col]);
        },
        orig->get_size(), orig, row_permutation_indices,
        column_permutation_indices, permuted);
}


template <typename ValueType, typename IndexType>
void inv_nonsymm_permute(std::shared_ptr<const DefaultExecutor> exec,
                         const IndexType* row_permutation_indices,
                         const IndexType* column_permutation_indices,
                         const matrix::Dense<ValueType>* orig,
                         matrix::Dense<ValueType>* permuted)
{
    run_kernel(
        exec,
        [] GKO_KERNEL(auto row, auto col, auto orig, auto row_perm,
                      auto col_perm, auto permuted) {
            permuted(row_perm[row], col_perm[col]) = orig(row, col);
        },
        orig->get_size(), orig, row_permutation_indices,
        column_permutation_indices, permuted);
}


template <typename ValueType, typename OutputType, typename IndexType>
void row_gather(std::shared_ptr<const DefaultExecutor> exec,
                const IndexType* row_idxs, const matrix::Dense<ValueType>* orig,
                matrix::Dense<OutputType>* row_collection)
{
    run_kernel(
        exec,
        [] GKO_KERNEL(auto row, auto col, auto orig, auto rows, auto gathered) {
            gathered(row, col) = orig(rows[row], col);
        },
        row_collection->get_size(), orig, row_idxs, row_collection);
}


template <typename ValueType, typename OutputType, typename IndexType>
void advanced_row_gather(std::shared_ptr<const DefaultExecutor> exec,
                         const matrix::Dense<ValueType>* alpha,
                         const IndexType* row_idxs,
                         const matrix::Dense<ValueType>* orig,
                         const matrix::Dense<ValueType>* beta,
                         matrix::Dense<OutputType>* row_collection)
{
    run_kernel(
        exec,
        [] GKO_KERNEL(auto row, auto col, auto alpha, auto orig, auto rows,
                      auto beta, auto gathered) {
            using type = device_type<highest_precision<ValueType, OutputType>>;
            gathered(row, col) =
                static_cast<type>(alpha[0] * orig(rows[row], col)) +
                static_cast<type>(beta[0]) *
                    static_cast<type>(gathered(row, col));
        },
        row_collection->get_size(), alpha->get_const_values(), orig, row_idxs,
        beta->get_const_values(), row_collection);
}


template <typename ValueType, typename IndexType>
void col_permute(std::shared_ptr<const DefaultExecutor> exec,
                 const IndexType* permutation_indices,
                 const matrix::Dense<ValueType>* orig,
                 matrix::Dense<ValueType>* col_permuted)
{
    run_kernel(
        exec,
        [] GKO_KERNEL(auto row, auto col, auto orig, auto perm, auto permuted) {
            permuted(row, col) = orig(row, perm[col]);
        },
        orig->get_size(), orig, permutation_indices, col_permuted);
}


template <typename ValueType, typename IndexType>
void inv_row_permute(std::shared_ptr<const DefaultExecutor> exec,
                     const IndexType* permutation_indices,
                     const matrix::Dense<ValueType>* orig,
                     matrix::Dense<ValueType>* row_permuted)
{
    run_kernel(
        exec,
        [] GKO_KERNEL(auto row, auto col, auto orig, auto perm, auto permuted) {
            permuted(perm[row], col) = orig(row, col);
        },
        orig->get_size(), orig, permutation_indices, row_permuted);
}


template <typename ValueType, typename IndexType>
void inv_col_permute(std::shared_ptr<const DefaultExecutor> exec,
                     const IndexType* permutation_indices,
                     const matrix::Dense<ValueType>* orig,
                     matrix::Dense<ValueType>* col_permuted)
{
    run_kernel(
        exec,
        [] GKO_KERNEL(auto row, auto col, auto orig, auto perm, auto permuted) {
            permuted(row, perm[col]) = orig(row, col);
        },
        orig->get_size(), orig, permutation_indices, col_permuted);
}


template <typename ValueType, typename IndexType>
void symm_scale_permute(std::shared_ptr<const DefaultExecutor> exec,
                        const ValueType* scale, const IndexType* perm,
                        const matrix::Dense<ValueType>* orig,
                        matrix::Dense<ValueType>* permuted)
{
    run_kernel(
        exec,
        [] GKO_KERNEL(auto i, auto j, auto scale, auto perm, auto orig,
                      auto permuted) {
            const auto row = perm[i];
            const auto col = perm[j];
            permuted(i, j) = scale[row] * scale[col] * orig(row, col);
        },
        orig->get_size(), scale, perm, orig, permuted);
}


template <typename ValueType, typename IndexType>
void inv_symm_scale_permute(std::shared_ptr<const DefaultExecutor> exec,
                            const ValueType* scale, const IndexType* perm,
                            const matrix::Dense<ValueType>* orig,
                            matrix::Dense<ValueType>* permuted)
{
    run_kernel(
        exec,
        [] GKO_KERNEL(auto i, auto j, auto scale, auto perm, auto orig,
                      auto permuted) {
            const auto row = perm[i];
            const auto col = perm[j];
            permuted(row, col) = orig(i, j) / (scale[row] * scale[col]);
        },
        orig->get_size(), scale, perm, orig, permuted);
}


template <typename ValueType, typename IndexType>
void nonsymm_scale_permute(std::shared_ptr<const DefaultExecutor> exec,
                           const ValueType* row_scale,
                           const IndexType* row_perm,
                           const ValueType* col_scale,
                           const IndexType* col_perm,
                           const matrix::Dense<ValueType>* orig,
                           matrix::Dense<ValueType>* permuted)
{
    run_kernel(
        exec,
        [] GKO_KERNEL(auto i, auto j, auto row_scale, auto row_perm,
                      auto col_scale, auto col_perm, auto orig, auto permuted) {
            const auto row = row_perm[i];
            const auto col = col_perm[j];
            permuted(i, j) = row_scale[row] * col_scale[col] * orig(row, col);
        },
        orig->get_size(), row_scale, row_perm, col_scale, col_perm, orig,
        permuted);
}


template <typename ValueType, typename IndexType>
void inv_nonsymm_scale_permute(std::shared_ptr<const DefaultExecutor> exec,
                               const ValueType* row_scale,
                               const IndexType* row_perm,
                               const ValueType* col_scale,
                               const IndexType* col_perm,
                               const matrix::Dense<ValueType>* orig,
                               matrix::Dense<ValueType>* permuted)
{
    run_kernel(
        exec,
        [] GKO_KERNEL(auto i, auto j, auto row_scale, auto row_perm,
                      auto col_scale, auto col_perm, auto orig, auto permuted) {
            const auto row = row_perm[i];
            const auto col = col_perm[j];
            permuted(row, col) = orig(i, j) / (row_scale[row] * col_scale[col]);
        },
        orig->get_size(), row_scale, row_perm, col_scale, col_perm, orig,
        permuted);
}


template <typename ValueType, typename IndexType>
void row_scale_permute(std::shared_ptr<const DefaultExecutor> exec,
                       const ValueType* scale, const IndexType* perm,
                       const matrix::Dense<ValueType>* orig,
                       matrix::Dense<ValueType>* permuted)
{
    run_kernel(
        exec,
        [] GKO_KERNEL(auto i, auto j, auto scale, auto perm, auto orig,
                      auto permuted) {
            const auto row = perm[i];
            permuted(i, j) = scale[row] * orig(row, j);
        },
        orig->get_size(), scale, perm, orig, permuted);
}


template <typename ValueType, typename IndexType>
void inv_row_scale_permute(std::shared_ptr<const DefaultExecutor> exec,
                           const ValueType* scale, const IndexType* perm,
                           const matrix::Dense<ValueType>* orig,
                           matrix::Dense<ValueType>* permuted)
{
    run_kernel(
        exec,
        [] GKO_KERNEL(auto i, auto j, auto scale, auto perm, auto orig,
                      auto permuted) {
            const auto row = perm[i];
            permuted(row, j) = orig(i, j) / scale[row];
        },
        orig->get_size(), scale, perm, orig, permuted);
}


template <typename ValueType, typename IndexType>
void col_scale_permute(std::shared_ptr<const DefaultExecutor> exec,
                       const ValueType* scale, const IndexType* perm,
                       const matrix::Dense<ValueType>* orig,
                       matrix::Dense<ValueType>* permuted)
{
    run_kernel(
        exec,
        [] GKO_KERNEL(auto i, auto j, auto scale, auto perm, auto orig,
                      auto permuted) {
            const auto col = perm[j];
            permuted(i, j) = scale[col] * orig(i, col);
        },
        orig->get_size(), scale, perm, orig, permuted);
}


template <typename ValueType, typename IndexType>
void inv_col_scale_permute(std::shared_ptr<const DefaultExecutor> exec,
                           const ValueType* scale, const IndexType* perm,
                           const matrix::Dense<ValueType>* orig,
                           matrix::Dense<ValueType>* permuted)
{
    run_kernel(
        exec,
        [] GKO_KERNEL(auto i, auto j, auto scale, auto perm, auto orig,
                      auto permuted) {
            const auto col = perm[j];
            permuted(i, col) = orig(i, j) / scale[col];
        },
        orig->get_size(), scale, perm, orig, permuted);
}


template <typename ValueType>
void extract_diagonal(std::shared_ptr<const DefaultExecutor> exec,
                      const matrix::Dense<ValueType>* orig,
                      matrix::Diagonal<ValueType>* diag)
{
    run_kernel(
        exec,
        [] GKO_KERNEL(auto i, auto orig, auto diag) { diag[i] = orig(i, i); },
        diag->get_size()[0], orig, diag->get_values());
}


template <typename ValueType>
void inplace_absolute_dense(std::shared_ptr<const DefaultExecutor> exec,
                            matrix::Dense<ValueType>* source)
{
    run_kernel(
        exec,
        [] GKO_KERNEL(auto row, auto col, auto source) {
            source(row, col) = abs(source(row, col));
        },
        source->get_size(), source);
}


template <typename ValueType>
void outplace_absolute_dense(std::shared_ptr<const DefaultExecutor> exec,
                             const matrix::Dense<ValueType>* source,
                             matrix::Dense<remove_complex<ValueType>>* result)
{
    run_kernel(
        exec,
        [] GKO_KERNEL(auto row, auto col, auto source, auto result) {
            result(row, col) = abs(source(row, col));
        },
        source->get_size(), source, result);
}


template <typename ValueType>
void make_complex(std::shared_ptr<const DefaultExecutor> exec,
                  const matrix::Dense<ValueType>* source,
                  matrix::Dense<to_complex<ValueType>>* result)
{
    run_kernel(
        exec,
        [] GKO_KERNEL(auto row, auto col, auto source, auto result) {
            result(row, col) = source(row, col);
        },
        source->get_size(), source, result);
}


template <typename ValueType>
void get_real(std::shared_ptr<const DefaultExecutor> exec,
              const matrix::Dense<ValueType>* source,
              matrix::Dense<remove_complex<ValueType>>* result)
{
    run_kernel(
        exec,
        [] GKO_KERNEL(auto row, auto col, auto source, auto result) {
            result(row, col) = real(source(row, col));
        },
        source->get_size(), source, result);
}


template <typename ValueType>
void get_imag(std::shared_ptr<const DefaultExecutor> exec,
              const matrix::Dense<ValueType>* source,
              matrix::Dense<remove_complex<ValueType>>* result)
{
    run_kernel(
        exec,
        [] GKO_KERNEL(auto row, auto col, auto source, auto result) {
            result(row, col) = imag(source(row, col));
        },
        source->get_size(), source, result);
}


template <typename ValueType, typename ScalarType>
void add_scaled_identity(std::shared_ptr<const DefaultExecutor> exec,
                         const matrix::Dense<ScalarType>* const alpha,
                         const matrix::Dense<ScalarType>* const beta,
                         matrix::Dense<ValueType>* const mtx)
{
    run_kernel(
        exec,
        [] GKO_KERNEL(auto row, auto col, auto alpha, auto beta, auto mtx) {
            mtx(row, col) = beta[0] * mtx(row, col);
            if (row == col) {
                mtx(row, row) += alpha[0];
            }
        },
        mtx->get_size(), alpha->get_const_values(), beta->get_const_values(),
        mtx);
}


}  // namespace dense
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko
