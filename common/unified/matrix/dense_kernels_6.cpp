/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2022, the Ginkgo authors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#include "core/matrix/dense_kernels.hpp"


#include <ginkgo/core/base/device_matrix_data.hpp>
#include <ginkgo/core/base/math.hpp>


#include "common/unified/base/kernel_launch.hpp"
#include "common/unified/base/kernel_launch_reduction.hpp"
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


template <typename ValueType, typename OutputType, typename IndexType>
void row_gather(std::shared_ptr<const DefaultExecutor> exec,
                const Array<IndexType>* row_idxs,
                const matrix::Dense<ValueType>* orig,
                matrix::Dense<OutputType>* row_collection)
{
    run_kernel(
        exec,
        [] GKO_KERNEL(auto row, auto col, auto orig, auto rows, auto gathered) {
            gathered(row, col) = orig(rows[row], col);
        },
        dim<2>{row_idxs->get_num_elems(), orig->get_size()[1]}, orig, *row_idxs,
        row_collection);
}

GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_AND_INDEX_TYPE_2(
    GKO_DECLARE_DENSE_ROW_GATHER_KERNEL);


template <typename ValueType, typename OutputType, typename IndexType>
void advanced_row_gather(std::shared_ptr<const DefaultExecutor> exec,
                         const matrix::Dense<ValueType>* alpha,
                         const Array<IndexType>* row_idxs,
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
        dim<2>{row_idxs->get_num_elems(), orig->get_size()[1]},
        alpha->get_const_values(), orig, *row_idxs, beta->get_const_values(),
        row_collection);
}

GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_AND_INDEX_TYPE_2(
    GKO_DECLARE_DENSE_ADVANCED_ROW_GATHER_KERNEL);


template <typename ValueType, typename IndexType>
void column_permute(std::shared_ptr<const DefaultExecutor> exec,
                    const Array<IndexType>* permutation_indices,
                    const matrix::Dense<ValueType>* orig,
                    matrix::Dense<ValueType>* column_permuted)
{
    run_kernel(
        exec,
        [] GKO_KERNEL(auto row, auto col, auto orig, auto perm, auto permuted) {
            permuted(row, col) = orig(row, perm[col]);
        },
        orig->get_size(), orig, *permutation_indices, column_permuted);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DENSE_COLUMN_PERMUTE_KERNEL);


template <typename ValueType, typename IndexType>
void inverse_row_permute(std::shared_ptr<const DefaultExecutor> exec,
                         const Array<IndexType>* permutation_indices,
                         const matrix::Dense<ValueType>* orig,
                         matrix::Dense<ValueType>* row_permuted)
{
    run_kernel(
        exec,
        [] GKO_KERNEL(auto row, auto col, auto orig, auto perm, auto permuted) {
            permuted(perm[row], col) = orig(row, col);
        },
        orig->get_size(), orig, *permutation_indices, row_permuted);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DENSE_INV_ROW_PERMUTE_KERNEL);


template <typename ValueType, typename IndexType>
void inverse_column_permute(std::shared_ptr<const DefaultExecutor> exec,
                            const Array<IndexType>* permutation_indices,
                            const matrix::Dense<ValueType>* orig,
                            matrix::Dense<ValueType>* column_permuted)
{
    run_kernel(
        exec,
        [] GKO_KERNEL(auto row, auto col, auto orig, auto perm, auto permuted) {
            permuted(row, perm[col]) = orig(row, col);
        },
        orig->get_size(), orig, *permutation_indices, column_permuted);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DENSE_INV_COLUMN_PERMUTE_KERNEL);


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

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_DENSE_EXTRACT_DIAGONAL_KERNEL);


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

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_INPLACE_ABSOLUTE_DENSE_KERNEL);


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

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_OUTPLACE_ABSOLUTE_DENSE_KERNEL);


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

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_MAKE_COMPLEX_KERNEL);


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

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_GET_REAL_KERNEL);


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

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_GET_IMAG_KERNEL);


}  // namespace dense
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko
