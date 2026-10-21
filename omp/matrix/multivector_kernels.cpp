// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/matrix/multivector_kernels.hpp"

#include <algorithm>

#include <omp.h>

#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/math.hpp>

#include "omp/components/atomic.hpp"


namespace gko {
namespace kernels {
namespace omp {
/**
 * @brief The MultiVector matrix format namespace.
 *
 */
namespace multivector {


template <typename ValueType>
void compute_dot_dispatch(std::shared_ptr<const DefaultExecutor> exec,
                          matrix::view::dense<const ValueType> x,
                          matrix::view::dense<const ValueType> y,
                          matrix::view::dense<ValueType> result,
                          array<char>& tmp)
{
    // OpenMP uses the unified kernel.
    compute_dot(exec, x, y, result, tmp);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_MULTIVECTOR_COMPUTE_DOT_DISPATCH_KERNEL);


template <typename ValueType>
void compute_conj_dot_dispatch(std::shared_ptr<const DefaultExecutor> exec,
                               matrix::view::dense<const ValueType> x,
                               matrix::view::dense<const ValueType> y,
                               matrix::view::dense<ValueType> result,
                               array<char>& tmp)
{
    compute_conj_dot(exec, x, y, result, tmp);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_MULTIVECTOR_COMPUTE_CONJ_DOT_DISPATCH_KERNEL);


template <typename ValueType>
void compute_norm2_dispatch(
    std::shared_ptr<const DefaultExecutor> exec,
    matrix::view::dense<const ValueType> x,
    matrix::view::dense<remove_complex<ValueType>> result, array<char>& tmp)
{
    compute_norm2(exec, x, result, tmp);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_MULTIVECTOR_COMPUTE_NORM2_DISPATCH_KERNEL);


template <typename ValueType>
void transpose(std::shared_ptr<const DefaultExecutor> exec,
               matrix::view::dense<const ValueType> orig,
               matrix::view::dense<ValueType> trans)
{
#pragma omp parallel for
    for (size_type i = 0; i < orig.size[0]; ++i) {
        for (size_type j = 0; j < orig.size[1]; ++j) {
            trans(j, i) = orig(i, j);
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_MULTIVECTOR_TRANSPOSE_KERNEL);


template <typename ValueType>
void conj_transpose(std::shared_ptr<const DefaultExecutor> exec,
                    matrix::view::dense<const ValueType> orig,
                    matrix::view::dense<ValueType> trans)
{
#pragma omp parallel for
    for (size_type i = 0; i < orig.size[0]; ++i) {
        for (size_type j = 0; j < orig.size[1]; ++j) {
            trans(j, i) = conj(orig(i, j));
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_MULTIVECTOR_CONJ_TRANSPOSE_KERNEL);


template <typename ValueType, typename IndexType>
void scatter_add(std::shared_ptr<const OmpExecutor> exec,
                 const IndexType* scatter_indices,
                 matrix::view::dense<const ValueType> source,
                 matrix::view::dense<ValueType> target)
{
    auto nrows = source.size[0];
    auto ncols = source.size[1];
    auto src_vals = source.values;
    auto src_stride = source.stride;
    auto tgt_vals = target.values;
    auto tgt_stride = target.stride;
#pragma omp parallel for
    for (size_type i = 0; i < nrows; ++i) {
        auto target_row = static_cast<size_type>(scatter_indices[i]);
        for (size_type j = 0; j < ncols; ++j) {
            atomic_add(tgt_vals[target_row * tgt_stride + j],
                       src_vals[i * src_stride + j]);
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_MULTIVECTOR_SCATTER_ADD_KERNEL);


}  // namespace multivector
}  // namespace omp
}  // namespace kernels
}  // namespace gko
