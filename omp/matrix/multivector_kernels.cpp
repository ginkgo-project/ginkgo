// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/matrix/multivector_kernels.hpp"

#include <algorithm>

#include <omp.h>

#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/math.hpp>


namespace gko {
namespace kernels {
namespace omp {
/**
 * @brief The MultiVector matrix format namespace.
 *
 * @ingroup dense
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


}  // namespace multivector
}  // namespace omp
}  // namespace kernels
}  // namespace gko
