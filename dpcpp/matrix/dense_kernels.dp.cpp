// SPDX-FileCopyrightText: 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/matrix/dense_kernels.hpp"

#include <oneapi/mkl.hpp>

#include <ginkgo/core/base/math.hpp>

#include "core/components/prefix_sum_kernels.hpp"
#include "core/matrix/multivector_kernels.hpp"
#include "dpcpp/base/config.hpp"
#include "dpcpp/base/dim3.dp.hpp"
#include "dpcpp/base/helper.hpp"
#include "dpcpp/base/math.hpp"
#include "dpcpp/base/onemkl_bindings.hpp"
#include "dpcpp/base/types.hpp"
#include "dpcpp/components/cooperative_groups.dp.hpp"
#include "dpcpp/components/reduction.dp.hpp"
#include "dpcpp/components/thread_ids.dp.hpp"
#include "dpcpp/components/uninitialized_array.hpp"
#include "dpcpp/synthesizer/implementation_selection.hpp"


namespace gko {
namespace kernels {
namespace dpcpp {
/**
 * @brief The dense matrix format namespace.
 *
 * @ingroup dense
 */
namespace dense {


template <typename ValueType>
void simple_apply(std::shared_ptr<const DefaultExecutor> exec,
                  matrix::view::dense<const ValueType> a,
                  matrix::view::dense<const ValueType> b,
                  matrix::view::dense<ValueType> c)
{
    using namespace oneapi::mkl;
    if constexpr (onemkl::is_supported<ValueType>::value) {
        if (b.stride != 0 && c.stride != 0) {
            if (a.size[1] > 0 && a.values && b.values && c.values) {
                oneapi::mkl::blas::row_major::gemm(
                    *exec->get_queue(), transpose::nontrans,
                    transpose::nontrans, c.size[0], c.size[1], a.size[1],
                    one<ValueType>(), as_device_type(a.values), a.stride,
                    as_device_type(b.values), b.stride, zero<ValueType>(),
                    as_device_type(c.values), c.stride);
            } else {
                multivector::fill(exec, c, zero<ValueType>());
            }
        }
    } else {
        GKO_NOT_IMPLEMENTED;
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_DENSE_SIMPLE_APPLY_KERNEL);


template <typename ValueType>
void apply(std::shared_ptr<const DefaultExecutor> exec,
           matrix::view::dense<const ValueType> alpha,
           matrix::view::dense<const ValueType> a,
           matrix::view::dense<const ValueType> b,
           matrix::view::dense<const ValueType> beta,
           matrix::view::dense<ValueType> c)
{
    using namespace oneapi::mkl;
    if constexpr (onemkl::is_supported<ValueType>::value) {
        if (b.stride != 0 && c.stride != 0) {
            if (a.size[1] > 0 && a.values && b.values && c.values) {
                oneapi::mkl::blas::row_major::gemm(
                    *exec->get_queue(), transpose::nontrans,
                    transpose::nontrans, c.size[0], c.size[1], a.size[1],
                    exec->copy_val_to_host(alpha.values),
                    as_device_type(a.values), a.stride,
                    as_device_type(b.values), b.stride,
                    exec->copy_val_to_host(beta.values),
                    as_device_type(c.values), c.stride);
            } else {
                dense::scale(exec, beta, c);
            }
        }
    } else {
        GKO_NOT_IMPLEMENTED;
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_DENSE_APPLY_KERNEL);


}  // namespace dense
}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko
