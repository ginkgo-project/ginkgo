// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/matrix/dense_kernels.hpp"

#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/range_accessors.hpp>
#include <ginkgo/core/matrix/dense.hpp>

#include "core/base/utils.hpp"
#include "core/components/prefix_sum_kernels.hpp"
#include "csl/cerebras_handle.hpp"
#include "ginkgo/core/base/types.hpp"

//#include <ginkgo/csl/cerebras_layout.hpp>

#include <type_traits>


namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {
/**
 * @brief The Dense matrix format namespace.
 *
 * @ingroup dense
 */
namespace dense {

template <typename ValueType>
void simple_apply(std::shared_ptr<const DefaultExecutor> exec,
                  const matrix::Dense<ValueType>* a,
                  const matrix::Dense<ValueType>* b,
                  matrix::Dense<ValueType>* c)
{
    int size1 = a->get_size()[0];
    int size2 = a->get_size()[1];
    int size3 = b->get_size()[0];
    int size4 = b->get_size()[1];
    if ((size1 == M) && (size2 == M) && (size3 == M) && (size4 == M)) {
        exec->get_handle()->copy_h2d("A", a->get_const_values(), size1 * size2,
                                     0, 0, size1, size2, (M * M) / (G * G),
                                     false, true);
        exec->get_handle()->copy_h2d("B", b->get_const_values(), size3 * size4,
                                     0, 0, size3, size4, (M * M) / (G * G),
                                     false, true);
        exec->get_handle()->call_func("main");
        exec->get_handle()->copy_d2h("C", c->get_values(), size1 * size4, 0, 0,
                                     size1, size4, (M * M) / (G * G), false,
                                     true);
    } else {
        GKO_NOT_IMPLEMENTED;
    }
}

template void simple_apply(std::shared_ptr<const DefaultExecutor> exec,
                           const matrix::Dense<float>* a,
                           const matrix::Dense<float>* b,
                           matrix::Dense<float>* c);


}  // namespace dense
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko
