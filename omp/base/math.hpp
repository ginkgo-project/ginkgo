// SPDX-FileCopyrightText: 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_OMP_BASE_MATH_H_
#define GKO_OMP_BASE_MATH_H_

#include <ginkgo/core/base/math.hpp>


namespace gko {

template <typename T>
using device_numeric_limits = std::numeric_limits<T>;

namespace kernels {
namespace omp {


template <typename T>
using to_complex = gko::to_complex<T>;

template <typename T>
using device_type = T;

using device_half = gko::half;
using device_bfloat16 = gko::bfloat16;
namespace complex_namespace = std;


}  // namespace omp
}  // namespace kernels

}  // namespace gko


#endif  // GKO_OMP_BASE_MATH_H_
