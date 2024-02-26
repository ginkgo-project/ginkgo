// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/matrix/fft_kernels.hpp"


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/dense.hpp>


namespace gko {
namespace kernels {
namespace dpcpp {
/**
 * @brief The FFT matrix format namespace.
 * @ref Fft
 * @ingroup fft
 */
namespace fft {


template <typename ValueType>
void fft(std::shared_ptr<const DefaultExecutor> exec,
         const matrix::Dense<std::complex<ValueType>>* b,
         matrix::Dense<std::complex<ValueType>>* x, bool inverse,
         array<char>& buffer) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_NON_COMPLEX_VALUE_TYPE(GKO_DECLARE_FFT_KERNEL);


template <typename ValueType>
void fft2(std::shared_ptr<const DefaultExecutor> exec,
          const matrix::Dense<std::complex<ValueType>>* b,
          matrix::Dense<std::complex<ValueType>>* x, size_type size1,
          size_type size2, bool inverse,
          array<char>& buffer) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_NON_COMPLEX_VALUE_TYPE(GKO_DECLARE_FFT2_KERNEL);


template <typename ValueType>
void fft3(std::shared_ptr<const DefaultExecutor> exec,
          const matrix::Dense<std::complex<ValueType>>* b,
          matrix::Dense<std::complex<ValueType>>* x, size_type size1,
          size_type size2, size_type size3, bool inverse,
          array<char>& buffer) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_NON_COMPLEX_VALUE_TYPE(GKO_DECLARE_FFT3_KERNEL);


}  // namespace fft
}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko
