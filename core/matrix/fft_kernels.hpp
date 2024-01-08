// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_MATRIX_FFT_KERNELS_HPP_
#define GKO_CORE_MATRIX_FFT_KERNELS_HPP_


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/dense.hpp>


namespace gko {
namespace kernels {


#define GKO_DECLARE_FFT_KERNEL(ValueType)                             \
    void fft(std::shared_ptr<const DefaultExecutor> exec,             \
             const matrix::Dense<std::complex<ValueType>>* b,         \
             matrix::Dense<std::complex<ValueType>>* x, bool inverse, \
             array<char>& buffer)

#define GKO_DECLARE_FFT2_KERNEL(ValueType)                                \
    void fft2(std::shared_ptr<const DefaultExecutor> exec,                \
              const matrix::Dense<std::complex<ValueType>>* b,            \
              matrix::Dense<std::complex<ValueType>>* x, size_type size1, \
              size_type size2, bool inverse, array<char>& buffer)

#define GKO_DECLARE_FFT3_KERNEL(ValueType)                                \
    void fft3(std::shared_ptr<const DefaultExecutor> exec,                \
              const matrix::Dense<std::complex<ValueType>>* b,            \
              matrix::Dense<std::complex<ValueType>>* x, size_type size1, \
              size_type size2, size_type size3, bool inverse,             \
              array<char>& buffer)


#define GKO_DECLARE_ALL_AS_TEMPLATES    \
    template <typename ValueType>       \
    GKO_DECLARE_FFT_KERNEL(ValueType);  \
    template <typename ValueType>       \
    GKO_DECLARE_FFT2_KERNEL(ValueType); \
    template <typename ValueType>       \
    GKO_DECLARE_FFT3_KERNEL(ValueType)


namespace omp {
namespace fft {

GKO_DECLARE_ALL_AS_TEMPLATES;

}  // namespace fft
}  // namespace omp


namespace cuda {
namespace fft {

GKO_DECLARE_ALL_AS_TEMPLATES;

}  // namespace fft
}  // namespace cuda


namespace reference {
namespace fft {

GKO_DECLARE_ALL_AS_TEMPLATES;

}  // namespace fft
}  // namespace reference


namespace hip {
namespace fft {

GKO_DECLARE_ALL_AS_TEMPLATES;

}  // namespace fft
}  // namespace hip


namespace dpcpp {
namespace fft {

GKO_DECLARE_ALL_AS_TEMPLATES;

}  // namespace fft
}  // namespace dpcpp


#undef GKO_DECLARE_ALL_AS_TEMPLATES


}  // namespace kernels
}  // namespace gko


#endif  // GKO_CORE_MATRIX_FFT_KERNELS_HPP_
