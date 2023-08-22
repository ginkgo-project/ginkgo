/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2023, the Ginkgo authors
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


namespace sycl {
namespace fft {

GKO_DECLARE_ALL_AS_TEMPLATES;

}  // namespace fft
}  // namespace sycl


#undef GKO_DECLARE_ALL_AS_TEMPLATES


}  // namespace kernels
}  // namespace gko


#endif  // GKO_CORE_MATRIX_FFT_KERNELS_HPP_
