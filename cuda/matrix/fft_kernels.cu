// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/matrix/fft_kernels.hpp"


#include <array>


#include <cufft.h>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/dense.hpp>


namespace gko {
namespace kernels {
namespace cuda {
/**
 * @brief The FFT matrix format namespace.
 * @ref Fft
 * @ingroup fft
 */
namespace fft {


template <typename InValueType, typename OutValueType>
struct cufft_type_impl {};

template <>
struct cufft_type_impl<std::complex<float>, std::complex<float>> {
    constexpr static auto value = CUFFT_C2C;
};

template <>
struct cufft_type_impl<std::complex<double>, std::complex<double>> {
    constexpr static auto value = CUFFT_Z2Z;
};


class cufft_handle {
    struct cufft_deleter {
        void operator()(cufftHandle* ptr)
        {
            auto data = *ptr;
            delete ptr;
            cufftDestroy(data);
        }
    };

public:
    operator cufftHandle() const { return *handle_; }

    cufft_handle(cudaStream_t stream) : handle_{new cufftHandle{}}
    {
        GKO_ASSERT_NO_CUFFT_ERRORS(cufftCreate(handle_.get()));
        GKO_ASSERT_NO_CUFFT_ERRORS(cufftSetStream(*handle_, stream));
    }

    template <int d, typename InValueType, typename OutValueType>
    void setup(std::array<size_type, d> fft_size, size_type in_batch_stride,
               size_type out_batch_stride, size_type batch_count,
               array<char>& work_area)
    {
        static_assert(d == 1 || d == 2 || d == 3,
                      "Only 1D, 2D or 3D FFT supported");
        std::array<long long, d> cast_fft_size;
        for (int i = 0; i < d; i++) {
            cast_fft_size[i] = static_cast<long long>(fft_size[i]);
        }
        size_type work_size{};
        GKO_ASSERT_NO_CUFFT_ERRORS(cufftSetAutoAllocation(*handle_, false));
        GKO_ASSERT_NO_CUFFT_ERRORS(cufftMakePlanMany64(
            *handle_, d, cast_fft_size.data(), cast_fft_size.data(),
            static_cast<int64>(in_batch_stride), 1, cast_fft_size.data(),
            static_cast<int64>(out_batch_stride), 1,
            cufft_type_impl<InValueType, OutValueType>::value,
            static_cast<int64>(batch_count), &work_size));
        work_area.resize_and_reset(work_size);
        GKO_ASSERT_NO_CUFFT_ERRORS(
            cufftSetWorkArea(*handle_, work_area.get_data()));
    }


    void execute(const std::complex<float>* in, std::complex<float>* out,
                 bool inverse)
    {
        cufftExecC2C(*handle_,
                     const_cast<cufftComplex*>(
                         reinterpret_cast<const cufftComplex*>(in)),
                     reinterpret_cast<cufftComplex*>(out),
                     inverse ? CUFFT_INVERSE : CUFFT_FORWARD);
    }

    void execute(const std::complex<double>* in, std::complex<double>* out,
                 bool inverse)
    {
        cufftExecZ2Z(*handle_,
                     const_cast<cufftDoubleComplex*>(
                         reinterpret_cast<const cufftDoubleComplex*>(in)),
                     reinterpret_cast<cufftDoubleComplex*>(out),
                     inverse ? CUFFT_INVERSE : CUFFT_FORWARD);
    }

private:
    std::unique_ptr<cufftHandle, cufft_deleter> handle_;
};


template <typename ValueType>
void fft(std::shared_ptr<const DefaultExecutor> exec,
         const matrix::Dense<std::complex<ValueType>>* b,
         matrix::Dense<std::complex<ValueType>>* x, bool inverse,
         array<char>& buffer)
{
    cufft_handle handle{exec->get_stream()};
    handle.template setup<1, std::complex<ValueType>, std::complex<ValueType>>(
        {b->get_size()[0]}, b->get_stride(), x->get_stride(), b->get_size()[1],
        buffer);
    handle.execute(b->get_const_values(), x->get_values(), inverse);
}

GKO_INSTANTIATE_FOR_EACH_NON_COMPLEX_VALUE_TYPE(GKO_DECLARE_FFT_KERNEL);


template <typename ValueType>
void fft2(std::shared_ptr<const DefaultExecutor> exec,
          const matrix::Dense<std::complex<ValueType>>* b,
          matrix::Dense<std::complex<ValueType>>* x, size_type size1,
          size_type size2, bool inverse, array<char>& buffer)
{
    cufft_handle handle{exec->get_stream()};
    handle.template setup<2, std::complex<ValueType>, std::complex<ValueType>>(
        {size1, size2}, b->get_stride(), x->get_stride(), b->get_size()[1],
        buffer);
    handle.execute(b->get_const_values(), x->get_values(), inverse);
}

GKO_INSTANTIATE_FOR_EACH_NON_COMPLEX_VALUE_TYPE(GKO_DECLARE_FFT2_KERNEL);


template <typename ValueType>
void fft3(std::shared_ptr<const DefaultExecutor> exec,
          const matrix::Dense<std::complex<ValueType>>* b,
          matrix::Dense<std::complex<ValueType>>* x, size_type size1,
          size_type size2, size_type size3, bool inverse, array<char>& buffer)
{
    cufft_handle handle{exec->get_stream()};
    handle.template setup<3, std::complex<ValueType>, std::complex<ValueType>>(
        {size1, size2, size3}, b->get_stride(), x->get_stride(),
        b->get_size()[1], buffer);
    handle.execute(b->get_const_values(), x->get_values(), inverse);
}

GKO_INSTANTIATE_FOR_EACH_NON_COMPLEX_VALUE_TYPE(GKO_DECLARE_FFT3_KERNEL);


}  // namespace fft
}  // namespace cuda
}  // namespace kernels
}  // namespace gko
