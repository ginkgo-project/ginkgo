// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/matrix/fft_kernels.hpp"


#include <array>


#include <hip/hip_runtime.h>
#if HIP_VERSION >= 50200000
#include <hipfft/hipfft.h>
#else
#include <hipfft.h>
#endif


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/dense.hpp>


namespace gko {


std::string HipfftError::get_error(int64 error_code)
{
#define GKO_REGISTER_HIPFFT_ERROR(error_name) \
    if (error_code == int64(error_name)) {    \
        return #error_name;                   \
    }
    GKO_REGISTER_HIPFFT_ERROR(HIPFFT_SUCCESS)
    GKO_REGISTER_HIPFFT_ERROR(HIPFFT_INVALID_PLAN)
    GKO_REGISTER_HIPFFT_ERROR(HIPFFT_ALLOC_FAILED)
    GKO_REGISTER_HIPFFT_ERROR(HIPFFT_INVALID_TYPE)
    GKO_REGISTER_HIPFFT_ERROR(HIPFFT_INVALID_VALUE)
    GKO_REGISTER_HIPFFT_ERROR(HIPFFT_INTERNAL_ERROR)
    GKO_REGISTER_HIPFFT_ERROR(HIPFFT_EXEC_FAILED)
    GKO_REGISTER_HIPFFT_ERROR(HIPFFT_SETUP_FAILED)
    GKO_REGISTER_HIPFFT_ERROR(HIPFFT_INVALID_SIZE)
    GKO_REGISTER_HIPFFT_ERROR(HIPFFT_UNALIGNED_DATA)
    GKO_REGISTER_HIPFFT_ERROR(HIPFFT_INCOMPLETE_PARAMETER_LIST)
    GKO_REGISTER_HIPFFT_ERROR(HIPFFT_INVALID_DEVICE)
    GKO_REGISTER_HIPFFT_ERROR(HIPFFT_PARSE_ERROR)
    GKO_REGISTER_HIPFFT_ERROR(HIPFFT_NO_WORKSPACE)
    GKO_REGISTER_HIPFFT_ERROR(HIPFFT_NOT_IMPLEMENTED)
    GKO_REGISTER_HIPFFT_ERROR(HIPFFT_NOT_SUPPORTED)
    return "Unknown error";

#undef GKO_REGISTER_HIPFFT_ERROR
}


namespace kernels {
namespace hip {
/**
 * @brief The FFT matrix format namespace.
 * @ref Fft
 * @ingroup fft
 */
namespace fft {


template <typename InValueType, typename OutValueType>
struct hipfft_type_impl {};


template <>
struct hipfft_type_impl<std::complex<float>, std::complex<float>> {
    constexpr static auto value = HIPFFT_C2C;
};

template <>
struct hipfft_type_impl<std::complex<double>, std::complex<double>> {
    constexpr static auto value = HIPFFT_Z2Z;
};


class hipfft_handle {
    struct hipfft_deleter {
        void operator()(hipfftHandle* ptr)
        {
            auto data = *ptr;
            delete ptr;
            hipfftDestroy(data);
        }
    };

public:
    operator hipfftHandle() const { return *handle_; }

    hipfft_handle(hipStream_t stream) : handle_{new hipfftHandle{}}
    {
        GKO_ASSERT_NO_HIPFFT_ERRORS(hipfftCreate(handle_.get()));
        GKO_ASSERT_NO_HIPFFT_ERRORS(hipfftSetStream(*handle_, stream));
    }

    template <int d, typename InValueType, typename OutValueType>
    void setup(std::array<size_type, d> fft_size, size_type in_batch_stride,
               size_type out_batch_stride, size_type batch_count,
               array<char>& work_area)
    {
        static_assert(d == 1 || d == 2 || d == 3,
                      "Only 1D, 2D or 3D FFT supported");
        std::array<int, d> cast_fft_size;
        for (int i = 0; i < d; i++) {
            // hipFFT only has 32bit index support
            if (fft_size[i] > std::numeric_limits<int>::max()) {
                GKO_NOT_IMPLEMENTED;
            }
            cast_fft_size[i] = static_cast<int>(fft_size[i]);
        }
        size_type work_size{};
        GKO_ASSERT_NO_HIPFFT_ERRORS(hipfftSetAutoAllocation(*handle_, false));
        GKO_ASSERT_NO_HIPFFT_ERRORS(hipfftMakePlanMany(
            *handle_, d, cast_fft_size.data(), cast_fft_size.data(),
            static_cast<int64>(in_batch_stride), 1, cast_fft_size.data(),
            static_cast<int64>(out_batch_stride), 1,
            hipfft_type_impl<InValueType, OutValueType>::value,
            static_cast<int64>(batch_count), &work_size));
        work_area.resize_and_reset(work_size);
        GKO_ASSERT_NO_HIPFFT_ERRORS(
            hipfftSetWorkArea(*handle_, work_area.get_data()));
    }


    void execute(const std::complex<float>* in, std::complex<float>* out,
                 bool inverse)
    {
        hipfftExecC2C(*handle_,
                      const_cast<hipfftComplex*>(
                          reinterpret_cast<const hipfftComplex*>(in)),
                      reinterpret_cast<hipfftComplex*>(out),
                      inverse ? HIPFFT_BACKWARD : HIPFFT_FORWARD);
    }

    void execute(const std::complex<double>* in, std::complex<double>* out,
                 bool inverse)
    {
        hipfftExecZ2Z(*handle_,
                      const_cast<hipfftDoubleComplex*>(
                          reinterpret_cast<const hipfftDoubleComplex*>(in)),
                      reinterpret_cast<hipfftDoubleComplex*>(out),
                      inverse ? HIPFFT_BACKWARD : HIPFFT_FORWARD);
    }

private:
    std::unique_ptr<hipfftHandle, hipfft_deleter> handle_;
};


template <typename ValueType>
void fft(std::shared_ptr<const DefaultExecutor> exec,
         const matrix::Dense<std::complex<ValueType>>* b,
         matrix::Dense<std::complex<ValueType>>* x, bool inverse,
         array<char>& buffer)
{
    hipfft_handle handle{exec->get_stream()};
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
    hipfft_handle handle{exec->get_stream()};
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
    hipfft_handle handle{exec->get_stream()};
    handle.template setup<3, std::complex<ValueType>, std::complex<ValueType>>(
        {size1, size2, size3}, b->get_stride(), x->get_stride(),
        b->get_size()[1], buffer);
    handle.execute(b->get_const_values(), x->get_values(), inverse);
}

GKO_INSTANTIATE_FOR_EACH_NON_COMPLEX_VALUE_TYPE(GKO_DECLARE_FFT3_KERNEL);


}  // namespace fft
}  // namespace hip
}  // namespace kernels
}  // namespace gko
