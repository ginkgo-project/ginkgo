// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/components/precision_conversion_kernels.hpp"

#include "common/unified/base/kernel_launch.hpp"


namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {
namespace components {


template <typename SourceType, typename TargetType>
void convert_precision(std::shared_ptr<const DefaultExecutor> exec,
                       size_type size, const SourceType* in, TargetType* out)
{
    run_kernel(
        exec,
        [] GKO_KERNEL(auto idx, auto in, auto out) {
#if defined(GKO_COMPILING_DPCPP) ||                            \
    (defined(GKO_COMPILING_HIP) && HIP_VERSION >= 60200000) || \
    (defined(CUDA_VERSION) && CUDA_VERSION < 12020)
            // need to use the type from lambda not template type directly
            // because cuda still checks the function availability in unused
            // branch like no constexpr.
            using source_type = std::remove_cv_t<
                std::remove_pointer_t<std::decay_t<decltype(in)>>>;
            using target_type = std::remove_cv_t<
                std::remove_pointer_t<std::decay_t<decltype(out)>>>;
            if constexpr (sizeof(remove_complex<source_type>) ==
                              sizeof(int16) &&
                          sizeof(remove_complex<target_type>) ==
                              sizeof(int16)) {
                if constexpr (is_complex<source_type>()) {
                    out[idx] = static_cast<device_type<target_type>>(
                        static_cast<device_type<std::complex<float>>>(in[idx]));
                } else {
                    out[idx] = static_cast<device_type<target_type>>(
                        static_cast<device_type<float>>(in[idx]));
                }
            } else
#endif
            {
                out[idx] = static_cast<device_type<TargetType>>(in[idx]);
            }
        },
        size, in, out);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_CONVERSION(GKO_DECLARE_CONVERT_PRECISION_KERNEL);


}  // namespace components
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko
