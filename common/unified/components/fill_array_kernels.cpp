// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/components/fill_array_kernels.hpp"

#include <type_traits>

#include "common/unified/base/kernel_launch.hpp"


namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {
namespace components {


template <typename ValueType>
void fill_array(std::shared_ptr<const DefaultExecutor> exec, ValueType* array,
                size_type n, ValueType val)
{
    run_kernel(
        exec,
        [] GKO_KERNEL(auto idx, auto array, auto val) { array[idx] = val; }, n,
        array, val);
}

GKO_INSTANTIATE_FOR_EACH_TEMPLATE_TYPE(GKO_DECLARE_FILL_ARRAY_KERNEL);
template GKO_DECLARE_FILL_ARRAY_KERNEL(bool);
template GKO_DECLARE_FILL_ARRAY_KERNEL(uint16);
template GKO_DECLARE_FILL_ARRAY_KERNEL(uint32);
#ifndef GKO_SIZE_T_IS_UINT64_T
template GKO_DECLARE_FILL_ARRAY_KERNEL(uint64);
#endif


template <typename ValueType>
void fill_seq_array(std::shared_ptr<const DefaultExecutor> exec,
                    ValueType* array, size_type n)
{
    run_kernel(
        exec,
        [] GKO_KERNEL(auto idx, auto array) {
#if defined(GKO_COMPILING_HIP) && HIP_VERSION < 60200000
            if constexpr (std::is_same_v<remove_complex<ValueType>, bfloat16>) {
                // hip_bfloat16 does not have implicit conversion, so the
                // thrust<hip_bfloat16> can not be from float. Also,
                // hip_bfloat16 does not have operator=(float) before 5.4. Thus,
                // we cast twice via float before 6.2
                array[idx] = static_cast<hip_bfloat16>(static_cast<float>(idx));

            } else
#endif
                if constexpr (std::is_same_v<remove_complex<ValueType>,
                                             float16> ||
                              std::is_same_v<remove_complex<ValueType>,
                                             bfloat16>) {
                // __half can not be from int64_t
                // __hip_bfloat16 can not be from long long
                array[idx] = static_cast<float>(idx);
            } else {
                array[idx] = idx;
            }
        },
        n, array);
}

GKO_INSTANTIATE_FOR_EACH_TEMPLATE_TYPE(GKO_DECLARE_FILL_SEQ_ARRAY_KERNEL);


}  // namespace components
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko
