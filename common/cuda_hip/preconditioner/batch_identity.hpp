// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_COMMON_CUDA_HIP_PRECONDITIONER_BATCH_IDENTITY_HPP_
#define GKO_COMMON_CUDA_HIP_PRECONDITIONER_BATCH_IDENTITY_HPP_


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>

#include "common/cuda_hip/base/batch_struct.hpp"
#include "common/cuda_hip/base/config.hpp"
#include "common/cuda_hip/matrix/batch_struct.hpp"


namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {
namespace batch_preconditioner {


/**
 * @see reference/preconditioner/batch_identity.hpp
 */
template <typename ValueType>
class Identity final {
public:
    using value_type = ValueType;

    static constexpr int work_size = 0;

    __host__ __device__ static constexpr int dynamic_work_size(int, int)
    {
        return 0;
    }

    template <typename batch_item_type>
    __device__ __forceinline__ void generate(size_type, const batch_item_type&,
                                             ValueType*)
    {}

    __device__ __forceinline__ void apply(const int num_rows,
                                          const ValueType* const r,
                                          ValueType* const z) const
    {
        for (int li = threadIdx.x; li < num_rows; li += blockDim.x) {
            z[li] = r[li];
        }
    }
};


}  // namespace batch_preconditioner
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko


#endif
