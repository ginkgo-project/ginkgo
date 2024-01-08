// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_COMMON_UNIFIED_BASE_KERNEL_LAUNCH_SOLVER_HPP_
#define GKO_COMMON_UNIFIED_BASE_KERNEL_LAUNCH_SOLVER_HPP_


#include "common/unified/base/kernel_launch.hpp"


namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {


/**
 * @internal
 * Wrapper class used by default_stride(matrix::Dense<ValueType>*) to wrap a
 * dense matrix using the default stride.
 */
template <typename ValueType>
struct default_stride_dense_wrapper {
    ValueType* data;
};


/**
 * @internal
 * Helper that creates a device representation of the input object based on the
 * default stride that was passed to run_kernel_solver.
 * @see default_stride_dense_wrapper
 * @see default_stride(matrix::Dense<ValueType>*)
 */
template <typename T>
struct device_unpack_solver_impl {
    using type = T;
    static GKO_INLINE GKO_ATTRIBUTES type unpack(T param, int64)
    {
        return param;
    }
};

template <typename ValueType>
struct device_unpack_solver_impl<default_stride_dense_wrapper<ValueType>> {
    using type = matrix_accessor<ValueType>;
    static GKO_INLINE GKO_ATTRIBUTES type
    unpack(default_stride_dense_wrapper<ValueType> param, int64 default_stride)
    {
        return {param.data, default_stride};
    }
};


/**
 * @internal
 * Wraps the given matrix in a wrapper signifying that it has the default stride
 * that was provided to run_kernel_solver. This avoids having individual stride
 * parameters for all dense matrix parameters.
 */
template <typename ValueType>
default_stride_dense_wrapper<device_type<ValueType>> default_stride(
    matrix::Dense<ValueType>* mtx)
{
    return {as_device_type(mtx->get_values())};
}

/**
 * @internal
 * @copydoc default_stride(matrix::Dense<ValueType>*)
 */
template <typename ValueType>
default_stride_dense_wrapper<const device_type<ValueType>> default_stride(
    const matrix::Dense<ValueType>* mtx)
{
    return {as_device_type(mtx->get_const_values())};
}


/**
 * @internal
 * Wraps the given matrix in a wrapper signifying that it is a row vector, i.e.
 * we don't need to pass a stride parameter, but can access it directly as a
 * pointer.
 */
template <typename ValueType>
device_type<ValueType>* row_vector(matrix::Dense<ValueType>* mtx)
{
    GKO_ASSERT(mtx->get_size()[0] == 1);
    return as_device_type(mtx->get_values());
}

/**
 * @internal
 * @copydoc row_vector(matrix::Dense<ValueType>*)
 */
template <typename ValueType>
const device_type<ValueType>* row_vector(const matrix::Dense<ValueType>* mtx)
{
    GKO_ASSERT(mtx->get_size()[0] == 1);
    return as_device_type(mtx->get_const_values());
}


}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko


#if defined(GKO_COMPILING_CUDA)
#include "cuda/base/kernel_launch_solver.cuh"
#elif defined(GKO_COMPILING_HIP)
#include "hip/base/kernel_launch_solver.hip.hpp"
#elif defined(GKO_COMPILING_DPCPP)
#include "dpcpp/base/kernel_launch_solver.dp.hpp"
#elif defined(GKO_COMPILING_OMP)
#include "omp/base/kernel_launch_solver.hpp"
#endif


#endif  // GKO_COMMON_UNIFIED_BASE_KERNEL_LAUNCH_SOLVER_HPP_
