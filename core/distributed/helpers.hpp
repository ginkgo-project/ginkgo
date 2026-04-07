// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_DISTRIBUTED_HELPERS_HPP_
#define GKO_CORE_DISTRIBUTED_HELPERS_HPP_


#include <memory>

#include <ginkgo/config.hpp>
#include <ginkgo/core/distributed/matrix.hpp>
#include <ginkgo/core/distributed/vector.hpp>
#include <ginkgo/core/matrix/dense.hpp>

#include "core/base/dispatch_helper.hpp"


namespace gko {
namespace experimental {
namespace distributed {
namespace detail {
#if GINKGO_BUILD_MPI


template <typename ValueType>
matrix::Dense<ValueType>* get_local_mutable(Vector<ValueType>* mtx)
{
    return const_cast<matrix::Dense<ValueType>*>(mtx->get_local_vector());
}


template <typename ValueType>
const matrix::Dense<ValueType>* get_local(const Vector<ValueType>* mtx)
{
    return mtx->get_local_vector();
}


#endif


template <typename Arg>
bool is_distributed(Arg* linop)
{
#if GINKGO_BUILD_MPI
    return dynamic_cast<const DistributedBase*>(linop);
#else
    return false;
#endif
}


template <typename Arg, typename... Rest>
bool is_distributed(Arg* linop, Rest*... rest)
{
#if GINKGO_BUILD_MPI
    bool is_distributed_value = dynamic_cast<const DistributedBase*>(linop);
    GKO_ASSERT(is_distributed_value == is_distributed(rest...));
    return is_distributed_value;
#else
    return false;
#endif
}


#if GINKGO_BUILD_MPI


/**
 * Specialization of run for distributed matrices.
 */
template <typename T, typename F, typename... Args>
auto run_matrix(T* linop, F&& f, Args&&... args)
{
    using namespace gko::detail;
    return run<
        with_same_constness_t<Matrix<double, int32, int32>, T>,
        with_same_constness_t<Matrix<double, int32, int64>, T>,
        with_same_constness_t<Matrix<double, int64, int64>, T>,
        with_same_constness_t<Matrix<float, int32, int32>, T>,
        with_same_constness_t<Matrix<float, int32, int64>, T>,
        with_same_constness_t<Matrix<float, int64, int64>, T>,
#if GINKGO_ENABLE_HALF
        with_same_constness_t<Matrix<float16, int32, int32>, T>,
        with_same_constness_t<Matrix<float16, int32, int64>, T>,
        with_same_constness_t<Matrix<float16, int64, int64>, T>,
        with_same_constness_t<Matrix<std::complex<float16>, int32, int32>, T>,
        with_same_constness_t<Matrix<std::complex<float16>, int32, int64>, T>,
        with_same_constness_t<Matrix<std::complex<float16>, int64, int64>, T>,
#endif
#if GINKGO_ENABLE_BFLOAT16
        with_same_constness_t<Matrix<bfloat16, int32, int32>, T>,
        with_same_constness_t<Matrix<bfloat16, int32, int64>, T>,
        with_same_constness_t<Matrix<bfloat16, int64, int64>, T>,
        with_same_constness_t<Matrix<std::complex<bfloat16>, int32, int32>, T>,
        with_same_constness_t<Matrix<std::complex<bfloat16>, int32, int64>, T>,
        with_same_constness_t<Matrix<std::complex<bfloat16>, int64, int64>, T>,
#endif
        with_same_constness_t<Matrix<std::complex<double>, int32, int32>, T>,
        with_same_constness_t<Matrix<std::complex<double>, int32, int64>, T>,
        with_same_constness_t<Matrix<std::complex<double>, int64, int64>, T>,
        with_same_constness_t<Matrix<std::complex<float>, int32, int32>, T>,
        with_same_constness_t<Matrix<std::complex<float>, int32, int64>, T>,
        with_same_constness_t<Matrix<std::complex<float>, int64, int64>, T>>(
        linop, std::forward<F>(f), std::forward<Args>(args)...);
}


#endif


inline const LinOp* get_local(const LinOp* mtx)
{
#if GINKGO_BUILD_MPI
    if (is_distributed(mtx)) {
        return run_matrix(mtx, [](auto concrete) {
            return concrete->get_diag_matrix().get();
        });
    }
#endif
    {
        return mtx;
    }
}


}  // namespace detail
}  // namespace distributed
}  // namespace experimental
}  // namespace gko


#endif  // GKO_CORE_DISTRIBUTED_HELPERS_HPP_
