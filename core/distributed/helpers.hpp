// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
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
namespace detail {


template <typename ValueType>
std::unique_ptr<matrix::Dense<ValueType>> create_with_config_of(
    const matrix::Dense<ValueType>* mtx)
{
    return matrix::Dense<ValueType>::create(mtx->get_executor(),
                                            mtx->get_size(), mtx->get_stride());
}


template <typename ValueType>
const matrix::Dense<ValueType>* get_local(const matrix::Dense<ValueType>* mtx)
{
    return mtx;
}


template <typename ValueType>
matrix::Dense<ValueType>* get_local(matrix::Dense<ValueType>* mtx)
{
    return mtx;
}


#if GINKGO_BUILD_MPI


template <typename ValueType>
std::unique_ptr<experimental::distributed::Vector<ValueType>>
create_with_config_of(const experimental::distributed::Vector<ValueType>* mtx)
{
    return experimental::distributed::Vector<ValueType>::create(
        mtx->get_executor(), mtx->get_communicator(), mtx->get_size(),
        mtx->get_local_vector()->get_size(),
        mtx->get_local_vector()->get_stride());
}


template <typename ValueType>
matrix::Dense<ValueType>* get_local(
    experimental::distributed::Vector<ValueType>* mtx)
{
    return const_cast<matrix::Dense<ValueType>*>(mtx->get_local_vector());
}


template <typename ValueType>
const matrix::Dense<ValueType>* get_local(
    const experimental::distributed::Vector<ValueType>* mtx)
{
    return mtx->get_local_vector();
}


#endif


template <typename Arg>
bool is_distributed(Arg* linop)
{
#if GINKGO_BUILD_MPI
    return dynamic_cast<const experimental::distributed::DistributedBase*>(
        linop);
#else
    return false;
#endif
}


template <typename Arg, typename... Rest>
bool is_distributed(Arg* linop, Rest*... rest)
{
#if GINKGO_BUILD_MPI
    bool is_distributed_value =
        dynamic_cast<const experimental::distributed::DistributedBase*>(linop);
    GKO_ASSERT(is_distributed_value == is_distributed(rest...));
    return is_distributed_value;
#else
    return false;
#endif
}


/**
 * Cast an input linop to the correct underlying vector type (dense/distributed)
 * and passes it to the given function.
 *
 * @tparam ValueType  The value type of the underlying dense or distributed
 * vector.
 * @tparam T  The linop type, either LinOp, or const LinOp.
 * @tparam F  The function type.
 * @tparam Args  The types for the additional arguments of f.
 *
 * @param linop  The linop to be casted into either a dense or distributed
 *               vector.
 * @param f  The function that is to be called with the correctly casted linop.
 * @param args  The additional arguments of f.
 */
template <typename ValueType, typename T, typename F, typename... Args>
void vector_dispatch(T* linop, F&& f, Args&&... args)
{
#if GINKGO_BUILD_MPI
    if (is_distributed(linop)) {
        using type = std::conditional_t<
            std::is_const<T>::value,
            const experimental::distributed::Vector<ValueType>,
            experimental::distributed::Vector<ValueType>>;
        f(dynamic_cast<type*>(linop), std::forward<Args>(args)...);
    } else
#endif
    {
        using type = std::conditional_t<std::is_const<T>::value,
                                        const matrix::Dense<ValueType>,
                                        matrix::Dense<ValueType>>;
        if (auto concrete_linop = dynamic_cast<type*>(linop)) {
            f(concrete_linop, std::forward<Args>(args)...);
        } else {
            GKO_NOT_SUPPORTED(linop);
        }
    }
}


#if GINKGO_BUILD_MPI


/**
 * Specialization of run for distributed matrices.
 */
template <typename T, typename F, typename... Args>
auto run_matrix(T* linop, F&& f, Args&&... args)
{
    using namespace gko::experimental::distributed;
    return run<
        with_same_constness_t<Matrix<double, int32, int32>, T>,
        with_same_constness_t<Matrix<double, int32, int64>, T>,
        with_same_constness_t<Matrix<double, int64, int64>, T>,
        with_same_constness_t<Matrix<float, int32, int32>, T>,
        with_same_constness_t<Matrix<float, int32, int64>, T>,
        with_same_constness_t<Matrix<float, int64, int64>, T>,
#if GINKGO_ENABLE_HALF || GINKGO_ENABLE_BFLOAT16
        with_same_constness_t<Matrix<float16, int32, int32>, T>,
        with_same_constness_t<Matrix<float16, int32, int64>, T>,
        with_same_constness_t<Matrix<float16, int64, int64>, T>,
        with_same_constness_t<Matrix<std::complex<float16>, int32, int32>, T>,
        with_same_constness_t<Matrix<std::complex<float16>, int32, int64>, T>,
        with_same_constness_t<Matrix<std::complex<float16>, int64, int64>, T>,
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


/**
 * Helper to extract a submatrix.
 *
 * @note  global_size is unused, since it can be inferred from rows and cols.
 */
template <typename ValueType>
std::unique_ptr<matrix::Dense<ValueType>> create_submatrix_helper(
    matrix::Dense<ValueType>* mtx, dim<2> global_size, span rows, span cols)
{
    return mtx->create_submatrix(rows, cols);
}


#if GINKGO_BUILD_MPI


/**
 * Helper to extract a submatrix.
 *
 * @param global_size  the global_size of the submatrix
 * @param rows  the rows of the submatrix in local indices
 * @param cols  the columns of the submatrix in local indices
 */
template <typename ValueType>
std::unique_ptr<experimental::distributed::Vector<ValueType>>
create_submatrix_helper(experimental::distributed::Vector<ValueType>* mtx,
                        dim<2> global_size, span rows, span cols)
{
    const auto exec = mtx->get_executor();
    auto local_view = matrix::Dense<ValueType>::create(
        exec, mtx->get_local_vector()->get_size(),
        make_array_view(exec,
                        mtx->get_local_vector()->get_num_stored_elements(),
                        mtx->get_local_values()),
        mtx->get_local_vector()->get_stride());
    return experimental::distributed::Vector<ValueType>::create(
        exec, mtx->get_communicator(), global_size,
        local_view->create_submatrix(rows, cols));
}


#endif


inline const LinOp* get_local(const LinOp* mtx)
{
#if GINKGO_BUILD_MPI
    if (is_distributed(mtx)) {
        return run_matrix(mtx, [](auto concrete) {
            return concrete->get_local_matrix().get();
        });
    }
#endif
    {
        return mtx;
    }
}


}  // namespace detail
}  // namespace gko


#endif  // GKO_CORE_DISTRIBUTED_HELPERS_HPP_
