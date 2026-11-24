// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_SOLVER_SOLVER_BASE_HPP_
#define GKO_CORE_SOLVER_SOLVER_BASE_HPP_


#include <core/base/dispatch_helper.hpp>

#include <ginkgo/core/distributed/vector.hpp>
#include <ginkgo/core/matrix/multivector.hpp>
#include <ginkgo/core/solver/solver_base.hpp>

namespace gko {
namespace solver {


/**
 * Modify the input vector x by the guess
 *
 * @param b  the right hand side vectors
 * @param x  the input vectors
 * @param guess  the input guess
 */
inline void prepare_initial_guess(const AbstractMultiVector* b,
                                  AbstractMultiVector* x,
                                  initial_guess_mode guess)
{
    if (guess == initial_guess_mode::zero) {
        x->fill(0.0);
    } else if (guess == initial_guess_mode::rhs) {
        x->copy_from(b);
    }
}


}  // namespace solver


template <typename SizeFn>
std::function<std::unique_ptr<AbstractMultiVector>()> stacked_creator(
    SizeFn&& size_fn, const AbstractMultiVector* vec, size_type stacks)
{
    auto stacked_size = size_fn(vec, stacks);

    return run<
#if GINKGO_BUILD_MPI
#if GINKGO_ENABLE_HALF
        experimental::distributed::Vector<half>,
        experimental::distributed::Vector<std::complex<half>>,
#endif
#if GINKGO_ENABLE_BFLOAT16
        experimental::distributed::Vector<bfloat16>,
        experimental::distributed::Vector<std::complex<bfloat16>>,
#endif
        experimental::distributed::Vector<float>,
        experimental::distributed::Vector<double>,
        experimental::distributed::Vector<std::complex<float>>,
        experimental::distributed::Vector<std::complex<double>>,
#endif
#if GINKGO_ENABLE_HALF
        matrix::MultiVector<half>, matrix::MultiVector<std::complex<half>>,
#endif
#if GINKGO_ENABLE_BFLOAT16
        matrix::MultiVector<bfloat16>,
        matrix::MultiVector<std::complex<bfloat16>>,
#endif
        matrix::MultiVector<float>, matrix::MultiVector<double>,
        matrix::MultiVector<std::complex<float>>,
        matrix::MultiVector<std::complex<double>>>(
        vec,
        [stacks, stacked_size, &size_fn](auto concrete_vec)
            -> std::function<std::unique_ptr<AbstractMultiVector>()> {
            using Vector = std::decay_t<decltype(*concrete_vec)>;
            auto exec = concrete_vec->get_executor();
#if GINKGO_BUILD_MPI
            if constexpr (std::is_base_of_v<
                              experimental::distributed::DistributedBase,
                              Vector>) {
                auto comm = concrete_vec->get_communicator();
                auto local_stacked_size =
                    size_fn(concrete_vec->get_local_vector(), stacks);
                return [exec, comm, stacked_size, local_stacked_size]()
                           -> std::unique_ptr<AbstractMultiVector> {
                    return Vector::create(exec, comm, stacked_size,
                                          local_stacked_size);
                };
            } else
#endif
            {
                return
                    [exec,
                     stacked_size]() -> std::unique_ptr<AbstractMultiVector> {
                        return Vector::create(exec, stacked_size);
                    };
            }
        });
}


inline dim<2> detail::vertically_stacked_size(const AbstractMultiVector* vec,
                                              size_type stacks)
{
    auto size = vec->get_size();
    return {size[0] * stacks, size[1]};
}


inline std::function<std::unique_ptr<AbstractMultiVector>()>
detail::vertically_stacked_creator(const AbstractMultiVector* vec,
                                   size_type stacks)
{
    return stacked_creator(vertically_stacked_size, vec, stacks);
}


inline dim<2> detail::horizontally_stacked_size(const AbstractMultiVector* vec,
                                                size_type stacks)
{
    auto size = vec->get_size();
    return {size[0], size[1] * stacks};
}


inline std::function<std::unique_ptr<AbstractMultiVector>()>
detail::horizontally_stacked_creator(const AbstractMultiVector* vec,
                                     size_type stacks)
{
    return stacked_creator(horizontally_stacked_size, vec, stacks);
}


}  // namespace gko


#endif  // GKO_CORE_SOLVER_SOLVER_BASE_HPP_
