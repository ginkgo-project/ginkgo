/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2021, the Ginkgo authors
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

#include <ginkgo/core/solver/cb_gmres.hpp>


#include <type_traits>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/utils_helper.hpp>
#include <ginkgo/core/log/logger.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/base/extended_float.hpp"
#include "core/solver/cb_gmres_accessor.hpp"
#include "core/solver/cb_gmres_kernels.hpp"


namespace gko {
namespace solver {


namespace cb_gmres {


GKO_REGISTER_OPERATION(initialize_1, cb_gmres::initialize_1);
GKO_REGISTER_OPERATION(initialize_2, cb_gmres::initialize_2);
GKO_REGISTER_OPERATION(step_1, cb_gmres::step_1);
GKO_REGISTER_OPERATION(step_2, cb_gmres::step_2);


}  // namespace cb_gmres


template <typename T>
struct to_integer_impl {
    using type = T;
};

template <>
struct to_integer_impl<double> {
    using type = int64;
};

template <>
struct to_integer_impl<float> {
    using type = int32;
};

template <>
struct to_integer_impl<half> {
    using type = int16;
};

template <typename T>
using to_integer = typename to_integer_impl<T>::type;


template <typename T, typename Skip>
using reduce_precision_skip =
    typename std::conditional_t<std::is_same<reduce_precision<T>, Skip>::value,
                                T, reduce_precision<T>>;


namespace detail {


template <typename T, typename Skip, int count>
struct reduce_precision_skip_count_impl {
    static_assert(count > 0,
                  "The count variable must be larger or equal to zero.");
    using type = typename reduce_precision_skip_count_impl<
        reduce_precision_skip<T, Skip>, Skip, count - 1>::type;
};

template <typename T, typename Skip>
struct reduce_precision_skip_count_impl<T, Skip, 0> {
    using type = T;
};


}  // namespace detail


template <typename T, typename Skip, int count>
using reduce_precision_skip_count =
    typename detail::reduce_precision_skip_count_impl<T, Skip, count>::type;


template <typename T, int count>
using reduce_precision_count =
    typename detail::reduce_precision_skip_count_impl<T, void, count>::type;


template <typename ValueType>
struct helper {
    template <typename Callable>
    static void call(Callable callable,
                     gko::solver::cb_gmres::storage_precision st)
    {
        switch (st) {
        case cb_gmres::storage_precision::reduce1:
            callable(reduce_precision_count<ValueType, 1>{});
            break;
        case cb_gmres::storage_precision::reduce2:
            callable(reduce_precision_count<ValueType, 2>{});
            break;
        case cb_gmres::storage_precision::integer:
            callable(to_integer<ValueType>{});
            break;
        case cb_gmres::storage_precision::ireduce1:
            callable(to_integer<reduce_precision_count<ValueType, 1>>{});
            break;
        case cb_gmres::storage_precision::ireduce2:
            callable(to_integer<reduce_precision_count<ValueType, 2>>{});
            break;
        default:
            callable(ValueType{});
        }
    }
};


// helper for complex numbers
template <typename T>
struct helper<std::complex<T>> {
    using ValueType = std::complex<T>;
    using skip_type = std::complex<half>;

    template <typename Callable>
    static void call(Callable callable,
                     gko::solver::cb_gmres::storage_precision st)
    {
        switch (st) {
        case cb_gmres::storage_precision::reduce1:
            callable(reduce_precision_skip_count<ValueType, skip_type, 1>{});
            break;
        case cb_gmres::storage_precision::reduce2:
            callable(reduce_precision_skip_count<ValueType, skip_type, 2>{});
            break;
        case cb_gmres::storage_precision::integer:
        case cb_gmres::storage_precision::ireduce1:
        case cb_gmres::storage_precision::ireduce2:
            GKO_NOT_SUPPORTED(st);
            break;
        default:
            callable(ValueType{});
        }
    }
};


template <typename ValueType>
void CbGmres<ValueType>::apply_impl(const LinOp *b, LinOp *x) const
{
    // Current workaround to get a lambda with a template argument (only the
    // type of `value` matters, the content does not)
    auto apply_templated = [&](auto value) {
        using storage_type = decltype(value);
        GKO_ASSERT_IS_SQUARE_MATRIX(system_matrix_);

        using Vector = matrix::Dense<ValueType>;
        using VectorNorms = matrix::Dense<remove_complex<ValueType>>;
        using Range3dHelper =
            gko::cb_gmres::Range3dHelper<ValueType, storage_type>;


        constexpr uint8 RelativeStoppingId{1};

        auto exec = this->get_executor();

        auto one_op = initialize<Vector>({one<ValueType>()}, exec);
        auto neg_one_op = initialize<Vector>({-one<ValueType>()}, exec);

        auto dense_b = as<const Vector>(b);
        auto dense_x = as<Vector>(x);
        auto residual = Vector::create_with_config_of(dense_b);
        /* The dimensions {x, y, z} explained for the krylov_bases:
         * - x: selects the krylov vector (which has krylov_dim + 1 vectors)
         * - y: selects the (row-)element of said krylov vector
         * - z: selects which column-element of said krylov vector should be
         *      used
         */
        const dim<3> krylov_bases_dim{krylov_dim_ + 1,
                                      system_matrix_->get_size()[1],
                                      dense_b->get_size()[1]};
        Range3dHelper helper(exec, krylov_bases_dim);
        auto krylov_bases_range = helper.get_range();

        auto next_krylov_basis = Vector::create_with_config_of(dense_b);
        std::shared_ptr<matrix::Dense<ValueType>> preconditioned_vector =
            Vector::create_with_config_of(dense_b);
        auto hessenberg = Vector::create(
            exec,
            dim<2>{krylov_dim_ + 1, krylov_dim_ * dense_b->get_size()[1]});
        auto buffer = Vector::create(
            exec, dim<2>{krylov_dim_ + 1, dense_b->get_size()[1]});
        auto givens_sin =
            Vector::create(exec, dim<2>{krylov_dim_, dense_b->get_size()[1]});
        auto givens_cos =
            Vector::create(exec, dim<2>{krylov_dim_, dense_b->get_size()[1]});
        auto residual_norm_collection = Vector::create(
            exec, dim<2>{krylov_dim_ + 1, dense_b->get_size()[1]});
        auto residual_norm =
            VectorNorms::create(exec, dim<2>{1, dense_b->get_size()[1]});
        // 1st row of arnoldi_norm: == eta * norm2(old_next_krylov_basis)
        //                          with eta == 1 / sqrt(2)
        //                          (computed right before updating
        //                          next_krylov_basis)
        // 2nd row of arnoldi_norm: The actual arnoldi norm
        //                          == norm2(next_krylov_basis)
        // 3rd row of arnoldi_norm: the infinity norm of next_krylov_basis
        //                          (ONLY when using a scalar accessor)
        auto arnoldi_norm =
            VectorNorms::create(exec, dim<2>{3, dense_b->get_size()[1]});
        Array<size_type> final_iter_nums(this->get_executor(),
                                         dense_b->get_size()[1]);
        auto y =
            Vector::create(exec, dim<2>{krylov_dim_, dense_b->get_size()[1]});

        bool one_changed{};
        Array<stopping_status> stop_status(this->get_executor(),
                                           dense_b->get_size()[1]);
        // reorth_status and num_reorth are both helper variables for GPU
        // implementations at the moment.
        // num_reorth := Number of vectors which require a re-orthogonalization
        // reorth_status := stopping status for the re-orthogonalization,
        //                  marking which RHS requires one, and which does not
        Array<stopping_status> reorth_status(this->get_executor(),
                                             dense_b->get_size()[1]);
        Array<size_type> num_reorth(this->get_executor(), 1);

        // Initialization
        exec->run(cb_gmres::make_initialize_1(
            dense_b, residual.get(), givens_sin.get(), givens_cos.get(),
            &stop_status, krylov_dim_));
        // residual = dense_b
        // givens_sin = givens_cos = 0
        system_matrix_->apply(neg_one_op.get(), dense_x, one_op.get(),
                              residual.get());
        // residual = residual - Ax

        exec->run(cb_gmres::make_initialize_2(
            residual.get(), residual_norm.get(), residual_norm_collection.get(),
            arnoldi_norm.get(), krylov_bases_range, next_krylov_basis.get(),
            &final_iter_nums, krylov_dim_));
        // residual_norm = norm(residual)
        // residual_norm_collection = {residual_norm, 0, ..., 0}
        // krylov_bases(:, 1) = residual / residual_norm
        // next_krylov_basis = residual / residual_norm
        // final_iter_nums = {0, ..., 0}

        auto stop_criterion = stop_criterion_factory_->generate(
            system_matrix_,
            std::shared_ptr<const LinOp>(b, [](const LinOp *) {}), x,
            residual.get());

        int total_iter = -1;
        size_type restart_iter = 0;

        auto before_preconditioner =
            matrix::Dense<ValueType>::create_with_config_of(dense_x);
        auto after_preconditioner =
            matrix::Dense<ValueType>::create_with_config_of(dense_x);

        Array<bool> stop_encountered_rhs(exec->get_master(),
                                         dense_b->get_size()[1]);
        Array<bool> fully_converged_rhs(exec->get_master(),
                                        dense_b->get_size()[1]);
        Array<stopping_status> host_stop_status(
            this->get_executor()->get_master(), stop_status);
        for (size_type i = 0; i < stop_encountered_rhs.get_num_elems(); ++i) {
            stop_encountered_rhs.get_data()[i] = false;
            fully_converged_rhs.get_data()[i] = false;
        }
        // Start only after this value with performing forced iterations after
        // convergence detection
        constexpr decltype(total_iter) start_force_reset{10};
        bool perform_reset{false};
        // Fraction of the krylov_dim_ (or total_iter if it is lower),
        // determining the number of forced iteration to perform
        constexpr decltype(krylov_dim_) forced_iteration_fraction{10};
        const decltype(krylov_dim_) forced_limit{krylov_dim_ /
                                                 forced_iteration_fraction};
        // Counter for the forced iterations. Start at max in order to properly
        // test convergence at the beginning
        decltype(krylov_dim_) forced_iterations{forced_limit};

        while (true) {
            ++total_iter;
            this->template log<log::Logger::iteration_complete>(
                this, total_iter, residual.get(), dense_x, residual_norm.get());
            // In the beginning, only force a fraction of the total iterations
            if (forced_iterations < forced_limit &&
                forced_iterations < total_iter / forced_iteration_fraction) {
                ++forced_iterations;
            } else {
                bool all_changed = stop_criterion->update()
                                       .num_iterations(total_iter)
                                       .residual(residual.get())
                                       .residual_norm(residual_norm.get())
                                       .solution(dense_x)
                                       .check(RelativeStoppingId, true,
                                              &stop_status, &one_changed);
                if (one_changed || all_changed) {
                    host_stop_status = stop_status;
                    bool host_array_changed{false};
                    for (size_type i = 0; i < host_stop_status.get_num_elems();
                         ++i) {
                        auto local_status = host_stop_status.get_data() + i;
                        // Ignore all actually converged ones!
                        if (fully_converged_rhs.get_data()[i]) {
                            continue;
                        }
                        if (local_status->has_converged()) {
                            // If convergence was detected earlier, or
                            // at the very beginning:
                            if (stop_encountered_rhs.get_data()[i] ||
                                total_iter < start_force_reset) {
                                fully_converged_rhs.get_data()[i] = true;
                            } else {
                                stop_encountered_rhs.get_data()[i] = true;
                                local_status->reset();
                                host_array_changed = true;
                            }
                        }
                    }
                    if (host_array_changed) {
                        perform_reset = true;
                        stop_status = host_stop_status;
                    } else {
                        // Stop here can happen if all RHS are "fully_converged"
                        // or if it was stopped for non-convergence reason
                        // (like time or iteration)
                        break;
                    }
                    forced_iterations = 0;

                } else {
                    for (size_type i = 0;
                         i < stop_encountered_rhs.get_num_elems(); ++i) {
                        stop_encountered_rhs.get_data()[i] = false;
                    }
                }
            }

            if (perform_reset || restart_iter == krylov_dim_) {
                perform_reset = false;
                // Restart
                // use a view in case this is called earlier
                auto hessenberg_view = hessenberg->create_submatrix(
                    span{0, restart_iter},
                    span{0, dense_b->get_size()[1] * (restart_iter)});

                exec->run(cb_gmres::make_step_2(
                    residual_norm_collection.get(),
                    krylov_bases_range.get_accessor().to_const(),
                    hessenberg_view.get(), y.get(), before_preconditioner.get(),
                    &final_iter_nums));
                // Solve upper triangular.
                // y = hessenberg \ residual_norm_collection

                this->get_preconditioner()->apply(before_preconditioner.get(),
                                                  after_preconditioner.get());
                dense_x->add_scaled(one_op.get(), after_preconditioner.get());
                // Solve x
                // x = x + get_preconditioner() * krylov_bases * y
                residual->copy_from(dense_b);
                // residual = dense_b
                system_matrix_->apply(neg_one_op.get(), dense_x, one_op.get(),
                                      residual.get());
                // residual = residual - Ax
                exec->run(cb_gmres::make_initialize_2(
                    residual.get(), residual_norm.get(),
                    residual_norm_collection.get(), arnoldi_norm.get(),
                    krylov_bases_range, next_krylov_basis.get(),
                    &final_iter_nums, krylov_dim_));
                // residual_norm = norm(residual)
                // residual_norm_collection = {residual_norm, 0, ..., 0}
                // krylov_bases(:, 1) = residual / residual_norm
                // next_krylov_basis = residual / residual_norm
                // final_iter_nums = {0, ..., 0}
                restart_iter = 0;
            }

            this->get_preconditioner()->apply(next_krylov_basis.get(),
                                              preconditioned_vector.get());
            // preconditioned_vector = get_preconditioner() *
            // next_krylov_basis

            // Do Arnoldi and givens rotation
            auto hessenberg_iter = hessenberg->create_submatrix(
                span{0, restart_iter + 2},
                span{dense_b->get_size()[1] * restart_iter,
                     dense_b->get_size()[1] * (restart_iter + 1)});
            auto buffer_iter = buffer->create_submatrix(
                span{0, restart_iter + 2}, span{0, dense_b->get_size()[1]});

            // Start of arnoldi
            system_matrix_->apply(preconditioned_vector.get(),
                                  next_krylov_basis.get());
            // next_krylov_basis = A * preconditioned_vector
            exec->run(cb_gmres::make_step_1(
                next_krylov_basis.get(), givens_sin.get(), givens_cos.get(),
                residual_norm.get(), residual_norm_collection.get(),
                krylov_bases_range, hessenberg_iter.get(), buffer_iter.get(),
                arnoldi_norm.get(), restart_iter, &final_iter_nums,
                &stop_status, &reorth_status, &num_reorth));
            // for i in 0:restart_iter
            //     hessenberg(restart_iter, i) = next_krylov_basis' *
            //     krylov_bases(:, i) next_krylov_basis  -=
            //     hessenberg(restart_iter, i) * krylov_bases(:, i)
            // end
            // hessenberg(restart_iter, restart_iter + 1) =
            // norm(next_krylov_basis) next_krylov_basis /=
            // hessenberg(restart_iter, restart_iter + 1) End of arnoldi
            // Start apply givens rotation for j in 0:restart_iter
            //     temp             =  cos(j)*hessenberg(j) +
            //                         sin(j)*hessenberg(j+1)
            //     hessenberg(j+1)  = -sin(j)*hessenberg(j) +
            //                         cos(j)*hessenberg(j+1)
            //     hessenberg(j)    =  temp;
            // end
            // Calculate sin and cos
            // hessenberg(restart_iter)   =
            // cos(restart_iter)*hessenberg(restart_iter) +
            //                      sin(restart_iter)*hessenberg(restart_iter)
            // hessenberg(restart_iter+1) = 0
            // End apply givens rotation
            // Calculate residual norm

            restart_iter++;
        }  // closes while(true)
        // Solve x

        auto hessenberg_small = hessenberg->create_submatrix(
            span{0, restart_iter},
            span{0, dense_b->get_size()[1] * (restart_iter)});

        exec->run(cb_gmres::make_step_2(
            residual_norm_collection.get(),
            krylov_bases_range.get_accessor().to_const(),
            hessenberg_small.get(), y.get(), before_preconditioner.get(),
            &final_iter_nums));
        // Solve upper triangular.
        // y = hessenberg \ residual_norm_collection
        this->get_preconditioner()->apply(before_preconditioner.get(),
                                          after_preconditioner.get());
        dense_x->add_scaled(one_op.get(), after_preconditioner.get());
        // Solve x
        // x = x + get_preconditioner() * krylov_bases * y
    };  // End of apply_lambda

    // Look which precision to use as the storage type
    helper<ValueType>::call(apply_templated, storage_precision_);
}


template <typename ValueType>
void CbGmres<ValueType>::apply_impl(const LinOp *alpha, const LinOp *b,
                                    const LinOp *residual_norm_collection,
                                    LinOp *x) const
{
    auto dense_x = as<matrix::Dense<ValueType>>(x);

    auto x_clone = dense_x->clone();
    this->apply(b, x_clone.get());
    dense_x->scale(residual_norm_collection);
    dense_x->add_scaled(alpha, x_clone.get());
}

#define GKO_DECLARE_CB_GMRES(_type1) class CbGmres<_type1>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_CB_GMRES);


}  // namespace solver
}  // namespace gko
