/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2020, the Ginkgo authors
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

#include <ginkgo/core/solver/gmres_mixed.hpp>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/name_demangling.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/identity.hpp>


#include <iostream>
#include <typeinfo>


#include "core/solver/gmres_mixed_accessor.hpp"
#include "core/solver/gmres_mixed_kernels.hpp"


//#define TIMING 1


#ifdef TIMING
using double_seconds = std::chrono::duration<double>;
#define TIMING_STEPS 1
#endif


namespace gko {
namespace solver {


namespace gmres_mixed {


GKO_REGISTER_OPERATION(initialize_1, gmres_mixed::initialize_1);
GKO_REGISTER_OPERATION(initialize_2, gmres_mixed::initialize_2);
GKO_REGISTER_OPERATION(step_1, gmres_mixed::step_1);
GKO_REGISTER_OPERATION(step_2, gmres_mixed::step_2);


}  // namespace gmres_mixed

// TODO: Remove output
template <typename T>
struct type_string {
    static const char *get() { return typeid(T).name(); }
};

#define GKO_SPECIALIZE_TYPE_STRING(type)           \
    template <>                                    \
    struct type_string<type> {                     \
        static const char *get() { return #type; } \
    }

GKO_SPECIALIZE_TYPE_STRING(int32);
GKO_SPECIALIZE_TYPE_STRING(int16);
GKO_SPECIALIZE_TYPE_STRING(double);
GKO_SPECIALIZE_TYPE_STRING(float);
GKO_SPECIALIZE_TYPE_STRING(half);


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
                  "The count variable must be larger or equal zero.");
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
                     gko::solver::gmres_mixed_storage_precision st)
    {
        switch (st) {
        case gmres_mixed_storage_precision::reduce1:
            callable(reduce_precision_count<ValueType, 1>{});
            break;
        case gmres_mixed_storage_precision::reduce2:
            callable(reduce_precision_count<ValueType, 2>{});
            break;
        case gmres_mixed_storage_precision::integer:
            callable(to_integer<ValueType>{});
            break;
        case gmres_mixed_storage_precision::ireduce1:
            callable(to_integer<reduce_precision_count<ValueType, 1>>{});
            break;
        case gmres_mixed_storage_precision::ireduce2:
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
                     gko::solver::gmres_mixed_storage_precision st)
    {
        switch (st) {
        case gmres_mixed_storage_precision::reduce1:
            callable(reduce_precision_skip_count<ValueType, skip_type, 1>{});
            break;
        case gmres_mixed_storage_precision::reduce2:
            callable(reduce_precision_skip_count<ValueType, skip_type, 2>{});
            break;
        case gmres_mixed_storage_precision::integer:
        case gmres_mixed_storage_precision::ireduce1:
        case gmres_mixed_storage_precision::ireduce2:
            GKO_NOT_SUPPORTED(st);
            break;
        default:
            callable(ValueType{});
        }
    }
};


template <typename ValueType>
void GmresMixed<ValueType>::apply_impl(const LinOp *b, LinOp *x) const
{
    // Current workaround to get a lambda with a template argument (only the
    // type of `value` matters, the content does not)
    auto apply_templated = [&](auto value) {
        using storage_type = decltype(value);
        GKO_ASSERT_IS_SQUARE_MATRIX(system_matrix_);

        using Vector = matrix::Dense<ValueType>;
        using VectorNorms = matrix::Dense<remove_complex<ValueType>>;
        using LowArray = Array<storage_type>;
        // using KrylovAccessor = kernels::Accessor3d<storage_type,
        // ValueType>;
        using Accessor3dHelper =
            kernels::Accessor3dHelper<ValueType, storage_type>;


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
        // const dim<2> krylov_bases_dim{
        //     system_matrix_->get_size()[1],
        //     (krylov_dim_ + 1) * dense_b->get_size()[1]};
        // const size_type krylov_bases_stride = krylov_bases_dim[1];
        // LowArray krylov_bases(exec, krylov_bases_dim[0] *
        // krylov_bases_stride); KrylovAccessor
        // krylov_bases_range(krylov_bases.get_data(),
        //                                     krylov_bases_stride);
        Accessor3dHelper helper(exec, krylov_bases_dim);
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
        auto b_norm =
            VectorNorms::create(exec, dim<2>{1, dense_b->get_size()[1]});
        // TODO: write description what the different rows represent
        // The optional entry stores the infinity_norm of each
        // next_krylov_vector, which is only used to compute the scale
        auto arnoldi_norm =
            VectorNorms::create(exec, dim<2>{3, dense_b->get_size()[1]});
        Array<size_type> final_iter_nums(this->get_executor(),
                                         dense_b->get_size()[1]);
        auto y =
            Vector::create(exec, dim<2>{krylov_dim_, dense_b->get_size()[1]});

        bool one_changed{};
        Array<stopping_status> stop_status(this->get_executor(),
                                           dense_b->get_size()[1]);
        Array<stopping_status> reorth_status(this->get_executor(),
                                             dense_b->get_size()[1]);
        Array<size_type> num_reorth(this->get_executor(),
                                    dense_b->get_size()[1]);
        int num_restarts = 0, num_reorth_steps = 0, num_reorth_vectors = 0;

        // std::cout << "Before initializate_1" << std::endl;
        // Initialization
        exec->run(gmres_mixed::make_initialize_1(
            dense_b, b_norm.get(), residual.get(), givens_sin.get(),
            givens_cos.get(), &stop_status, krylov_dim_));
        // b_norm = norm(b)
        // residual = dense_b
        // givens_sin = givens_cos = 0
        system_matrix_->apply(neg_one_op.get(), dense_x, one_op.get(),
                              residual.get());
        // residual = residual - Ax

        // std::cout << "Before initializate_2" << std::endl;
        exec->run(gmres_mixed::make_initialize_2(
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

#ifdef TIMING
        exec->synchronize();
        auto start = std::chrono::steady_clock::now();
#ifdef TIMING_STEPS
        auto time_RSTRT = start - start;
        auto time_SPMV = start - start;
        auto time_STEP1 = start - start;
#endif
#endif
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
        bool perform_reset{false};
        decltype(krylov_dim_) forced_iterations{0};
        const decltype(forced_iterations) forced_limit{krylov_dim_ / 10};
        decltype(krylov_dim_) total_checks{0};  // TODO: Remove debug output
        const char *type_str = type_string<storage_type>::get();
        // TODO: take care of multiple RHS. Currently, we restart
        // everything,
        //       even though a lot of parts might have already converged!
        //       Use `one_changed` to take care of that!
        while (true) {
            ++total_iter;
            this->template log<log::Logger::iteration_complete>(
                this, total_iter, residual.get(), dense_x, residual_norm.get());
            if (forced_iterations < forced_limit) {
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
                    /*
                    std::cout << type_str << ": " << ++total_checks
                              << ". check in iteration " << total_iter << ";
                    "
                              << forced_iterations << " / " << forced_limit
                    <<
                    '\n';
                    */
                    host_stop_status = stop_status;
                    bool host_array_changed{false};
                    for (size_type i = 0; i < host_stop_status.get_num_elems();
                         ++i) {
                        auto local_status = host_stop_status.get_data() + i;
                        // TODO: ignore all actually converged ones!
                        if (fully_converged_rhs.get_data()[i]) {
                            continue;
                        }
                        if (local_status->has_converged()) {
                            if (stop_encountered_rhs.get_data()[i]) {
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
#ifdef TIMING_STEPS
                exec->synchronize();
                auto t_aux_0 = std::chrono::steady_clock::now();
#endif
                num_restarts++;
                // Restart
                // use a view in case this is called earlier
                auto hessenberg_view = hessenberg->create_submatrix(
                    span{0, restart_iter},
                    span{0, dense_b->get_size()[1] * (restart_iter)});

                exec->run(gmres_mixed::make_step_2(
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
                exec->run(gmres_mixed::make_initialize_2(
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
#ifdef TIMING_STEPS
                exec->synchronize();
                time_RSTRT += std::chrono::steady_clock::now() - t_aux_0;
#endif
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

#ifdef TIMING_STEPS
            exec->synchronize();
            auto t_aux_1 = std::chrono::steady_clock::now();
#endif
            // Start of arnoldi
            system_matrix_->apply(preconditioned_vector.get(),
                                  next_krylov_basis.get());
            // next_krylov_basis = A * preconditioned_vector
#ifdef TIMING_STEPS
            exec->synchronize();
            time_SPMV += std::chrono::steady_clock::now() - t_aux_1;
#endif

#ifdef TIMING_STEPS
            exec->synchronize();
            auto t_aux_2 = std::chrono::steady_clock::now();
#endif
            exec->run(gmres_mixed::make_step_1(
                next_krylov_basis.get(), givens_sin.get(), givens_cos.get(),
                residual_norm.get(), residual_norm_collection.get(),
                krylov_bases_range, hessenberg_iter.get(), buffer_iter.get(),
                b_norm.get(), arnoldi_norm.get(), restart_iter,
                &final_iter_nums, &stop_status, &reorth_status, &num_reorth,
                &num_reorth_steps, &num_reorth_vectors));
#ifdef TIMING_STEPS
            exec->synchronize();
            time_STEP1 += std::chrono::steady_clock::now() - t_aux_2;
#endif
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
        }
        std::cout << type_str << ": " << total_checks
                  << ": exiting in iteration " << total_iter << "; "
                  << forced_iterations << " / " << forced_limit << '\n';

        // Solve x
#ifdef TIMING_STEPS
        exec->synchronize();
        auto t_aux_3 = std::chrono::steady_clock::now();
#endif

        auto hessenberg_small = hessenberg->create_submatrix(
            span{0, restart_iter},
            span{0, dense_b->get_size()[1] * (restart_iter)});

        exec->run(gmres_mixed::make_step_2(
            residual_norm_collection.get(),
            krylov_bases_range.get_accessor().to_const(),
            hessenberg_small.get(), y.get(), before_preconditioner.get(),
            &final_iter_nums));
        // Solve upper triangular.
        // y = hessenberg \ residual_norm_collection
#ifdef TIMING_STEPS
        exec->synchronize();
        auto time_STEP2 = std::chrono::steady_clock::now() - t_aux_3;
#endif

#ifdef TIMING_STEPS
        exec->synchronize();
        auto t_aux_4 = std::chrono::steady_clock::now();
#endif
        this->get_preconditioner()->apply(before_preconditioner.get(),
                                          after_preconditioner.get());
        dense_x->add_scaled(one_op.get(), after_preconditioner.get());
#ifdef TIMING_STEPS
        exec->synchronize();
        auto time_SOLVEX = std::chrono::steady_clock::now() - t_aux_4;
#endif
        // Solve x
        // x = x + get_preconditioner() * krylov_bases * y

#ifdef TIMING
        exec->synchronize();
        auto time = std::chrono::steady_clock::now() - start;
#endif
#ifdef TIMING
        std::cout << "total_iter = " << total_iter << std::endl;
        std::cout << "num_restarts = " << num_restarts << std::endl;
        std::cout << "reorth_steps = " << num_reorth_steps << std::endl;
        std::cout << "reorth_vectors = " << num_reorth_vectors << std::endl;
        std::cout << "time = "
                  << std::chrono::duration_cast<double_seconds>(time).count()
                  << std::endl;
#ifdef TIMING_STEPS
        std::cout
            << "time_RSTRT = "
            << std::chrono::duration_cast<double_seconds>(time_RSTRT).count()
            << std::endl;
        std::cout
            << "time_SPMV = "
            << std::chrono::duration_cast<double_seconds>(time_SPMV).count()
            << std::endl;
        std::cout
            << "time_STEP1 = "
            << std::chrono::duration_cast<double_seconds>(time_STEP1).count()
            << std::endl;
        std::cout
            << "time_STEP2 = "
            << std::chrono::duration_cast<double_seconds>(time_STEP2).count()
            << std::endl;
        std::cout
            << "time_SOLVEX = "
            << std::chrono::duration_cast<double_seconds>(time_SOLVEX).count()
            << std::endl;
#endif
        write(std::cout, lend(residual_norm));
#endif
    };  // End of apply_lambda

    // Look which precision to use as the storage type
    helper<ValueType>::call(apply_templated, storage_precision_);
}


template <typename ValueType>
void GmresMixed<ValueType>::apply_impl(const LinOp *alpha, const LinOp *b,
                                       const LinOp *residual_norm_collection,
                                       LinOp *x) const
{
    auto dense_x = as<matrix::Dense<ValueType>>(x);

    auto x_clone = dense_x->clone();
    this->apply(b, x_clone.get());
    dense_x->scale(residual_norm_collection);
    dense_x->add_scaled(alpha, x_clone.get());
}

#define GKO_DECLARE_GMRES_MIXED(_type1) class GmresMixed<_type1>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_GMRES_MIXED);


}  // namespace solver
}  // namespace gko
