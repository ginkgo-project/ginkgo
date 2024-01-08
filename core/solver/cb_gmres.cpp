// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/solver/cb_gmres.hpp>


#include <type_traits>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/precision_dispatch.hpp>
#include <ginkgo/core/base/utils_helper.hpp>
#include <ginkgo/core/log/logger.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/base/extended_float.hpp"
#include "core/solver/cb_gmres_accessor.hpp"
#include "core/solver/cb_gmres_kernels.hpp"


namespace gko {
namespace solver {
namespace cb_gmres {
namespace {


GKO_REGISTER_OPERATION(initialize, cb_gmres::initialize);
GKO_REGISTER_OPERATION(restart, cb_gmres::restart);
GKO_REGISTER_OPERATION(arnoldi, cb_gmres::arnoldi);
GKO_REGISTER_OPERATION(solve_krylov, cb_gmres::solve_krylov);


}  // anonymous namespace
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
void CbGmres<ValueType>::apply_impl(const LinOp* b, LinOp* x) const
{
    if (!this->get_system_matrix()) {
        return;
    }
    precision_dispatch_real_complex<ValueType>(
        [this](auto dense_b, auto dense_x) {
            this->apply_dense_impl(dense_b, dense_x);
        },
        b, x);
}


template <typename ValueType>
void CbGmres<ValueType>::apply_dense_impl(
    const matrix::Dense<ValueType>* dense_b,
    matrix::Dense<ValueType>* dense_x) const
{
    // Current workaround to get a lambda with a template argument (only
    // the type of `value` matters, the content does not)
    auto apply_templated = [&](auto value) {
        using storage_type = decltype(value);

        using Vector = matrix::Dense<ValueType>;
        using VectorNorms = matrix::Dense<remove_complex<ValueType>>;
        using Range3dHelper =
            gko::cb_gmres::Range3dHelper<ValueType, storage_type>;


        constexpr uint8 RelativeStoppingId{1};

        auto exec = this->get_executor();

        auto one_op = initialize<Vector>({one<ValueType>()}, exec);
        auto neg_one_op = initialize<Vector>({-one<ValueType>()}, exec);

        const auto num_rows = this->get_size()[0];
        const auto num_rhs = dense_b->get_size()[1];
        const auto krylov_dim = this->get_krylov_dim();
        auto residual = Vector::create_with_config_of(dense_b);
        /* The dimensions {x, y, z} explained for the krylov_bases:
         * - x: selects the krylov vector (which has krylov_dim + 1 vectors)
         * - y: selects the (row-)element of said krylov vector
         * - z: selects which column-element of said krylov vector should be
         *      used
         */
        const dim<3> krylov_bases_dim{krylov_dim + 1, num_rows, num_rhs};
        Range3dHelper helper(exec, krylov_bases_dim);
        auto krylov_bases_range = helper.get_range();

        auto next_krylov_basis = Vector::create_with_config_of(dense_b);
        std::shared_ptr<matrix::Dense<ValueType>> preconditioned_vector =
            Vector::create_with_config_of(dense_b);
        auto hessenberg =
            Vector::create(exec, dim<2>{krylov_dim + 1, krylov_dim * num_rhs});
        auto buffer = Vector::create(exec, dim<2>{krylov_dim + 1, num_rhs});
        auto givens_sin = Vector::create(exec, dim<2>{krylov_dim, num_rhs});
        auto givens_cos = Vector::create(exec, dim<2>{krylov_dim, num_rhs});
        auto residual_norm_collection =
            Vector::create(exec, dim<2>{krylov_dim + 1, num_rhs});
        auto residual_norm = VectorNorms::create(exec, dim<2>{1, num_rhs});
        // 1st row of arnoldi_norm: == eta * norm2(old_next_krylov_basis)
        //                          with eta == 1 / sqrt(2)
        //                          (computed right before updating
        //                          next_krylov_basis)
        // 2nd row of arnoldi_norm: The actual arnoldi norm
        //                          == norm2(next_krylov_basis)
        // 3rd row of arnoldi_norm: the infinity norm of next_krylov_basis
        //                          (ONLY when using a scalar accessor)
        auto arnoldi_norm = VectorNorms::create(exec, dim<2>{3, num_rhs});
        array<size_type> final_iter_nums(this->get_executor(), num_rhs);
        auto y = Vector::create(exec, dim<2>{krylov_dim, num_rhs});

        bool one_changed{};
        array<char> reduction_tmp{this->get_executor()};
        array<stopping_status> stop_status(this->get_executor(), num_rhs);
        // reorth_status and num_reorth are both helper variables for GPU
        // implementations at the moment.
        // num_reorth := Number of vectors which require a re-orthogonalization
        // reorth_status := stopping status for the re-orthogonalization,
        //                  marking which RHS requires one, and which does not
        array<stopping_status> reorth_status(this->get_executor(), num_rhs);
        array<size_type> num_reorth(this->get_executor(), 1);

        // Initialization
        exec->run(cb_gmres::make_initialize(dense_b, residual.get(),
                                            givens_sin.get(), givens_cos.get(),
                                            &stop_status, krylov_dim));
        // residual = dense_b
        // givens_sin = givens_cos = 0
        this->get_system_matrix()->apply(neg_one_op, dense_x, one_op, residual);
        // residual = residual - Ax

        exec->run(cb_gmres::make_restart(
            residual.get(), residual_norm.get(), residual_norm_collection.get(),
            arnoldi_norm.get(), krylov_bases_range, next_krylov_basis.get(),
            &final_iter_nums, reduction_tmp, krylov_dim));
        // residual_norm = norm(residual)
        // residual_norm_collection = {residual_norm, 0, ..., 0}
        // krylov_bases(:, 1) = residual / residual_norm
        // next_krylov_basis = residual / residual_norm
        // final_iter_nums = {0, ..., 0}

        auto stop_criterion = this->get_stop_criterion_factory()->generate(
            this->get_system_matrix(),
            std::shared_ptr<const LinOp>(dense_b, [](const LinOp*) {}), dense_x,
            residual.get());

        int total_iter = -1;
        size_type restart_iter = 0;

        auto before_preconditioner =
            matrix::Dense<ValueType>::create_with_config_of(dense_x);
        auto after_preconditioner =
            matrix::Dense<ValueType>::create_with_config_of(dense_x);

        array<bool> stop_encountered_rhs(exec->get_master(), num_rhs);
        array<bool> fully_converged_rhs(exec->get_master(), num_rhs);
        array<stopping_status> host_stop_status(
            this->get_executor()->get_master(), stop_status);
        for (size_type i = 0; i < stop_encountered_rhs.get_size(); ++i) {
            stop_encountered_rhs.get_data()[i] = false;
            fully_converged_rhs.get_data()[i] = false;
        }
        // Start only after this value with performing forced iterations after
        // convergence detection
        constexpr int start_force_reset{10};
        bool perform_reset{false};
        // Fraction of the krylov_dim (or total_iter if it is lower),
        // determining the number of forced iteration to perform
        constexpr size_type forced_iteration_fraction{10};
        const size_type forced_limit{krylov_dim / forced_iteration_fraction};
        // Counter for the forced iterations. Start at max in order to properly
        // test convergence at the beginning
        size_type forced_iterations{forced_limit};

        while (true) {
            ++total_iter;
            // In the beginning, only force a fraction of the total iterations
            if (forced_iterations < forced_limit &&
                forced_iterations < total_iter / forced_iteration_fraction) {
                this->template log<log::Logger::iteration_complete>(
                    this, dense_b, dense_x, total_iter, residual.get(),
                    residual_norm.get(), nullptr, &stop_status, false);
                ++forced_iterations;
            } else {
                bool all_changed = stop_criterion->update()
                                       .num_iterations(total_iter)
                                       .residual(residual)
                                       .residual_norm(residual_norm)
                                       .solution(dense_x)
                                       .check(RelativeStoppingId, true,
                                              &stop_status, &one_changed);
                this->template log<log::Logger::iteration_complete>(
                    this, dense_b, dense_x, total_iter, residual.get(),
                    residual_norm.get(), nullptr, &stop_status, all_changed);
                if (one_changed || all_changed) {
                    host_stop_status = stop_status;
                    bool host_array_changed{false};
                    for (size_type i = 0; i < host_stop_status.get_size();
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
                    for (size_type i = 0; i < stop_encountered_rhs.get_size();
                         ++i) {
                        stop_encountered_rhs.get_data()[i] = false;
                    }
                }
            }

            if (perform_reset || restart_iter == krylov_dim) {
                perform_reset = false;
                // Restart
                // use a view in case this is called earlier
                auto hessenberg_view = hessenberg->create_submatrix(
                    span{0, restart_iter}, span{0, num_rhs * (restart_iter)});

                exec->run(cb_gmres::make_solve_krylov(
                    residual_norm_collection.get(),
                    krylov_bases_range.get_accessor().to_const(),
                    hessenberg_view.get(), y.get(), before_preconditioner.get(),
                    &final_iter_nums));
                // Solve upper triangular.
                // y = hessenberg \ residual_norm_collection

                this->get_preconditioner()->apply(before_preconditioner,
                                                  after_preconditioner);
                dense_x->add_scaled(one_op, after_preconditioner);
                // Solve x
                // x = x + get_preconditioner() * krylov_bases * y
                residual->copy_from(dense_b);
                // residual = dense_b
                this->get_system_matrix()->apply(neg_one_op, dense_x, one_op,
                                                 residual);
                // residual = residual - Ax
                exec->run(cb_gmres::make_restart(
                    residual.get(), residual_norm.get(),
                    residual_norm_collection.get(), arnoldi_norm.get(),
                    krylov_bases_range, next_krylov_basis.get(),
                    &final_iter_nums, reduction_tmp, krylov_dim));
                // residual_norm = norm(residual)
                // residual_norm_collection = {residual_norm, 0, ..., 0}
                // krylov_bases(:, 1) = residual / residual_norm
                // next_krylov_basis = residual / residual_norm
                // final_iter_nums = {0, ..., 0}
                restart_iter = 0;
            }

            this->get_preconditioner()->apply(next_krylov_basis,
                                              preconditioned_vector);
            // preconditioned_vector = get_preconditioner() *
            // next_krylov_basis

            // Do Arnoldi and givens rotation
            auto hessenberg_iter = hessenberg->create_submatrix(
                span{0, restart_iter + 2},
                span{num_rhs * restart_iter, num_rhs * (restart_iter + 1)});
            auto buffer_iter = buffer->create_submatrix(
                span{0, restart_iter + 2}, span{0, num_rhs});

            // Start of arnoldi
            this->get_system_matrix()->apply(preconditioned_vector,
                                             next_krylov_basis);
            // next_krylov_basis = A * preconditioned_vector
            exec->run(cb_gmres::make_arnoldi(
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
            span{0, restart_iter}, span{0, num_rhs * restart_iter});

        exec->run(cb_gmres::make_solve_krylov(
            residual_norm_collection.get(),
            krylov_bases_range.get_accessor().to_const(),
            hessenberg_small.get(), y.get(), before_preconditioner.get(),
            &final_iter_nums));
        // Solve upper triangular.
        // y = hessenberg \ residual_norm_collection
        this->get_preconditioner()->apply(before_preconditioner,
                                          after_preconditioner);
        dense_x->add_scaled(one_op, after_preconditioner);
        // Solve x
        // x = x + get_preconditioner() * krylov_bases * y
    };  // End of apply_lambda

    // Look which precision to use as the storage type
    helper<ValueType>::call(apply_templated, this->get_storage_precision());
}


template <typename ValueType>
void CbGmres<ValueType>::apply_impl(const LinOp* alpha, const LinOp* b,
                                    const LinOp* beta, LinOp* x) const
{
    if (!this->get_system_matrix()) {
        return;
    }
    precision_dispatch_real_complex<ValueType>(
        [this](auto dense_alpha, auto dense_b, auto dense_beta, auto dense_x) {
            auto x_clone = dense_x->clone();
            this->apply_dense_impl(dense_b, x_clone.get());
            dense_x->scale(dense_beta);
            dense_x->add_scaled(dense_alpha, x_clone);
        },
        alpha, b, beta, x);
}

#define GKO_DECLARE_CB_GMRES(_type1) class CbGmres<_type1>
#define GKO_DECLARE_CB_GMRES_TRAITS(_type1) \
    struct workspace_traits<CbGmres<_type1>>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_CB_GMRES);
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_CB_GMRES_TRAITS);


}  // namespace solver
}  // namespace gko
