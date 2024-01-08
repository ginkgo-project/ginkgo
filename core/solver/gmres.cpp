// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/solver/gmres.hpp>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/name_demangling.hpp>
#include <ginkgo/core/base/precision_dispatch.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/identity.hpp>


#include "core/distributed/helpers.hpp"
#include "core/solver/common_gmres_kernels.hpp"
#include "core/solver/gmres_kernels.hpp"
#include "core/solver/solver_boilerplate.hpp"


namespace gko {
namespace solver {
namespace gmres {
namespace {


GKO_REGISTER_OPERATION(initialize, common_gmres::initialize);
GKO_REGISTER_OPERATION(restart, gmres::restart);
GKO_REGISTER_OPERATION(hessenberg_qr, common_gmres::hessenberg_qr);
GKO_REGISTER_OPERATION(solve_krylov, common_gmres::solve_krylov);
GKO_REGISTER_OPERATION(multi_axpy, gmres::multi_axpy);


}  // anonymous namespace
}  // namespace gmres


template <typename ValueType>
std::unique_ptr<LinOp> Gmres<ValueType>::transpose() const
{
    return build()
        .with_generated_preconditioner(
            share(as<Transposable>(this->get_preconditioner())->transpose()))
        .with_criteria(this->get_stop_criterion_factory())
        .with_krylov_dim(this->get_krylov_dim())
        .with_flexible(this->get_parameters().flexible)
        .on(this->get_executor())
        ->generate(
            share(as<Transposable>(this->get_system_matrix())->transpose()));
}


template <typename ValueType>
std::unique_ptr<LinOp> Gmres<ValueType>::conj_transpose() const
{
    return build()
        .with_generated_preconditioner(share(
            as<Transposable>(this->get_preconditioner())->conj_transpose()))
        .with_criteria(this->get_stop_criterion_factory())
        .with_krylov_dim(this->get_krylov_dim())
        .with_flexible(this->get_parameters().flexible)
        .on(this->get_executor())
        ->generate(share(
            as<Transposable>(this->get_system_matrix())->conj_transpose()));
}


template <typename ValueType>
void Gmres<ValueType>::apply_impl(const LinOp* b, LinOp* x) const
{
    if (!this->get_system_matrix()) {
        return;
    }
    experimental::precision_dispatch_real_complex_distributed<ValueType>(
        [this](auto dense_b, auto dense_x) {
            this->apply_dense_impl(dense_b, dense_x);
        },
        b, x);
}


template <typename ValueType, typename = void>
struct help_compute_norm {
    template <typename VectorType>
    static void compute_next_krylov_norm_into_hessenberg(
        const VectorType* next_krylov,
        matrix::Dense<ValueType>* hessenberg_norm_entry,
        matrix::Dense<remove_complex<ValueType>>*, array<char>& reduction_tmp)
    {
        next_krylov->compute_norm2(hessenberg_norm_entry, reduction_tmp);
    }
};

template <typename ValueType>
struct help_compute_norm<ValueType,
                         std::enable_if_t<is_complex_s<ValueType>::value>> {
    template <typename VectorType>
    static void compute_next_krylov_norm_into_hessenberg(
        const VectorType* next_krylov,
        matrix::Dense<ValueType>* hessenberg_norm_entry,
        matrix::Dense<remove_complex<ValueType>>* next_krylov_norm_tmp,
        array<char>& reduction_tmp)
    {
        next_krylov->compute_norm2(next_krylov_norm_tmp, reduction_tmp);
        next_krylov_norm_tmp->make_complex(hessenberg_norm_entry);
    }
};


template <typename ValueType>
template <typename VectorType>
void Gmres<ValueType>::apply_dense_impl(const VectorType* dense_b,
                                        VectorType* dense_x) const
{
    using Vector = VectorType;
    using LocalVector = matrix::Dense<typename Vector::value_type>;
    using NormVector = typename LocalVector::absolute_type;
    using ws = workspace_traits<Gmres>;

    constexpr uint8 RelativeStoppingId{1};

    auto exec = this->get_executor();
    this->setup_workspace();
    const auto is_flexible = this->get_parameters().flexible;
    const auto num_rows = this->get_size()[0];
    const auto local_num_rows =
        ::gko::detail::get_local(dense_b)->get_size()[0];
    const auto num_rhs = dense_b->get_size()[1];
    const auto krylov_dim = this->get_krylov_dim();
    GKO_SOLVER_VECTOR(residual, dense_b);
    GKO_SOLVER_VECTOR(preconditioned_vector, dense_b);
    // TODO: check whether it can be empty when it is flexible GMRES
    auto krylov_bases = this->create_workspace_op_with_type_of(
        ws::krylov_bases, dense_b, dim<2>{num_rows * (krylov_dim + 1), num_rhs},
        dim<2>{local_num_rows * (krylov_dim + 1), num_rhs});
    VectorType* preconditioned_krylov_bases = nullptr;
    if (is_flexible) {
        preconditioned_krylov_bases = this->create_workspace_op_with_type_of(
            ws::preconditioned_krylov_bases, dense_b,
            dim<2>{num_rows * (krylov_dim + 1), num_rhs},
            dim<2>{local_num_rows * (krylov_dim + 1), num_rhs});
    }
    // rows: rows of Hessenberg matrix, columns: block for each entry
    auto hessenberg = this->template create_workspace_op<LocalVector>(
        ws::hessenberg, dim<2>{krylov_dim + 1, krylov_dim * num_rhs});
    auto givens_sin = this->template create_workspace_op<LocalVector>(
        ws::givens_sin, dim<2>{krylov_dim, num_rhs});
    auto givens_cos = this->template create_workspace_op<LocalVector>(
        ws::givens_cos, dim<2>{krylov_dim, num_rhs});
    auto residual_norm_collection =
        this->template create_workspace_op<LocalVector>(
            ws::residual_norm_collection, dim<2>{krylov_dim + 1, num_rhs});
    auto residual_norm = this->template create_workspace_op<NormVector>(
        ws::residual_norm, dim<2>{1, num_rhs});
    auto y = this->template create_workspace_op<LocalVector>(
        ws::y, dim<2>{krylov_dim, num_rhs});
    // next_krylov_norm_tmp is only required for complex types to move real
    // values into a complex matrix
    auto next_krylov_norm_tmp = this->template create_workspace_op<NormVector>(
        ws::next_krylov_norm_tmp,
        dim<2>{1, is_complex_s<ValueType>::value ? num_rhs : 0});

    GKO_SOLVER_VECTOR(before_preconditioner, dense_x);
    GKO_SOLVER_VECTOR(after_preconditioner, dense_x);

    GKO_SOLVER_ONE_MINUS_ONE();

    bool one_changed{};
    GKO_SOLVER_STOP_REDUCTION_ARRAYS();
    auto& final_iter_nums = this->template create_workspace_array<size_type>(
        ws::final_iter_nums, num_rhs);

    // Initialization
    // residual = dense_b
    // givens_sin = givens_cos = 0
    // reset stop status
    exec->run(gmres::make_initialize(
        gko::detail::get_local(dense_b), gko::detail::get_local(residual),
        givens_sin, givens_cos, stop_status.get_data()));
    // residual = residual - Ax
    this->get_system_matrix()->apply(neg_one_op, dense_x, one_op, residual);

    // residual_norm = norm(residual)
    residual->compute_norm2(residual_norm, reduction_tmp);
    // residual_norm_collection = {residual_norm, unchanged}
    // krylov_bases(:, 1) = residual / residual_norm
    // final_iter_nums = {0, ..., 0}
    exec->run(gmres::make_restart(gko::detail::get_local(residual),
                                  residual_norm, residual_norm_collection,
                                  gko::detail::get_local(krylov_bases),
                                  final_iter_nums.get_data()));

    auto stop_criterion = this->get_stop_criterion_factory()->generate(
        this->get_system_matrix(),
        std::shared_ptr<const LinOp>(dense_b, [](const LinOp*) {}), dense_x,
        residual);

    int total_iter = -1;
    size_type restart_iter = 0;

    /* Memory movement summary for average iteration with krylov_dim d:
     * (5/2d+21/2+14/d)n * values + (1+1/d) * matrix/preconditioner storage
     * 1x SpMV:                2n * values + storage
     * 1x Preconditioner:      2n * values + storage
     * MGS:          (5/2d+11/2)n = sum k=0 to d-1 of (5k+8)n/d
     *       1x dots           2(k+1)n in iteration k (0-based)
     *       1x axpys          3(k+1)n in iteration k (0-based)
     *       1x norm2               n
     *       1x scal               2n
     * Restart:         (1+14/d)n  (every dth iteration)
     *       1x gemv           (d+1)n
     *       1x Preconditioner     2n * values + storage
     *       1x axpy               3n
     *       1x copy               2n
     *       1x Advanced SpMV      3n * values + storage
     *       1x norm2               n
     *       1x scal               2n
     */
    while (true) {
        ++total_iter;
        bool all_stopped =
            stop_criterion->update()
                .num_iterations(total_iter)
                .residual(residual)
                .residual_norm(residual_norm)
                .solution(dense_x)
                .check(RelativeStoppingId, false, &stop_status, &one_changed);
        this->template log<log::Logger::iteration_complete>(
            this, dense_b, dense_x, total_iter, residual, residual_norm,
            nullptr, &stop_status, all_stopped);
        if (all_stopped) {
            break;
        }

        if (restart_iter == krylov_dim) {
            // Restart
            // Solve upper triangular.
            // y = hessenberg \ residual_norm_collection
            exec->run(gmres::make_solve_krylov(residual_norm_collection,
                                               hessenberg, y,
                                               final_iter_nums.get_const_data(),
                                               stop_status.get_const_data()));
            // before_preconditioner = krylov_bases * y
            exec->run(gmres::make_multi_axpy(
                gko::detail::get_local(krylov_bases), y,
                gko::detail::get_local(before_preconditioner),
                final_iter_nums.get_const_data(), stop_status.get_data()));

            // x = x + get_preconditioner() * before_preconditioner
            this->get_preconditioner()->apply(before_preconditioner,
                                              after_preconditioner);
            dense_x->add_scaled(one_op, after_preconditioner);
            // residual = dense_b
            residual->copy_from(dense_b);
            // residual = residual - Ax
            this->get_system_matrix()->apply(neg_one_op, dense_x, one_op,
                                             residual);
            // residual_norm = norm(residual)
            residual->compute_norm2(residual_norm, reduction_tmp);
            // residual_norm_collection = {residual_norm, unchanged}
            // krylov_bases(:, 1) = residual / residual_norm
            // final_iter_nums = {0, ..., 0}
            exec->run(gmres::make_restart(
                gko::detail::get_local(residual), residual_norm,
                residual_norm_collection, gko::detail::get_local(krylov_bases),
                final_iter_nums.get_data()));
            restart_iter = 0;
        }
        auto this_krylov = ::gko::detail::create_submatrix_helper(
            krylov_bases, dim<2>{num_rows, num_rhs},
            span{local_num_rows * restart_iter,
                 local_num_rows * (restart_iter + 1)},
            span{0, num_rhs});

        auto next_krylov = ::gko::detail::create_submatrix_helper(
            krylov_bases, dim<2>{num_rows, num_rhs},
            span{local_num_rows * (restart_iter + 1),
                 local_num_rows * (restart_iter + 2)},
            span{0, num_rhs});
        std::unique_ptr<VectorType> preconditioned_krylov;
        auto preconditioned_krylov_vector = preconditioned_vector;
        if (is_flexible) {
            preconditioned_krylov = ::gko::detail::create_submatrix_helper(
                preconditioned_krylov_bases, dim<2>{num_rows, num_rhs},
                span{local_num_rows * restart_iter,
                     local_num_rows * (restart_iter + 1)},
                span{0, num_rhs});
            preconditioned_krylov_vector = preconditioned_krylov.get();
        }
        // preconditioned_krylov_vector = get_preconditioner() * this_krylov
        this->get_preconditioner()->apply(this_krylov,
                                          preconditioned_krylov_vector);

        // Create view of current column in the hessenberg matrix:
        // hessenberg_iter = hessenberg(:, restart_iter);
        auto hessenberg_iter = hessenberg->create_submatrix(
            span{0, restart_iter + 2},
            span{num_rhs * restart_iter, num_rhs * (restart_iter + 1)});

        // Start of Arnoldi
        // next_krylov = A * preconditioned_krylov_vector
        this->get_system_matrix()->apply(preconditioned_krylov_vector,
                                         next_krylov);

        for (size_type i = 0; i <= restart_iter; i++) {
            // orthogonalize against krylov_bases(:, i):
            // hessenberg(i, restart_iter) = next_krylov' * krylov_bases(:, i)
            // next_krylov -= hessenberg(i, restart_iter) * krylov_bases(:, i)
            auto hessenberg_entry = hessenberg_iter->create_submatrix(
                span{i, i + 1}, span{0, num_rhs});
            auto krylov_basis = ::gko::detail::create_submatrix_helper(
                krylov_bases, dim<2>{num_rows, num_rhs},
                span{local_num_rows * i, local_num_rows * (i + 1)},
                span{0, num_rhs});
            next_krylov->compute_conj_dot(krylov_basis, hessenberg_entry,
                                          reduction_tmp);
            next_krylov->sub_scaled(hessenberg_entry, krylov_basis);
        }
        // normalize next_krylov:
        // hessenberg(restart_iter+1, restart_iter) = norm(next_krylov)
        // next_krylov /= hessenberg(restart_iter+1, restart_iter)
        auto hessenberg_norm_entry = hessenberg_iter->create_submatrix(
            span{restart_iter + 1, restart_iter + 2}, span{0, num_rhs});
        help_compute_norm<ValueType>::compute_next_krylov_norm_into_hessenberg(
            next_krylov.get(), hessenberg_norm_entry.get(),
            next_krylov_norm_tmp, reduction_tmp);
        next_krylov->inv_scale(hessenberg_norm_entry);
        // End of Arnoldi

        // update QR factorization and Krylov RHS for last column:
        // apply givens rotation
        // for j in 0:restart_iter(exclude)
        //     temp             =  cos(j)*hessenberg(j) +
        //                         sin(j)*hessenberg(j+1)
        //     hessenberg(j+1)  = -conj(sin(j))*hessenberg(j) +
        //                         conj(cos(j))*hessenberg(j+1)
        //     hessenberg(j)    =  temp;
        // end
        // calculate next Givens parameters
        // this_hess = hessenberg(restart_iter)
        // next_hess = hessenberg(restart_iter+1)
        // hypotenuse = ||(this_hess, next_hess)||
        // cos(restart_iter) = conj(this_hess) / hypotenuse
        // sin(restart_iter) = conj(next_hess) / hypotenuse
        // update Krylov approximation of b, apply new Givens rotation
        // this_rnc = residual_norm_collection(restart_iter)
        // residual_norm = abs(-conj(sin(restart_iter)) * this_rnc)
        // residual_norm_collection(restart_iter) =
        //              cos(restart_iter) * this_rnc
        // residual_norm_collection(restart_iter + 1) =
        //              -conj(sin(restart_iter)) * this_rnc
        exec->run(gmres::make_hessenberg_qr(
            givens_sin, givens_cos, residual_norm, residual_norm_collection,
            hessenberg_iter.get(), restart_iter, final_iter_nums.get_data(),
            stop_status.get_const_data()));

        restart_iter++;
    }

    auto hessenberg_small = hessenberg->create_submatrix(
        span{0, restart_iter}, span{0, num_rhs * (restart_iter)});

    // Solve upper triangular.
    // y = hessenberg \ residual_norm_collection
    exec->run(gmres::make_solve_krylov(
        residual_norm_collection, hessenberg_small.get(), y,
        final_iter_nums.get_const_data(), stop_status.get_const_data()));
    if (is_flexible) {
        auto preconditioned_krylov_bases_small =
            ::gko::detail::create_submatrix_helper(
                preconditioned_krylov_bases, dim<2>{num_rows, num_rhs},
                span{0, local_num_rows * (restart_iter + 1)}, span{0, num_rhs});
        // after_preconditioner = preconditioned_krylov_bases * y
        exec->run(gmres::make_multi_axpy(
            gko::detail::get_local(preconditioned_krylov_bases_small.get()), y,
            gko::detail::get_local(after_preconditioner),
            final_iter_nums.get_const_data(), stop_status.get_data()));
    } else {
        auto krylov_bases_small = ::gko::detail::create_submatrix_helper(
            krylov_bases, dim<2>{num_rows, num_rhs},
            span{0, local_num_rows * (restart_iter + 1)}, span{0, num_rhs});
        // before_preconditioner = krylov_bases * y
        exec->run(gmres::make_multi_axpy(
            gko::detail::get_local(krylov_bases_small.get()), y,
            gko::detail::get_local(before_preconditioner),
            final_iter_nums.get_const_data(), stop_status.get_data()));

        // after_preconditioner = get_preconditioner() * before_preconditioner
        this->get_preconditioner()->apply(before_preconditioner,
                                          after_preconditioner);
    }
    // x = x + after_preconditioner
    dense_x->add_scaled(one_op, after_preconditioner);
}


template <typename ValueType>
void Gmres<ValueType>::apply_impl(const LinOp* alpha, const LinOp* b,
                                  const LinOp* beta, LinOp* x) const
{
    if (!this->get_system_matrix()) {
        return;
    }
    experimental::precision_dispatch_real_complex_distributed<ValueType>(
        [this](auto dense_alpha, auto dense_b, auto dense_beta, auto dense_x) {
            auto x_clone = dense_x->clone();
            this->apply_dense_impl(dense_b, x_clone.get());
            dense_x->scale(dense_beta);
            dense_x->add_scaled(dense_alpha, x_clone);
        },
        alpha, b, beta, x);
}


template <typename ValueType>
int workspace_traits<Gmres<ValueType>>::num_arrays(const Solver&)
{
    return 3;
}


template <typename ValueType>
int workspace_traits<Gmres<ValueType>>::num_vectors(const Solver&)
{
    return 15;
}


template <typename ValueType>
std::vector<std::string> workspace_traits<Gmres<ValueType>>::op_names(
    const Solver&)
{
    return {"residual",
            "preconditioned_vector",
            "krylov_bases",
            "hessenberg",
            "givens_sin",
            "givens_cos",
            "residual_norm_collection",
            "residual_norm",
            "y",
            "before_preconditioner",
            "after_preconditioner",
            "one",
            "minus_one",
            "next_krylov_norm_tmp",
            "preconditioned_krylov_bases"};
}


template <typename ValueType>
std::vector<std::string> workspace_traits<Gmres<ValueType>>::array_names(
    const Solver&)
{
    return {"stop", "tmp", "final_iter_nums"};
}


template <typename ValueType>
std::vector<int> workspace_traits<Gmres<ValueType>>::scalars(const Solver&)
{
    return {hessenberg,          givens_sin,
            givens_cos,          residual_norm_collection,
            residual_norm,       y,
            next_krylov_norm_tmp};
}


template <typename ValueType>
std::vector<int> workspace_traits<Gmres<ValueType>>::vectors(const Solver&)
{
    return {residual,
            preconditioned_vector,
            krylov_bases,
            before_preconditioner,
            after_preconditioner,
            preconditioned_krylov_bases};
}


#define GKO_DECLARE_GMRES(_type) class Gmres<_type>
#define GKO_DECLARE_GMRES_TRAITS(_type) struct workspace_traits<Gmres<_type>>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_GMRES);
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_GMRES_TRAITS);


}  // namespace solver
}  // namespace gko
