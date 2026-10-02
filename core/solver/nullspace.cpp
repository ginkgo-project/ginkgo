// SPDX-FileCopyrightText: 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "ginkgo/core/solver/nullspace.hpp"

#include <limits>
#include <type_traits>
#include <utility>

#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/precision_dispatch.hpp>
#include <ginkgo/core/base/temporary_clone.hpp>

#include "core/distributed/helpers.hpp"
#include "core/solver/nullspace_kernels.hpp"


namespace gko {
namespace solver {
namespace nullspace {
namespace {


GKO_REGISTER_OPERATION(compute_coefficients, nullspace::compute_coefficients);
GKO_REGISTER_OPERATION(subtract_projection, nullspace::subtract_projection);


}  // anonymous namespace
}  // namespace nullspace


namespace {


template <typename ValueType>
std::unique_ptr<matrix::Dense<ValueType>> column_view(
    matrix::Dense<ValueType>* v, size_type col)
{
    return v->create_submatrix(span{0, v->get_size()[0]}, span{col, col + 1});
}


template <typename ValueType>
std::unique_ptr<matrix::Dense<ValueType>> create_like(
    const matrix::Dense<ValueType>* like, std::shared_ptr<const Executor> exec,
    size_type num_cols)
{
    return matrix::Dense<ValueType>::create(
        std::move(exec), dim<2>{like->get_size()[0], num_cols});
}


#if GINKGO_BUILD_MPI


template <typename ValueType>
std::unique_ptr<experimental::distributed::Vector<ValueType>> column_view(
    experimental::distributed::Vector<ValueType>* v, size_type col)
{
    const auto local_rows = v->get_local_vector()->get_size()[0];
    return v->create_submatrix(local_span{0, local_rows},
                               local_span{col, col + 1},
                               dim<2>{v->get_size()[0], 1});
}


template <typename ValueType>
std::unique_ptr<experimental::distributed::Vector<ValueType>> create_like(
    const experimental::distributed::Vector<ValueType>* like,
    std::shared_ptr<const Executor> exec, size_type num_cols)
{
    return experimental::distributed::Vector<ValueType>::create(
        std::move(exec), like->get_communicator(),
        dim<2>{like->get_size()[0], num_cols},
        dim<2>{like->get_local_vector()->get_size()[0], num_cols});
}


#endif


// sums `values` over all ranks if `v` is a distributed vector
template <typename ValueType>
void sum_over_ranks(const LinOp* v, matrix::Dense<ValueType>* values,
                    const gko::detail::DenseCache<ValueType>& host_buffer)
{
#if GINKGO_BUILD_MPI
    if (auto distributed =
            dynamic_cast<const experimental::distributed::DistributedBase*>(
                v)) {
        gko::detail::all_reduce_sum(distributed->get_communicator(), values,
                                    host_buffer);
    }
#endif
}


template <typename ValueType>
matrix::view::dense<const ValueType> basis_view(
    const matrix::Dense<ValueType>* basis, size_type num_rows)
{
    return basis ? basis->get_const_device_view()
                 : matrix::view::dense<const ValueType>{dim<2>{num_rows, 0}, 0,
                                                        nullptr};
}


}  // anonymous namespace


template <typename ValueType>
std::unique_ptr<Nullspace<ValueType>> Nullspace<ValueType>::create(
    std::shared_ptr<const Executor> exec,
    std::vector<std::shared_ptr<const LinOp>> basis, bool contains_constant)
{
    if (basis.empty()) {
        GKO_INVALID_STATE(
            "Nullspace::create needs a non-empty basis, use "
            "create_from_constant for a constant-only nullspace");
    }
    const auto n = basis[0]->get_size()[0];
    return std::unique_ptr<Nullspace>(new Nullspace(
        std::move(exec), dim<2>{n, n}, std::move(basis), contains_constant));
}


template <typename ValueType>
std::unique_ptr<Nullspace<ValueType>>
Nullspace<ValueType>::create_from_constant(std::shared_ptr<const Executor> exec,
                                           dim<2> size)
{
    return std::unique_ptr<Nullspace>(
        new Nullspace(std::move(exec), size, {}, true));
}


template <typename ValueType>
Nullspace<ValueType>::Nullspace(std::shared_ptr<const Executor> exec)
    : LinOp(exec), contains_constant_{false}, reduction_tmp_{exec}
{}


template <typename ValueType>
Nullspace<ValueType>::Nullspace(std::shared_ptr<const Executor> exec,
                                dim<2> size,
                                std::vector<std::shared_ptr<const LinOp>> basis,
                                bool contains_constant)
    : Nullspace(std::move(exec))
{
    GKO_ASSERT_IS_SQUARE_MATRIX(size);
    this->set_size(size);
    contains_constant_ = contains_constant;
    if (!basis.empty()) {
        gko::detail::vector_dispatch<ValueType>(
            basis[0].get(), [&](auto first) {
                using vector_type =
                    std::remove_const_t<std::remove_pointer_t<decltype(first)>>;
                this->template setup_basis<vector_type>(basis);
            });
    }
}


template <typename ValueType>
template <typename VectorType>
void Nullspace<ValueType>::setup_basis(
    const std::vector<std::shared_ptr<const LinOp>>& basis)
{
    auto exec = this->get_executor();
    const auto first = as<VectorType>(basis[0].get());
    size_type num_cols = 0;
    for (const auto& vec : basis) {
        GKO_ASSERT_EQUAL_ROWS(vec, this);
        num_cols += vec->get_size()[1];
    }

    // gather all columns into a single block on this executor
    auto block = create_like(first, exec, num_cols);
    auto block_local = gko::detail::get_local(block.get());
    const auto local_rows = block_local->get_size()[0];
    size_type col = 0;
    for (const auto& vec : basis) {
        auto vec_local = gko::detail::get_local(as<VectorType>(vec.get()));
        const auto vec_cols = vec_local->get_size()[1];
        GKO_ASSERT_EQUAL_ROWS(vec_local, block_local);
        auto target = block_local->create_submatrix(span{0, local_rows},
                                                    span{col, col + vec_cols});
        target->copy_from(vec_local);
        col += vec_cols;
    }

    // Twice-iterated classical Gram-Schmidt against the constant and the
    // accepted columns, which are kept at the front of the block. A column is
    // dropped as linearly dependent if orthogonalization shrinks it below
    // sqrt(eps) of its original norm, which makes the decision independent of
    // its scale.
    auto orig_norms =
        matrix::Dense<absolute_type>::create(exec, dim<2>{1, num_cols});
    block->compute_norm2(orig_norms);
    auto host_orig_norms = gko::clone(exec->get_master(), orig_norms);
    const auto drop_tolerance =
        sqrt(std::numeric_limits<absolute_type>::epsilon());
    auto norm = matrix::Dense<absolute_type>::create(exec, dim<2>{1, 1});
    size_type num_kept = 0;
    for (size_type j = 0; j < num_cols; ++j) {
        auto col_j = column_view(block.get(), j);
        auto kept = block_local->create_submatrix(span{0, local_rows},
                                                  span{0, num_kept});
        const auto kept_basis = num_kept > 0 ? kept.get() : nullptr;
        for (int pass = 0; pass < 2; ++pass) {
            this->remove_components(col_j.get(), kept_basis,
                                    contains_constant_);
        }
        col_j->compute_norm2(norm);
        const auto host_norm = exec->copy_val_to_host(norm->get_const_values());
        const auto orig_norm = host_orig_norms->at(0, j);
        if (orig_norm > zero<absolute_type>() &&
            host_norm > drop_tolerance * orig_norm) {
            col_j->inv_scale(norm);
            if (j != num_kept) {
                auto target = block_local->create_submatrix(
                    span{0, local_rows}, span{num_kept, num_kept + 1});
                target->copy_from(gko::detail::get_local(col_j.get()));
            }
            ++num_kept;
        }
    }
    if (num_kept == 0) {
        return;
    }
    auto result = create_like(first, exec, num_kept);
    auto kept =
        block_local->create_submatrix(span{0, local_rows}, span{0, num_kept});
    gko::detail::get_local(result.get())->copy_from(kept.get());
    basis_ = std::move(result);
}


template <typename ValueType>
void Nullspace<ValueType>::project(ptr_param<LinOp> v) const
{
    gko::detail::vector_dispatch<ValueType>(v.get(), [&](auto vec) {
        if (vec == nullptr) {
            GKO_NOT_SUPPORTED(v.get());
        }
        GKO_ASSERT_EQUAL_ROWS(vec, this);
        this->remove_components(vec, this->get_local_basis(vec),
                                contains_constant_);
    });
}


template <typename ValueType>
template <typename VectorType>
const matrix::Dense<ValueType>* Nullspace<ValueType>::get_local_basis(
    const VectorType* v) const
{
    if (!basis_) {
        return nullptr;
    }
    auto basis = dynamic_cast<const VectorType*>(basis_.get());
    if (basis == nullptr) {
        // e.g. a non-distributed basis applied to a distributed vector
        GKO_NOT_SUPPORTED(v);
    }
    return gko::detail::get_local(basis);
}


template <typename ValueType>
template <typename VectorType>
void Nullspace<ValueType>::compute_components(
    const VectorType* v, const matrix::Dense<ValueType>* basis_local,
    bool has_constant) const
{
    auto exec = this->get_executor();
    auto v_local = gko::detail::get_local(v);
    const auto local_rows = v_local->get_size()[0];
    const size_type num_const = has_constant ? 1 : 0;
    const auto num_basis = basis_local ? basis_local->get_size()[1] : 0;
    if (basis_local) {
        GKO_ASSERT_EQUAL_ROWS(v_local, basis_local);
    }
    const auto inv_size = static_cast<absolute_type>(
        1.0 / static_cast<double>(this->get_size()[0]));
    // coefficients = [ mean(v) ; V^H v ] in one pass over v, then combined
    // over all ranks with a single all-reduce
    coefficients_.init(exec, dim<2>{num_const + num_basis, v->get_size()[1]});
    exec->run(nullspace::make_compute_coefficients(
        v_local->get_const_device_view(), basis_view(basis_local, local_rows),
        has_constant, inv_size, coefficients_->get_device_view(),
        reduction_tmp_));
    sum_over_ranks(v, coefficients_.get(), host_coefficients_);
}


template <typename ValueType>
template <typename VectorType>
void Nullspace<ValueType>::subtract_components(
    VectorType* v, const matrix::Dense<ValueType>* basis_local,
    bool has_constant) const
{
    auto v_local = gko::detail::get_local(v);
    // v -= mean + V C in one pass; the basis is orthogonal to the constant, so
    // both components can be removed at once
    this->get_executor()->run(nullspace::make_subtract_projection(
        basis_view(basis_local, v_local->get_size()[0]), has_constant,
        coefficients_->get_const_device_view(), v_local->get_device_view()));
}


template <typename ValueType>
template <typename VectorType>
void Nullspace<ValueType>::remove_components(
    VectorType* v, const matrix::Dense<ValueType>* basis_local,
    bool has_constant) const
{
    if ((!has_constant && !basis_local) || v->get_size()[1] == 0) {
        return;
    }
    auto v_exec = make_temporary_clone(this->get_executor(), v);
    this->compute_components(v_exec.get(), basis_local, has_constant);
    this->subtract_components(v_exec.get(), basis_local, has_constant);
}


template <typename ValueType>
void Nullspace<ValueType>::apply_impl(const LinOp* b, LinOp* x) const
{
    experimental::precision_dispatch_real_complex_distributed<ValueType>(
        [this](auto dense_b, auto dense_x) {
            dense_x->copy_from(dense_b);
            this->project(dense_x);
        },
        b, x);
}


template <typename ValueType>
void Nullspace<ValueType>::apply_impl(const LinOp* alpha, const LinOp* b,
                                      const LinOp* beta, LinOp* x) const
{
    experimental::precision_dispatch_real_complex_distributed<ValueType>(
        [this](auto dense_alpha, auto dense_b, auto dense_beta, auto dense_x) {
            // x = beta x + alpha (b - V C) with the components C of b
            const auto basis_local = this->get_local_basis(dense_b);
            const auto has_components = contains_constant_ || basis_local;
            if (has_components) {
                this->compute_components(dense_b, basis_local,
                                         contains_constant_);
                coefficients_->scale(dense_alpha);
            }
            dense_x->scale(dense_beta);
            dense_x->add_scaled(dense_alpha, dense_b);
            if (has_components) {
                this->subtract_components(dense_x, basis_local,
                                          contains_constant_);
            }
        },
        alpha, b, beta, x);
}


template <typename ValueType>
Nullspace<ValueType>& Nullspace<ValueType>::operator=(const Nullspace& other)
{
    if (&other != this) {
        LinOp::operator=(other);
        auto exec = this->get_executor();
        contains_constant_ = other.contains_constant_;
        basis_ = other.basis_;
        if (basis_ && other.get_executor() != exec) {
            basis_ = gko::clone(exec, basis_);
        }
    }
    return *this;
}


template <typename ValueType>
Nullspace<ValueType>& Nullspace<ValueType>::operator=(Nullspace&& other)
{
    if (&other != this) {
        LinOp::operator=(std::move(other));
        auto exec = this->get_executor();
        contains_constant_ = std::exchange(other.contains_constant_, false);
        basis_ = std::move(other.basis_);
        if (basis_ && other.get_executor() != exec) {
            basis_ = gko::clone(exec, basis_);
        }
    }
    return *this;
}


template <typename ValueType>
Nullspace<ValueType>::Nullspace(const Nullspace& other)
    : Nullspace(other.get_executor())
{
    *this = other;
}


template <typename ValueType>
Nullspace<ValueType>::Nullspace(Nullspace&& other)
    : Nullspace(other.get_executor())
{
    *this = std::move(other);
}


#define GKO_DECLARE_NULLSPACE(ValueType) class Nullspace<ValueType>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_NULLSPACE);


}  // namespace solver
}  // namespace gko
