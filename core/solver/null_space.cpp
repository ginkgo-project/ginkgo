// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "ginkgo/core/solver/null_space.hpp"

#include <limits>
#include <type_traits>
#include <utility>

#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/precision_dispatch.hpp>
#include <ginkgo/core/base/temporary_clone.hpp>

#include "core/distributed/helpers.hpp"
#include "core/solver/null_space_kernels.hpp"

#if GINKGO_BUILD_MPI
#include <ginkgo/core/base/mpi.hpp>

#include "core/mpi/mpi_op.hpp"
#endif


namespace gko {
namespace null_space {
namespace {


GKO_REGISTER_OPERATION(compute_scaled_column_sums,
                       null_space::compute_scaled_column_sums);
GKO_REGISTER_OPERATION(remove_constant, null_space::remove_constant);


}  // anonymous namespace
}  // namespace null_space


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


// Sums `values` over all ranks if `v` is a distributed vector, otherwise does
// nothing. `values` must be contiguous (stride == number of columns).
template <typename ValueType>
void all_reduce_sum(const LinOp* v, matrix::Dense<ValueType>* values,
                    const detail::DenseCache<ValueType>& host_buffer)
{
#if GINKGO_BUILD_MPI
    auto distributed =
        dynamic_cast<const experimental::distributed::DistributedBase*>(v);
    if (distributed == nullptr) {
        return;
    }
    auto exec = values->get_executor();
    const auto comm = distributed->get_communicator();
    const auto count = static_cast<int>(values->get_num_stored_elements());
    auto sum_op = experimental::mpi::sum<ValueType>();
    exec->synchronize();
    if (experimental::mpi::requires_host_buffer(exec, comm)) {
        host_buffer.init(exec->get_master(), values->get_size());
        host_buffer->copy_from(values);
        comm.all_reduce(exec->get_master(), host_buffer->get_values(), count,
                        sum_op.get_op());
        values->copy_from(host_buffer.get());
    } else {
        comm.all_reduce(exec, values->get_values(), count, sum_op.get_op());
    }
#endif
}


}  // anonymous namespace


template <typename ValueType>
std::unique_ptr<NullSpace<ValueType>> NullSpace<ValueType>::create(
    std::shared_ptr<const Executor> exec,
    std::vector<std::shared_ptr<const LinOp>> basis, bool contains_constant)
{
    if (basis.empty()) {
        GKO_INVALID_STATE(
            "NullSpace::create needs a non-empty basis, use "
            "create_from_constant for a constant-only nullspace");
    }
    const auto n = basis[0]->get_size()[0];
    return std::unique_ptr<NullSpace>(new NullSpace(
        std::move(exec), dim<2>{n, n}, std::move(basis), contains_constant));
}


template <typename ValueType>
std::unique_ptr<NullSpace<ValueType>>
NullSpace<ValueType>::create_from_constant(std::shared_ptr<const Executor> exec,
                                           dim<2> size)
{
    return std::unique_ptr<NullSpace>(
        new NullSpace(std::move(exec), size, {}, true));
}


template <typename ValueType>
NullSpace<ValueType>::NullSpace(std::shared_ptr<const Executor> exec)
    : LinOp(exec),
      contains_constant_{false},
      one_{initialize<matrix::Dense<ValueType>>({one<ValueType>()}, exec)},
      neg_one_{initialize<matrix::Dense<ValueType>>({-one<ValueType>()}, exec)},
      reduction_tmp_{exec}
{}


template <typename ValueType>
NullSpace<ValueType>::NullSpace(std::shared_ptr<const Executor> exec,
                                dim<2> size,
                                std::vector<std::shared_ptr<const LinOp>> basis,
                                bool contains_constant)
    : NullSpace(std::move(exec))
{
    GKO_ASSERT_IS_SQUARE_MATRIX(size);
    this->set_size(size);
    contains_constant_ = contains_constant;
    if (!basis.empty()) {
        detail::vector_dispatch<ValueType>(basis[0].get(), [&](auto first) {
            using vector_type =
                std::remove_const_t<std::remove_pointer_t<decltype(first)>>;
            this->template setup_basis<vector_type>(basis);
        });
    }
}


template <typename ValueType>
template <typename VectorType>
void NullSpace<ValueType>::setup_basis(
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
    auto block_local = detail::get_local(block.get());
    const auto local_rows = block_local->get_size()[0];
    size_type col = 0;
    for (const auto& vec : basis) {
        auto vec_local = detail::get_local(as<VectorType>(vec.get()));
        const auto vec_cols = vec_local->get_size()[1];
        GKO_ASSERT_EQUAL_ROWS(vec_local, block_local);
        auto target = block_local->create_submatrix(span{0, local_rows},
                                                    span{col, col + vec_cols});
        target->copy_from(vec_local);
        col += vec_cols;
    }

    // Twice-iterated modified Gram-Schmidt. A column is dropped as linearly
    // dependent if orthogonalization shrinks it below sqrt(eps) of its
    // original norm, which makes the decision independent of its scale.
    auto orig_norms =
        matrix::Dense<absolute_type>::create(exec, dim<2>{1, num_cols});
    block->compute_norm2(orig_norms);
    auto host_orig_norms = gko::clone(exec->get_master(), orig_norms);
    if (contains_constant_) {
        this->remove_constant_impl(block.get());
    }
    const auto drop_tolerance =
        sqrt(std::numeric_limits<absolute_type>::epsilon());
    auto coeff = matrix::Dense<ValueType>::create(exec, dim<2>{1, 1});
    auto norm = matrix::Dense<absolute_type>::create(exec, dim<2>{1, 1});
    array<size_type> kept{exec->get_master(), num_cols};
    size_type num_kept = 0;
    for (size_type j = 0; j < num_cols; ++j) {
        auto col_j = column_view(block.get(), j);
        for (int pass = 0; pass < 2; ++pass) {
            for (size_type i = 0; i < num_kept; ++i) {
                auto col_i = column_view(block.get(), kept.get_data()[i]);
                // coeff = col_i^H col_j
                col_j->compute_conj_dot(col_i.get(), coeff.get());
                col_j->sub_scaled(coeff.get(), col_i.get());
            }
            if (contains_constant_) {
                this->remove_constant_impl(col_j.get());
            }
        }
        col_j->compute_norm2(norm.get());
        const auto host_norm = exec->copy_val_to_host(norm->get_const_values());
        const auto orig_norm = host_orig_norms->at(0, j);
        if (orig_norm > zero<absolute_type>() &&
            host_norm > drop_tolerance * orig_norm) {
            col_j->inv_scale(norm.get());
            kept.get_data()[num_kept++] = j;
        }
    }
    if (num_kept == 0) {
        return;
    }

    auto result = create_like(first, exec, num_kept);
    auto result_local = detail::get_local(result.get());
    for (size_type i = 0; i < num_kept; ++i) {
        const auto j = kept.get_const_data()[i];
        auto source =
            block_local->create_submatrix(span{0, local_rows}, span{j, j + 1});
        auto target =
            result_local->create_submatrix(span{0, local_rows}, span{i, i + 1});
        target->copy_from(source.get());
    }
    basis_conj_trans_ =
        share(as<matrix::Dense<ValueType>>(result_local->conj_transpose()));
    basis_ = std::move(result);
}


template <typename ValueType>
template <typename VectorType>
void NullSpace<ValueType>::remove_constant_impl(VectorType* v) const
{
    auto exec = this->get_executor();
    auto v_local = detail::get_local(v);
    const auto inv_size = static_cast<absolute_type>(
        1.0 / static_cast<double>(this->get_size()[0]));
    auto mean = matrix::Dense<ValueType>::create(
        exec, dim<2>{1, v_local->get_size()[1]});
    exec->run(null_space::make_compute_scaled_column_sums(
        v_local->get_const_device_view(), inv_size, mean->get_device_view(),
        reduction_tmp_));
    all_reduce_sum(v, mean.get(), host_coefficients_);
    exec->run(null_space::make_remove_constant(mean->get_const_device_view(),
                                               v_local->get_device_view()));
}


template <typename ValueType>
void NullSpace<ValueType>::project(ptr_param<LinOp> v) const
{
    detail::vector_dispatch<ValueType>(v.get(), [&](auto vec) {
        if (vec == nullptr) {
            GKO_NOT_SUPPORTED(v.get());
        }
        this->project_impl(vec);
    });
}


template <typename ValueType>
template <typename VectorType>
void NullSpace<ValueType>::project_impl(VectorType* v) const
{
    GKO_ASSERT_EQUAL_ROWS(v, this);
    const size_type num_const = contains_constant_ ? 1 : 0;
    const auto num_basis = this->get_num_basis_vectors();
    const auto num_rhs = v->get_size()[1];
    if (num_const + num_basis == 0 || num_rhs == 0) {
        return;
    }
    const matrix::Dense<ValueType>* basis_local = nullptr;
    if (num_basis > 0) {
        auto basis = dynamic_cast<const VectorType*>(basis_.get());
        if (basis == nullptr) {
            // e.g. a non-distributed basis applied to a distributed vector
            GKO_NOT_SUPPORTED(v);
        }
        basis_local = detail::get_local(basis);
    }
    auto exec = this->get_executor();
    auto v_exec = make_temporary_clone(exec, v);
    auto v_local = detail::get_local(v_exec.get());
    if (basis_local) {
        GKO_ASSERT_EQUAL_ROWS(v_local, basis_local);
    }

    // coefficients = [ mean(v) ; V^H v ], reduced in a single all-reduce
    coefficients_.init(exec, dim<2>{num_const + num_basis, num_rhs});
    std::unique_ptr<matrix::Dense<ValueType>> mean;
    std::unique_ptr<matrix::Dense<ValueType>> coeffs;
    if (num_const > 0) {
        mean = coefficients_->create_submatrix(span{0, 1}, span{0, num_rhs});
        const auto inv_size = static_cast<absolute_type>(
            1.0 / static_cast<double>(this->get_size()[0]));
        exec->run(null_space::make_compute_scaled_column_sums(
            v_local->get_const_device_view(), inv_size, mean->get_device_view(),
            reduction_tmp_));
    }
    if (num_basis > 0) {
        coeffs = coefficients_->create_submatrix(
            span{num_const, num_const + num_basis}, span{0, num_rhs});
        basis_conj_trans_->apply(v_local, coeffs.get());
    }
    all_reduce_sum(v_exec.get(), coefficients_.get(), host_coefficients_);
    // The basis is orthogonal to the constant, so both parts can be removed
    // independently: v -= mean(v), v -= V (V^H v)
    if (mean) {
        exec->run(null_space::make_remove_constant(
            mean->get_const_device_view(), v_local->get_device_view()));
    }
    if (coeffs) {
        basis_local->apply(neg_one_, coeffs.get(), one_, v_local);
    }
}


template <typename ValueType>
void NullSpace<ValueType>::apply_impl(const LinOp* b, LinOp* x) const
{
    experimental::precision_dispatch_real_complex_distributed<ValueType>(
        [this](auto dense_b, auto dense_x) {
            dense_x->copy_from(dense_b);
            this->project(dense_x);
        },
        b, x);
}


template <typename ValueType>
void NullSpace<ValueType>::apply_impl(const LinOp* alpha, const LinOp* b,
                                      const LinOp* beta, LinOp* x) const
{
    experimental::precision_dispatch_real_complex_distributed<ValueType>(
        [this](auto dense_alpha, auto dense_b, auto dense_beta, auto dense_x) {
            auto projected = dense_b->clone();
            this->project(projected);
            dense_x->scale(dense_beta);
            dense_x->add_scaled(dense_alpha, projected);
        },
        alpha, b, beta, x);
}


template <typename ValueType>
NullSpace<ValueType>& NullSpace<ValueType>::operator=(const NullSpace& other)
{
    if (&other != this) {
        LinOp::operator=(other);
        auto exec = this->get_executor();
        contains_constant_ = other.contains_constant_;
        basis_ = other.basis_;
        basis_conj_trans_ = other.basis_conj_trans_;
        if (basis_ && other.get_executor() != exec) {
            basis_ = gko::clone(exec, basis_);
            basis_conj_trans_ = gko::clone(exec, basis_conj_trans_);
        }
    }
    return *this;
}


template <typename ValueType>
NullSpace<ValueType>& NullSpace<ValueType>::operator=(NullSpace&& other)
{
    if (&other != this) {
        LinOp::operator=(std::move(other));
        auto exec = this->get_executor();
        contains_constant_ = std::exchange(other.contains_constant_, false);
        basis_ = std::move(other.basis_);
        basis_conj_trans_ = std::move(other.basis_conj_trans_);
        if (basis_ && other.get_executor() != exec) {
            basis_ = gko::clone(exec, basis_);
            basis_conj_trans_ = gko::clone(exec, basis_conj_trans_);
        }
    }
    return *this;
}


template <typename ValueType>
NullSpace<ValueType>::NullSpace(const NullSpace& other)
    : NullSpace(other.get_executor())
{
    *this = other;
}


template <typename ValueType>
NullSpace<ValueType>::NullSpace(NullSpace&& other)
    : NullSpace(other.get_executor())
{
    *this = std::move(other);
}


#define GKO_DECLARE_NULL_SPACE(ValueType) class NullSpace<ValueType>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_NULL_SPACE);


}  // namespace gko
