// SPDX-FileCopyrightText: 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "ginkgo/core/solver/nullspace.hpp"

#include <limits>
#include <string>
#include <type_traits>
#include <typeinfo>
#include <utility>

#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/name_demangling.hpp>
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
std::unique_ptr<matrix::MultiVector<ValueType>> column_view(
    matrix::MultiVector<ValueType>* v, size_type col)
{
    return v->create_submatrix(span{0, v->get_size()[0]}, span{col, col + 1});
}


template <typename ValueType>
std::unique_ptr<matrix::MultiVector<ValueType>> create_like(
    const matrix::MultiVector<ValueType>* like,
    std::shared_ptr<const Executor> exec, size_type num_cols)
{
    return matrix::MultiVector<ValueType>::create(
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
void sum_over_ranks(const AbstractMultiVector* v,
                    matrix::MultiVector<ValueType>* values,
                    const gko::detail::DenseCache<ValueType>& host_buffer)
{
#if GINKGO_BUILD_MPI
    if (auto distributed =
            dynamic_cast<const experimental::distributed::DistributedBase*>(
                v)) {
        experimental::distributed::detail::all_reduce_sum(
            distributed->get_communicator(), values, host_buffer);
    }
#endif
}


template <typename ValueType>
matrix::view::dense<const ValueType> basis_view(
    ptr_param<const AbstractMultiVector> basis)
{
    return basis ? basis->template get_const_local_device_view<ValueType>()
                 : matrix::view::dense<const ValueType>{
                       dim<2>{basis->get_local_size()[0], 0}, 0, nullptr};
}
//
//
// std::string type_name_of(const LinOp* obj)
// {
//     return name_demangling::get_type_name(typeid(*obj));
// }


}  // anonymous namespace


template <typename ValueType>
std::unique_ptr<Nullspace<ValueType>> Nullspace<ValueType>::create(
    std::shared_ptr<const Executor> exec,
    std::vector<std::shared_ptr<const AbstractMultiVector>> basis,
    bool contains_constant)
{
    if (basis.empty()) {
        GKO_INVALID_STATE(
            "Nullspace::create needs a non-empty basis, use "
            "create_from_constant for a constant-only nullspace");
    }
    for (const auto& vec : basis) {
        if (!vec) {
            GKO_INVALID_STATE("Nullspace::create got a null basis entry");
        }
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
bool Nullspace<ValueType>::contains_constant() const noexcept
{
    return contains_constant_;
}


template <typename ValueType>
size_type Nullspace<ValueType>::get_dimension() const noexcept
{
    return (basis_ ? basis_->get_size()[1] : 0) + (contains_constant_ ? 1 : 0);
}


template <typename ValueType>
std::shared_ptr<const AbstractMultiVector> Nullspace<ValueType>::get_basis()
    const noexcept
{
    return basis_;
}


template <typename ValueType>
Nullspace<ValueType>::Nullspace(std::shared_ptr<const Executor> exec)
    : LinOp(exec), contains_constant_{false}, reduction_tmp_{exec}
{}


template <typename ValueType>
Nullspace<ValueType>::Nullspace(
    std::shared_ptr<const Executor> exec, dim<2> size,
    std::vector<std::shared_ptr<const AbstractMultiVector>> basis,
    bool contains_constant)
    : LinOp(exec, size),
      contains_constant_{contains_constant},
      reduction_tmp_{exec}
{
    GKO_ASSERT_IS_SQUARE_MATRIX(size);
    if (!basis.empty()) {
        this->setup_basis(basis);
    }
}


template <typename ValueType>
void Nullspace<ValueType>::setup_basis(
    const std::vector<std::shared_ptr<const AbstractMultiVector>>& basis)
{
    auto exec = this->get_executor();
    const auto first = basis[0].get();
    size_type num_cols = 0;
    for (const auto& vec : basis) {
        GKO_ASSERT_EQUAL_ROWS(vec, this);
        num_cols += vec->get_size()[1];
    }

    // gather all columns into a single block on this executor
    auto block = AbstractMultiVector::create_with_type_of(
        first, exec, {first->get_size()[0], num_cols},
        {first->get_local_size()[0], num_cols});
    const auto local_rows = block->get_local_size()[0];
    size_type col = 0;
    for (const auto& vec : basis) {
        const auto vec_cols = vec->get_size()[1];
        GKO_ASSERT_EQUAL_ROWS(vec, block);
        auto target = block->create_subview(local_span{0, local_rows},
                                            local_span{col, col + vec_cols});
        target->copy_from(vec);
        col += vec_cols;
    }

    // Twice-iterated classical Gram-Schmidt against the constant and the
    // accepted columns, which are kept at the front of the block. A column is
    // dropped as linearly dependent if orthogonalization shrinks it below
    // sqrt(eps) of its original norm, which makes the decision independent of
    // its scale.
    auto orig_norms =
        matrix::MultiVector<absolute_type>::create(exec, dim<2>{1, num_cols});
    block->compute_norm2(orig_norms);
    auto host_orig_norms = gko::clone(exec->get_master(), orig_norms);
    const auto drop_tolerance =
        sqrt(std::numeric_limits<absolute_type>::epsilon());
    auto norm = matrix::MultiVector<absolute_type>::create(exec, dim<2>{1, 1});
    size_type num_kept = 0;
    for (size_type j = 0; j < num_cols; ++j) {
        auto col_j = block->create_subview(local_span{0, local_rows}, j);
        auto kept = block->create_subview(local_span{0, local_rows},
                                          local_span{0, num_kept});
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
                auto target =
                    block->create_subview(local_span{0, local_rows}, num_kept);
                target->copy_from(col_j.get());
            }
            ++num_kept;
        }
    }
    if (num_kept == 0) {
        return;
    }
    auto result = AbstractMultiVector::create_with_type_of(
        first, exec, {first->get_size()[0], num_kept},
        {first->get_local_size()[0], num_kept});
    auto kept = block->create_subview(local_span{0, local_rows},
                                      local_span{0, num_kept});
    result->copy_from(kept.get());
    basis_ = std::move(result);
}


template <typename ValueType>
void Nullspace<ValueType>::project(ptr_param<AbstractMultiVector> v) const
{
    // v is passed as input and output to convert it like apply does
    // a constant-only nullspace without size projects any size
    if (basis_ || this->get_size()[0] > 0) {
        GKO_ASSERT_EQUAL_ROWS(v, this);
    }
    this->remove_components(v.get(), v.get(), contains_constant_);
}


template <typename ValueType>
void Nullspace<ValueType>::compute_components(const AbstractMultiVector* v,
                                              const AbstractMultiVector* basis,
                                              bool has_constant) const
{
    auto exec = this->get_executor();
    const size_type num_const = has_constant ? 1 : 0;
    const auto num_basis = basis ? basis->get_size()[1] : 0;
    if (basis) {
        GKO_ASSERT_EQUAL_ROWS(v, basis);
    }
    const auto inv_size =
        static_cast<absolute_type>(1.0 / static_cast<double>(v->get_size()[0]));
    // coefficients = [ mean(v) ; V^H v ] in one pass over v, then combined
    // over all ranks with a single all-reduce
    coefficients_.init(exec, dim<2>{num_const + num_basis, v->get_size()[1]});
    exec->run(nullspace::make_compute_coefficients(
        v->template get_const_local_device_view<ValueType>(),
        basis_view<ValueType>(basis), has_constant, inv_size,
        coefficients_->get_device_view(), reduction_tmp_));
    sum_over_ranks(v, coefficients_.get(), host_coefficients_);
}


template <typename ValueType>
void Nullspace<ValueType>::subtract_components(const AbstractMultiVector* v,
                                               AbstractMultiVector* result,
                                               const AbstractMultiVector* basis,
                                               bool has_constant) const
{
    // result = v - mean - V C in one pass; the basis is orthogonal to the
    // constant, so both components can be removed at once
    this->get_executor()->run(nullspace::make_subtract_projection(
        basis_view<ValueType>(basis), has_constant,
        coefficients_->get_const_device_view(),
        v->template get_const_local_device_view<ValueType>(),
        result->template get_local_device_view<ValueType>()));
}


template <typename ValueType>
void Nullspace<ValueType>::remove_components(AbstractMultiVector* v,
                                             const AbstractMultiVector* basis,
                                             bool has_constant) const
{
    if ((!has_constant && !basis) || v->get_size()[1] == 0) {
        return;
    }
    auto v_exec = make_temporary_clone(this->get_executor(), v);
    this->compute_components(v_exec.get(), basis, has_constant);
    this->subtract_components(v_exec.get(), v_exec.get(), basis, has_constant);
}


template <typename ValueType>
void Nullspace<ValueType>::apply_impl(const AbstractMultiVector* b,
                                      AbstractMultiVector* x) const
{
    // x = b - V C with the components C of b, without copying b first
    if (!contains_constant_ && !basis_) {
        x->copy_from(b);
        return;
    }
    this->compute_components(b, basis_.get(), contains_constant_);
    this->subtract_components(b, x, basis_.get(), contains_constant_);
}


template <typename ValueType>
void Nullspace<ValueType>::apply_impl(const AbstractMultiVector* alpha,
                                      const AbstractMultiVector* b,
                                      const AbstractMultiVector* beta,
                                      AbstractMultiVector* x) const
{
    // x = beta x + alpha (b - V C) with the components C of b
    const auto has_components = contains_constant_ || basis_;
    if (has_components) {
        this->compute_components(b, basis_.get(), contains_constant_);
        coefficients_->scale(alpha);
    }
    x->scale(beta);
    x->add_scaled(alpha, b);
    if (has_components) {
        this->subtract_components(x, x, basis_.get(), contains_constant_);
    }
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


namespace detail {
namespace {


template <typename... Types>
bool is_constant_only(const LinOp* param)
{
    return (... || [param] {
        auto ns = dynamic_cast<const Nullspace<Types>*>(param);
        return ns && ns->contains_constant() && !ns->get_basis();
    }());
}


// whether `param` is a constant-only Nullspace of any value type
template <typename ValueType>
bool is_constant_only_nullspace(const LinOp* param)
{
    using real = remove_complex<ValueType>;
    using real1 = next_precision<real, 1>;
    using real2 = next_precision<real, 2>;
    using real3 = next_precision<real, 3>;
    return is_constant_only<real, real1, real2, real3, to_complex<real>,
                            to_complex<real1>, to_complex<real2>,
                            to_complex<real3>>(param);
}


}  // anonymous namespace


template <typename ValueType>
std::shared_ptr<const Nullspace<ValueType>> prepare_nullspace(
    std::shared_ptr<const LinOp> param, const char* name, const LinOp* op,
    size_type size, std::shared_ptr<const Executor> exec)
{
    if (!param) {
        return nullptr;
    }
    if (is_constant_only_nullspace<ValueType>(param.get())) {
        // the constant does not depend on the size or value type
        return Nullspace<ValueType>::create_from_constant(exec,
                                                          dim<2>{size, size});
    }
    auto ns = std::dynamic_pointer_cast<const Nullspace<ValueType>>(param);
    if (!ns) {
        throw InvalidStateError(
            __FILE__, __LINE__, __func__,
            std::string{"the "} + name + " parameter must be a " +
                name_demangling::get_type_name(typeid(Nullspace<ValueType>)) +
                " matching the value type of the operator, or a "
                "constant-only Nullspace, got " +
                name_demangling::get_type_name(typeid(param.get())));
    }
    if (!ns->get_basis()) {
        // all basis vectors were dropped, there is nothing to remove
        return nullptr;
    }
    if (ns->get_size()[0] != size) {
        throw DimensionMismatch(__FILE__, __LINE__, __func__, name,
                                ns->get_size()[0], ns->get_size()[1],
                                "system_matrix", size, size,
                                "expected a nullspace of matching size");
    }
    if (dynamic_cast<const experimental::distributed::DistributedBase*>(op) &&
        !dynamic_cast<const experimental::distributed::DistributedBase*>(
            ns->get_basis().get())) {
        throw InvalidStateError(
            __FILE__, __LINE__, __func__,
            std::string{"the "} + name +
                " of a distributed operator needs a distributed basis, got " +
                name_demangling::get_type_name(typeid(ns->get_basis().get())));
    }
    // a separate copy keeps the scratch buffers of operators generated from
    // the same parameter apart
    return gko::clone(exec, ns);
}


#define GKO_DECLARE_PREPARE_NULLSPACE(ValueType)                               \
    std::shared_ptr<const Nullspace<ValueType>> prepare_nullspace<ValueType>(  \
        std::shared_ptr<const LinOp> param, const char* name, const LinOp* op, \
        size_type size, std::shared_ptr<const Executor> exec)
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_PREPARE_NULLSPACE);


}  // namespace detail


#define GKO_DECLARE_NULLSPACE(ValueType) class Nullspace<ValueType>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_NULLSPACE);


}  // namespace solver
}  // namespace gko
