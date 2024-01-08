// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/matrix/permutation.hpp>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/precision_dispatch.hpp>
#include <ginkgo/core/base/temporary_clone.hpp>
#include <ginkgo/core/base/utils_helper.hpp>


#include "core/base/dispatch_helper.hpp"
#include "core/matrix/permutation_kernels.hpp"


namespace gko {
namespace matrix {
namespace permutation {


GKO_REGISTER_OPERATION(invert, permutation::invert);
GKO_REGISTER_OPERATION(compose, permutation::compose);


}  // namespace permutation


void validate_permute_dimensions(dim<2> size, dim<2> permutation_size,
                                 permute_mode mode)
{
    if ((mode & permute_mode::symmetric) == permute_mode::symmetric) {
        GKO_ASSERT_IS_SQUARE_MATRIX(size);
    }
    if ((mode & permute_mode::rows) == permute_mode::rows) {
        if (size[0] != permutation_size[0]) {
            throw DimensionMismatch(
                __FILE__, __LINE__, __func__, "matrix", size[0], size[1],
                "permutation", permutation_size[0], permutation_size[0],
                "expected the permutation size to match the number of rows");
        };
    }
    if ((mode & permute_mode::columns) == permute_mode::columns) {
        if (size[1] != permutation_size[0]) {
            throw DimensionMismatch(
                __FILE__, __LINE__, __func__, "matrix", size[0], size[1],
                "permutation", permutation_size[0], permutation_size[0],
                "expected the permutation size to match the number of columns");
        };
    }
}


permute_mode operator|(permute_mode a, permute_mode b)
{
    return static_cast<permute_mode>(static_cast<unsigned>(a) |
                                     static_cast<unsigned>(b));
}


permute_mode operator&(permute_mode a, permute_mode b)
{
    return static_cast<permute_mode>(static_cast<unsigned>(a) &
                                     static_cast<unsigned>(b));
}


permute_mode operator^(permute_mode a, permute_mode b)
{
    return static_cast<permute_mode>(static_cast<unsigned>(a) ^
                                     static_cast<unsigned>(b));
}


std::ostream& operator<<(std::ostream& stream, permute_mode mode)
{
    switch (mode) {
    case permute_mode::none:
        return stream << "none";
    case permute_mode::rows:
        return stream << "rows";
    case permute_mode::columns:
        return stream << "columns";
    case permute_mode::symmetric:
        return stream << "symmetric";
    case permute_mode::inverse:
        return stream << "inverse";
    case permute_mode::inverse_rows:
        return stream << "inverse_rows";
    case permute_mode::inverse_columns:
        return stream << "inverse_columns";
    case permute_mode::inverse_symmetric:
        return stream << "inverse_symmetric";
    }
    return stream;
}


template <typename IndexType>
std::unique_ptr<const Permutation<IndexType>>
Permutation<IndexType>::create_const(
    std::shared_ptr<const Executor> exec, size_type size,
    gko::detail::const_array_view<IndexType>&& perm_idxs,
    mask_type enabled_permute)
{
    GKO_ASSERT_EQ(enabled_permute, row_permute);
    GKO_ASSERT_EQ(size, perm_idxs.get_size());
    return create_const(std::move(exec), std::move(perm_idxs));
}


template <typename IndexType>
std::unique_ptr<const Permutation<IndexType>>
Permutation<IndexType>::create_const(
    std::shared_ptr<const Executor> exec,
    gko::detail::const_array_view<IndexType>&& perm_idxs)
{
    // cast const-ness away, but return a const object afterwards,
    // so we can ensure that no modifications take place.
    return std::unique_ptr<const Permutation<IndexType>>(
        new Permutation<IndexType>{
            exec, gko::detail::array_const_cast(std::move(perm_idxs))});
}


template <typename IndexType>
Permutation<IndexType>::Permutation(std::shared_ptr<const Executor> exec,
                                    size_type size)
    : EnableLinOp<Permutation>(exec, dim<2>{size}), permutation_{exec, size}
{}


template <typename IndexType>
Permutation<IndexType>::Permutation(std::shared_ptr<const Executor> exec,
                                    array<index_type> permutation_indices)
    : EnableLinOp<Permutation>(exec, dim<2>{permutation_indices.get_size()}),
      permutation_{exec, std::move(permutation_indices)}
{}


template <typename IndexType>
Permutation<IndexType>::Permutation(std::shared_ptr<const Executor> exec,
                                    const dim<2>& size)
    : Permutation{exec, size[0]}
{
    GKO_ASSERT_IS_SQUARE_MATRIX(size);
}


template <typename IndexType>
Permutation<IndexType>::Permutation(std::shared_ptr<const Executor> exec,
                                    const dim<2>& size,
                                    const mask_type& enabled_permute)
    : Permutation{exec, size[0]}
{
    GKO_ASSERT_EQ(enabled_permute, row_permute);
    GKO_ASSERT_IS_SQUARE_MATRIX(size);
}


template <typename IndexType>
size_type Permutation<IndexType>::get_permutation_size() const noexcept
{
    return this->get_size()[0];
}


template <typename IndexType>
mask_type Permutation<IndexType>::get_permute_mask() const
{
    return row_permute;
}


template <typename IndexType>
void Permutation<IndexType>::set_permute_mask(mask_type permute_mask)
{
    GKO_ASSERT_EQ(permute_mask, row_permute);
}


template <typename IndexType>
std::unique_ptr<Permutation<IndexType>>
Permutation<IndexType>::compute_inverse() const
{
    const auto exec = this->get_executor();
    const auto size = this->get_size()[0];
    auto result = Permutation<IndexType>::create(exec, size);
    exec->run(permutation::make_invert(this->get_const_permutation(), size,
                                       result->get_permutation()));
    return result;
}


template <typename IndexType>
std::unique_ptr<Permutation<IndexType>> Permutation<IndexType>::compose(
    ptr_param<const Permutation<IndexType>> other) const
{
    GKO_ASSERT_EQUAL_DIMENSIONS(this, other);
    const auto exec = this->get_executor();
    const auto size = this->get_size()[0];
    const auto local_other = make_temporary_clone(exec, other);
    auto result = Permutation<IndexType>::create(exec, size);
    exec->run(permutation::make_compose(this->get_const_permutation(),
                                        local_other->get_const_permutation(),
                                        size, result->get_permutation()));
    return result;
}


template <typename IndexType>
void Permutation<IndexType>::write(
    gko::matrix_data<value_type, index_type>& data) const
{
    const auto host_this =
        make_temporary_clone(this->get_executor()->get_master(), this);
    data.size = this->get_size();
    data.nonzeros.clear();
    data.nonzeros.reserve(data.size[0]);
    for (IndexType row = 0; row < this->get_size()[0]; row++) {
        data.nonzeros.emplace_back(row, host_this->get_const_permutation()[row],
                                   one<value_type>());
    }
}


template <typename Functor>
void dispatch_dense(const LinOp* op, Functor fn)
{
    using matrix::Dense;
    using std::complex;
    run<const Dense<double>*, const Dense<float>*,
        const Dense<complex<double>>*, const Dense<complex<float>>*>(op, fn);
}


template <typename IndexType>
void Permutation<IndexType>::apply_impl(const LinOp* in, LinOp* out) const
{
    dispatch_dense(in, [&](auto dense_in) {
        auto dense_out = make_temporary_conversion<
            typename gko::detail::pointee<decltype(dense_in)>::value_type>(out);
        dense_in->permute(this, dense_out.get(), permute_mode::rows);
    });
}


template <typename IndexType>
void Permutation<IndexType>::apply_impl(const LinOp* alpha, const LinOp* in,
                                        const LinOp* beta, LinOp* out) const
{
    dispatch_dense(in, [&](auto dense_in) {
        auto dense_out = make_temporary_conversion<
            typename gko::detail::pointee<decltype(dense_in)>::value_type>(out);
        auto tmp = dense_in->permute(this, permute_mode::rows);
        dense_out->scale(beta);
        dense_out->add_scaled(alpha, tmp);
    });
}


#define GKO_DECLARE_PERMUTATION_MATRIX(_type) class Permutation<_type>
GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_PERMUTATION_MATRIX);


}  // namespace matrix
}  // namespace gko
