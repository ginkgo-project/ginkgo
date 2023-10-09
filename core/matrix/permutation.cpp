/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2023, the Ginkgo authors
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

#include <ginkgo/core/matrix/permutation.hpp>
#include "core/base/dispatch_helper.hpp"
#include "core/matrix/permutation_kernels.hpp"
#include "ginkgo/core/base/exception_helpers.hpp"
#include "ginkgo/core/base/executor.hpp"
#include "ginkgo/core/base/precision_dispatch.hpp"
#include "ginkgo/core/base/utils_helper.hpp"


namespace gko {
namespace matrix {
namespace permutation {


GKO_REGISTER_OPERATION(invert, permutation::invert);


}


template <typename IndexType>
std::unique_ptr<const Permutation<IndexType>>
Permutation<IndexType>::create_const(
    std::shared_ptr<const Executor> exec, size_type size,
    gko::detail::const_array_view<IndexType>&& perm_idxs,
    mask_type enabled_permute)
{
    GKO_ASSERT_EQ(enabled_permute, row_permute);
    GKO_ASSERT_EQ(size, perm_idxs.get_num_elems());
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
    : EnableLinOp<Permutation>(exec, size), permutation_{exec, size}
{}


template <typename IndexType>
Permutation<IndexType>::Permutation(std::shared_ptr<const Executor> exec,
                                    array<index_type> permutation_indices)
    : EnableLinOp<Permutation>(exec, permutation_indices.get_num_elems()),
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
Permutation<IndexType>::Permutation(std::shared_ptr<const Executor> exec,
                                    const dim<2>& size,
                                    array<index_type> permutation_indices)
    : Permutation{std::move(exec), std::move(permutation_indices)}
{
    GKO_ASSERT_EQ(size[0], permutation_.get_num_elems());
    GKO_ASSERT_IS_SQUARE_MATRIX(size);
}


template <typename IndexType>
Permutation<IndexType>::Permutation(std::shared_ptr<const Executor> exec,
                                    const dim<2>& size,
                                    array<index_type> permutation_indices,
                                    const mask_type& enabled_permute)
    : Permutation{std::move(exec), std::move(permutation_indices)}
{
    GKO_ASSERT_EQ(enabled_permute, row_permute);
    GKO_ASSERT_EQ(size[0], permutation_.get_num_elems());
    GKO_ASSERT_IS_SQUARE_MATRIX(size);
}


template <typename IndexType>
size_type Permutation<IndexType>::get_permutation_size() const noexcept
{
    return permutation_.get_num_elems();
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
std::unique_ptr<Permutation<IndexType>> Permutation<IndexType>::invert() const
{
    const auto exec = this->get_executor();
    const auto size = this->get_size()[0];
    array<index_type> inv_permutation{exec, size};
    exec->run(permutation::make_invert(this->get_const_permutation(), size,
                                       inv_permutation.get_data()));
    return Permutation::create(exec, std::move(inv_permutation));
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
    if (dynamic_cast<const ConvertibleTo<Dense<double>>*>(op)) {
        run<const Dense<double>*, const Dense<float>*>(op, fn);
    } else if (dynamic_cast<const ConvertibleTo<Dense<complex<double>>>*>(op)) {
        run<const Dense<complex<double>>*, const Dense<complex<float>>*>(op,
                                                                         fn);
    } else {
        GKO_NOT_SUPPORTED(*op);
    }
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
