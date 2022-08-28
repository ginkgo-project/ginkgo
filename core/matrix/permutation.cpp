/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2022, the Ginkgo authors
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


namespace gko {
namespace matrix {


template <typename IndexType>
IndexType* Permutation<IndexType>::get_permutation() noexcept
{
    return permutation_.get_data();
}


template <typename IndexType>
const IndexType* Permutation<IndexType>::get_const_permutation() const noexcept
{
    return permutation_.get_const_data();
}


template <typename IndexType>
size_type Permutation<IndexType>::get_permutation_size() const noexcept
{
    return permutation_.get_num_elems();
}


template <typename IndexType>
mask_type Permutation<IndexType>::get_permute_mask() const
{
    return enabled_permute_;
}


template <typename IndexType>
void Permutation<IndexType>::set_permute_mask(mask_type permute_mask)
{
    enabled_permute_ = permute_mask;
}


template <typename IndexType>
std::unique_ptr<const Permutation<IndexType>>
Permutation<IndexType>::create_const(
    std::shared_ptr<const Executor> exec, size_type size,
    gko::detail::const_array_view<IndexType>&& perm_idxs,
    mask_type enabled_permute)
{
    // cast const-ness away, but return a const object afterwards,
    // so we can ensure that no modifications take place.
    return std::unique_ptr<const Permutation>(new Permutation{
        exec, size, gko::detail::array_const_cast(std::move(perm_idxs)),
        enabled_permute});
}


template <typename IndexType>
Permutation<IndexType>::Permutation(std::shared_ptr<const Executor> exec)
    : Permutation(std::move(exec), dim<2>{})
{}


template <typename IndexType>
Permutation<IndexType>::Permutation(std::shared_ptr<const Executor> exec,
                                    const dim<2>& size,
                                    const mask_type& enabled_permute)
    : EnableLinOp<Permutation>(exec, size),
      permutation_(exec, size[0]),
      row_size_(size[0]),
      col_size_(size[1]),
      enabled_permute_(enabled_permute)
{}


template <typename IndexType>
void Permutation<IndexType>::apply_impl(const LinOp* in, LinOp* out) const
{
    auto perm = as<Permutable<index_type>>(in);
    std::unique_ptr<gko::LinOp> tmp{};
    if (enabled_permute_ & inverse_permute) {
        if (enabled_permute_ & row_permute) {
            tmp = perm->inverse_row_permute(&permutation_);
        }
        if (enabled_permute_ & column_permute) {
            if (enabled_permute_ & row_permute) {
                tmp = as<Permutable<index_type>>(tmp.get())
                          ->inverse_column_permute(&permutation_);
            } else {
                tmp = perm->inverse_column_permute(&permutation_);
            }
        }
    } else {
        if (enabled_permute_ & row_permute) {
            tmp = perm->row_permute(&permutation_);
        }
        if (enabled_permute_ & column_permute) {
            if (enabled_permute_ & row_permute) {
                tmp = as<Permutable<index_type>>(tmp.get())->column_permute(
                    &permutation_);
            } else {
                tmp = perm->column_permute(&permutation_);
            }
        }
    }
    out->copy_from(std::move(tmp));
}


template <typename IndexType>
void Permutation<IndexType>::apply_impl(const LinOp*, const LinOp* in,
                                        const LinOp*, LinOp* out) const
{
    GKO_NOT_SUPPORTED(this);
}


#define GKO_DECLARE_PERMUTATION_MATRIX(_type) class Permutation<_type>
GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_PERMUTATION_MATRIX);


}  // namespace matrix
}  // namespace gko
