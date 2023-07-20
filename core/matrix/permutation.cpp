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


namespace gko {
namespace matrix {


template <typename IndexType>
template <typename ValueType>
void Permutation<IndexType>::write_impl(
    gko::matrix_data<ValueType, index_type>& data) const
{
    const auto host_permutation = make_temporary_clone(
        this->get_executor()->get_master(), &this->permutation_);
    const auto num_rows = host_permutation->get_num_elems();
    data.size = {num_rows, num_rows};
    data.nonzeros.clear();
    auto perm = host_permutation->get_const_data();
    for (size_type row = 0; row < num_rows; row++) {
        data.nonzeros.emplace_back(static_cast<IndexType>(row), perm[row], 1);
    }
}


template <typename IndexType>
void Permutation<IndexType>::write(
    gko::matrix_data<float, index_type>& data) const
{
    this->write_impl(data);
}


template <typename IndexType>
void Permutation<IndexType>::write(
    gko::matrix_data<double, index_type>& data) const
{
    this->write_impl(data);
}


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
    out->move_from(tmp);
}


template <typename IndexType>
void Permutation<IndexType>::apply_impl(const LinOp*, const LinOp* in,
                                        const LinOp*, LinOp* out) const
{
    GKO_NOT_SUPPORTED(*this);
}


#define GKO_DECLARE_PERMUTATION_MATRIX(_type) class Permutation<_type>
GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_PERMUTATION_MATRIX);


}  // namespace matrix
}  // namespace gko
