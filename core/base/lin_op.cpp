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

#include <ginkgo/core/base/lin_op.hpp>


#include <ginkgo/core/matrix/permutation.hpp>


namespace gko {


template <typename IndexType>
static gko::array<IndexType> create_permutation_view(
    const matrix::Permutation<IndexType>* permutation)
{
    return make_array_view(
        permutation->get_executor(), permutation->get_permutation_size(),
        const_cast<IndexType*>(permutation->get_const_permutation()));
}


template <typename IndexType>
std::unique_ptr<LinOp> Permutable<IndexType>::permute(
    ptr_param<const matrix::Permutation<IndexType>> permutation) const
{
    auto array = create_permutation_view(permutation.get());
    return this->permute(&array);
}


template <typename IndexType>
std::unique_ptr<LinOp> Permutable<IndexType>::row_permute(
    ptr_param<const matrix::Permutation<IndexType>> permutation) const
{
    auto array = create_permutation_view(permutation.get());
    return this->row_permute(&array);
}


template <typename IndexType>
std::unique_ptr<LinOp> Permutable<IndexType>::column_permute(
    ptr_param<const matrix::Permutation<IndexType>> permutation) const
{
    auto array = create_permutation_view(permutation.get());
    return this->column_permute(&array);
}


template <typename IndexType>
std::unique_ptr<LinOp> Permutable<IndexType>::inverse_permute(
    ptr_param<const matrix::Permutation<IndexType>> permutation) const
{
    auto array = create_permutation_view(permutation.get());
    return this->inverse_permute(&array);
}


template <typename IndexType>
std::unique_ptr<LinOp> Permutable<IndexType>::inverse_row_permute(
    ptr_param<const matrix::Permutation<IndexType>> permutation) const
{
    auto array = create_permutation_view(permutation.get());
    return this->inverse_row_permute(&array);
}


template <typename IndexType>
std::unique_ptr<LinOp> Permutable<IndexType>::inverse_column_permute(
    ptr_param<const matrix::Permutation<IndexType>> permutation) const
{
    auto array = create_permutation_view(permutation.get());
    return this->inverse_column_permute(&array);
}


#define GKO_DECLARE_PERMUTABLE(IndexType) class Permutable<IndexType>

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_PERMUTABLE);


}  // namespace gko
