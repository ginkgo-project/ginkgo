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

#include <memory>


#include <ginkgo/config.hpp>
#include <ginkgo/core/distributed/vector.hpp>
#include <ginkgo/core/matrix/dense.hpp>


namespace gko {
namespace detail {


template <typename ValueType>
std::unique_ptr<matrix::Dense<ValueType>> create_with_same_size(
    const matrix::Dense<ValueType>* mtx)
{
    return matrix::Dense<ValueType>::create(mtx->get_executor(),
                                            mtx->get_size(), mtx->get_stride());
}


template <typename ValueType>
const matrix::Dense<ValueType>* get_local(const matrix::Dense<ValueType>* mtx)
{
    return mtx;
}


template <typename ValueType>
matrix::Dense<ValueType>* get_local(matrix::Dense<ValueType>* mtx)
{
    return mtx;
}


#if GINKGO_BUILD_MPI


template <typename ValueType>
std::unique_ptr<distributed::Vector<ValueType>> create_with_same_size(
    const distributed::Vector<ValueType>* mtx)
{
    return distributed::Vector<ValueType>::create(
        mtx->get_executor(), mtx->get_communicator(), mtx->get_size(),
        mtx->get_local_vector()->get_size(),
        mtx->get_local_vector()->get_stride());
}


template <typename ValueType>
matrix::Dense<ValueType>* get_local(distributed::Vector<ValueType>* mtx)
{
    return const_cast<matrix::Dense<ValueType>*>(mtx->get_local_vector());
}


template <typename ValueType>
const matrix::Dense<ValueType>* get_local(
    const distributed::Vector<ValueType>* mtx)
{
    return mtx->get_local_vector();
}


#endif


}  // namespace detail
}  // namespace gko
