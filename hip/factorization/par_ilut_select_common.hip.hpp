/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2021, the Ginkgo authors
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

#ifndef GKO_HIP_FACTORIZATION_PAR_ILUT_SELECT_COMMON_HIP_HPP_
#define GKO_HIP_FACTORIZATION_PAR_ILUT_SELECT_COMMON_HIP_HPP_


#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>


namespace gko {
namespace kernels {
namespace hip {
namespace par_ilut_factorization {


constexpr auto default_block_size = 512;
constexpr auto items_per_thread = 16;


template <typename ValueType, typename IndexType>
void sampleselect_count(std::shared_ptr<const DefaultExecutor> exec,
                        const ValueType *values, IndexType size,
                        remove_complex<ValueType> *tree, unsigned char *oracles,
                        IndexType *partial_counts, IndexType *total_counts);


template <typename IndexType>
struct sampleselect_bucket {
    IndexType idx;
    IndexType begin;
    IndexType size;
};


template <typename IndexType>
sampleselect_bucket<IndexType> sampleselect_find_bucket(
    std::shared_ptr<const DefaultExecutor> exec, IndexType *prefix_sum,
    IndexType rank);


}  // namespace par_ilut_factorization
}  // namespace hip
}  // namespace kernels
}  // namespace gko


#endif  // GKO_HIP_FACTORIZATION_PAR_ILUT_SELECT_COMMON_HIP_HPP_
