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

#include <ginkgo/core/matrix/row_gatherer.hpp>


#include <ginkgo/core/matrix/dense.hpp>


#include "core/base/dispatch_helper.hpp"


namespace gko {
namespace matrix {


template <typename IndexType>
void RowGatherer<IndexType>::apply_impl(const LinOp* in, LinOp* out) const
{
    run<const Dense<gko::half>*, const Dense<float>*, const Dense<double>*,
        const Dense<std::complex<gko::half>>*,
        const Dense<std::complex<float>>*, const Dense<std::complex<double>>*>(
        in, [&](auto gather) { gather->row_gather(&row_idxs_, out); });
}

template <typename IndexType>
void RowGatherer<IndexType>::apply_impl(const LinOp* alpha, const LinOp* in,
                                        const LinOp* beta, LinOp* out) const
{
    run<const Dense<gko::half>*, const Dense<float>*, const Dense<double>*,
        const Dense<std::complex<gko::half>>*,
        const Dense<std::complex<float>>*, const Dense<std::complex<double>>*>(
        in,
        [&](auto gather) { gather->row_gather(alpha, &row_idxs_, beta, out); });
}


#define GKO_DECLARE_ROWGATHERER_MATRIX(_type) class RowGatherer<_type>
GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_ROWGATHERER_MATRIX);


}  // namespace matrix
}  // namespace gko
