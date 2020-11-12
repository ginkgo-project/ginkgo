/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2020, the Ginkgo authors
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


#ifndef GKO_CORE_MATRIX_TEST_FBCSR_SAMPLE_HPP
#define GKO_CORE_MATRIX_TEST_FBCSR_SAMPLE_HPP

#include <ginkgo/core/base/matrix_data.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/fbcsr.hpp>
#include <ginkgo/core/matrix/sparsity_csr.hpp>

namespace gko {
namespace testing {


/// Generates the same sample block CSR matrix in different formats
/** This currently a 6 x 12 matrix with 3x3 blocks.
 */
template <typename ValueType, typename IndexType>
class FbcsrSample {
public:
    using value_type = ValueType;
    using index_type = IndexType;
    using Fbcsr = gko::matrix::Fbcsr<value_type, index_type>;
    using Csr = gko::matrix::Csr<value_type, index_type>;
    using Coo = gko::matrix::Coo<value_type, index_type>;
    using Dense = gko::matrix::Dense<value_type>;
    using MatData = gko::matrix_data<value_type, index_type>;
    using SparCsr = gko::matrix::SparsityCsr<value_type, index_type>;

    FbcsrSample(std::shared_ptr<const gko::ReferenceExecutor> exec);

    std::unique_ptr<Fbcsr> generate_fbcsr() const;

    /// Generates CSR matrix equal to the BSR matrix. Keeps explicit zeros.
    std::unique_ptr<Csr> generate_csr() const;

    std::unique_ptr<Dense> generate_dense() const;

    /// Returns the matrix in COO format keeping explicit nonzeros
    /** The nonzeros are sorted by row and column.
     */
    std::unique_ptr<Coo> generate_coo() const;

    std::unique_ptr<SparCsr> generate_sparsity_csr() const;

    MatData generate_matrix_data() const;

    MatData generate_matrix_data_with_explicit_zeros() const;

    const size_type nrows;
    const size_type ncols;
    const size_type nnz;
    const size_type nbrows;
    const size_type nbcols;
    const size_type nbnz;
    const int bs;
    const std::shared_ptr<const gko::Executor> exec;
};

}  // namespace testing
}  // namespace gko

#endif
