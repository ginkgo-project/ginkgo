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
#include <ginkgo/core/matrix/diagonal.hpp>
#include <ginkgo/core/matrix/fbcsr.hpp>
#include <ginkgo/core/matrix/sparsity_csr.hpp>


namespace gko {
namespace testing {


/** Generates the same sample block CSR matrix in different formats
 *
 * This currently a 6 x 12 matrix with 3x3 blocks.
 * Assumes that the layout within each block is row-major.
 * Generates complex data when instantiated with a complex value type.
 */
template <typename ValueType, typename IndexType>
class FbcsrSample {
public:
    using value_type = ValueType;
    using index_type = IndexType;
    using absvalue_type = remove_complex<value_type>;
    using Fbcsr = gko::matrix::Fbcsr<value_type, index_type>;
    using Csr = gko::matrix::Csr<value_type, index_type>;
    using Coo = gko::matrix::Coo<value_type, index_type>;
    using Dense = gko::matrix::Dense<value_type>;
    using MatData = gko::matrix_data<value_type, index_type>;
    using SparCsr = gko::matrix::SparsityCsr<value_type, index_type>;

    FbcsrSample(std::shared_ptr<const gko::ReferenceExecutor> exec);

    /**
     * @return The sample matrix in FBCSR format
     */
    std::unique_ptr<Fbcsr> generate_fbcsr() const;

    /**
     * @return Sample matrix in CSR format
     *
     * Keeps explicit zeros.
     */
    std::unique_ptr<Csr> generate_csr() const;

    /**
     * @return Sample matrix as dense
     */
    std::unique_ptr<Dense> generate_dense() const;

    /**
     * @return The matrix in COO format keeping explicit nonzeros
     *
     * The nonzeros are sorted by row and column.
     */
    std::unique_ptr<Coo> generate_coo() const;

    /**
     * @return Sparsity structure of the matrix
     */
    std::unique_ptr<SparCsr> generate_sparsity_csr() const;

    /**
     * @return Array of COO triplets that represent the matrix
     *
     * @note The order of the triplets assumes the blocks are stored row-major
     */
    MatData generate_matrix_data() const;

    /**
     * @return Array of COO triplets that represent the matrix; includes
     *         explicit zeros
     *
     * @note The order of the triplets assumes the blocks are stored row-major
     */
    MatData generate_matrix_data_with_explicit_zeros() const;

    /**
     *  @return An array containing number of stored values in each row
     */
    gko::Array<index_type> getNonzerosPerRow() const;

    /**
     * @return FBCSR matrix with absolute values of respective entries
     */
    std::unique_ptr<Fbcsr> generate_abs_fbcsr() const;

    /**
     * @return FBCSR matrix with real scalar type,
     *         with absolute values of respective entries
     */
    std::unique_ptr<gko::matrix::Fbcsr<remove_complex<value_type>, index_type>>
    generate_abs_fbcsr_abstype() const;


    const size_type nrows;
    const size_type ncols;
    const size_type nnz;
    const size_type nbrows;
    const size_type nbcols;
    const size_type nbnz;
    const int bs;
    const std::shared_ptr<const gko::Executor> exec;

private:
    template <typename FbcsrType>
    void correct_abs_for_complex_values(FbcsrType *const mat) const;

    /// Enables complex data to be used for complex instantiations...
    template <typename U>
    constexpr std::enable_if_t<!is_complex<U>() || is_complex<ValueType>(),
                               ValueType>
    sct(U u) const
    {
        return static_cast<ValueType>(u);
    }

    /// ... while ignoring imaginary parts for real instantiations
    template <typename U>
    constexpr std::enable_if_t<!is_complex<U>() && !is_complex<ValueType>(),
                               ValueType>
    sct(std::complex<U> cu) const
    {
        return static_cast<ValueType>(cu.real());
    }
};

/**
 * Generates a sample block CSR matrix in different formats.
 * 6 x 8 matrix with 2x2 blocks.
 */
template <typename ValueType, typename IndexType>
class FbcsrSample2 {
public:
    using value_type = ValueType;
    using index_type = IndexType;
    using Fbcsr = gko::matrix::Fbcsr<value_type, index_type>;
    using Dense = gko::matrix::Dense<value_type>;
    using Diagonal = gko::matrix::Diagonal<value_type>;

    FbcsrSample2(std::shared_ptr<const gko::ReferenceExecutor> exec);

    std::unique_ptr<Fbcsr> generate_fbcsr() const;

    std::unique_ptr<Fbcsr> generate_transpose_fbcsr() const;

    std::unique_ptr<Diagonal> extract_diagonal() const;

    void apply(const Dense *x, Dense *y) const;

    gko::Array<index_type> getNonzerosPerRow() const;

    std::unique_ptr<Fbcsr> generate_abs_fbcsr() const;

    std::unique_ptr<gko::matrix::Fbcsr<remove_complex<value_type>, index_type>>
    generate_abs_fbcsr_abstype() const;

    /// Enables use of literals to instantiate value data
    template <typename U>
    constexpr ValueType sct(U u) const
    {
        return static_cast<ValueType>(u);
    }


    const size_type nrows;
    const size_type ncols;
    const size_type nnz;
    const size_type nbrows;
    const size_type nbcols;
    const size_type nbnz;
    const int bs;
    const std::shared_ptr<const gko::Executor> exec;
};

/// Generates the a sample block CSR square matrix in different formats
/** This currently a 4 x 4 matrix with 2x2 blocks.
 */
template <typename ValueType, typename IndexType>
class FbcsrSampleSquare {
public:
    using value_type = ValueType;
    using index_type = IndexType;
    using Fbcsr = gko::matrix::Fbcsr<value_type, index_type>;

    FbcsrSampleSquare(std::shared_ptr<const gko::ReferenceExecutor> exec);

    std::unique_ptr<Fbcsr> generate_fbcsr() const;

    std::unique_ptr<Fbcsr> generate_transpose_fbcsr() const;

    const size_type nrows;
    const size_type ncols;
    const size_type nnz;
    const size_type nbrows;
    const size_type nbcols;
    const size_type nbnz;
    const int bs;
    const std::shared_ptr<const gko::Executor> exec;
};

/**
 * Generates the a sample block CSR matrix with complex values
 * This is a 6 x 8 matrix with 2x2 blocks.
 */
template <typename ValueType, typename IndexType>
class FbcsrSampleComplex {
public:
    using value_type = ValueType;
    using index_type = IndexType;
    using Fbcsr = gko::matrix::Fbcsr<value_type, index_type>;

    static_assert(is_complex<ValueType>(), "Only for complex types!");

    FbcsrSampleComplex(std::shared_ptr<const gko::ReferenceExecutor> exec);

    std::unique_ptr<Fbcsr> generate_fbcsr() const;

    std::unique_ptr<Fbcsr> generate_conjtranspose_fbcsr() const;


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
