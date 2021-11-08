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

#include <ginkgo/ginkgo.hpp>


#include <cinttypes>
#include <iostream>
#include <utility>


// Helper function to properly print any Ginkgo matrix in Coordinate format
template <typename MatrixType>
void write_sparse(std::ostream& os, MatrixType* mtx)
{
    // gko::layout_type::array (the default) prints it as a column-major array
    gko::write(os, mtx, gko::layout_type::coordinate);
}


int main(int argc, char* argv[])
{
    using ValueType = double;
    using RealValueType = gko::remove_complex<ValueType>;
    using IndexType = std::int32_t;

    // Type aliases for different matrix representations
    using dense = gko::matrix::Dense<ValueType>;
    using csr = gko::matrix::Csr<ValueType, IndexType>;
    using coo = gko::matrix::Coo<ValueType, IndexType>;
    using matrix_data = gko::matrix_data<ValueType, IndexType>;

    // Exemplary input data. Stores a 3x4 matrix in row-major
    std::array<ValueType, 12> input_data{0,  1,  2,  3,    // First row
                                         10, 0,  12, 13,   // Second row
                                         20, 21, 0,  23};  // Third row
    const auto data_size = gko::dim<2>{3, 4};
    const gko::size_type nnz = input_data.size() - 3;

    auto omp = gko::OmpExecutor::create();

    std::cout << "Note: MatrixMarket prints indices in base-1\n";

    // --------------------------------------------------
    // Example 1: Use existing memory by creating an array view
    //            This can be done with any matrix as long as your existing
    //            memory is matching the format perfectly
    auto example1_view =
        gko::Array<ValueType>::view(omp, input_data.size(), input_data.data());

    auto example1_dense =
        dense::create(omp, data_size, std::move(example1_view), data_size[1]);
    std::cout << "Example 1 dense matrix:\n";
    write_sparse(std::cout, gko::lend(example1_dense));

    // --------------------------------------------------
    // Example 2: Filling directly into a Dense matrix
    auto example2_dense = dense::create(omp, data_size);
    for (int i = 0; i < data_size[0]; ++i) {
        for (int j = 0; j < data_size[1]; ++j) {
            example2_dense->at(i, j) = input_data[i * data_size[1] + j];
        }
    }

    // Potential step 2: converting it into destination format.
    auto example2_csr = csr::create(omp);
    example2_dense->convert_to(gko::lend(example2_csr));

    // Make sure it is correct by printing it
    std::cout << "Example 2 dense matrix:\n";
    write_sparse(std::cout, gko::lend(example2_dense));
    std::cout << "Example 2 CSR matrix:\n";
    write_sparse(std::cout, gko::lend(example2_csr));

    // --------------------------------------------------
    // Example 3: Directly writing it to the CSR format
    // Note: this is complicated when doing it from dense, as the conversion
    //       Dense -> CSR is done manually. However, if the matrix properties
    //       are known beforehand, this process can be simplified significantly
    auto example3 = csr::create(omp, data_size, nnz);
    gko::size_type csr_idx{0};
    gko::size_type last_row = 0;
    example3->get_row_ptrs()[0] = 0;
    for (gko::size_type i = 0; i < input_data.size(); ++i) {
        if (input_data[i] != 0) {
            const auto row = i / data_size[1];
            const auto col = i % data_size[1];
            example3->get_values()[csr_idx] = input_data[i];
            example3->get_col_idxs()[csr_idx] = static_cast<IndexType>(col);
            for (gko::size_type fix = last_row + 1; fix <= row; ++fix) {
                example3->get_row_ptrs()[fix] = static_cast<IndexType>(csr_idx);
            }
            ++csr_idx;
        }
    }
    // Fill in the remaining row_ptrs (all pointing to the end)
    for (gko::size_type fix = last_row + 1; fix <= data_size[0]; ++fix) {
        example3->get_row_ptrs()[fix] = static_cast<IndexType>(csr_idx);
    }
    if (csr_idx != nnz) {
        std::cerr << "Manual conversion from Dense to CSR failed!\n";
        std::cerr << "csr_idx: " << csr_idx << "; nnz: " << nnz << '\n';
        // return 0;
    }

    std::cout << "Example 3 CSR matrix:\n";
    write_sparse(std::cout, gko::lend(example2_csr));

    // --------------------------------------------------
    // Example 4: Write the data to matrix_data, and use this intermediate
    // representation to convert to any Ginkgo format
    matrix_data example4_data{data_size};
    for (IndexType row = 0; row < data_size[0]; ++row) {
        for (IndexType col = 0; col < data_size[1]; ++col) {
            const auto input_idx = row * data_size[1] + col;
            if (input_data[input_idx] != 0) {
                example4_data.nonzeros.emplace_back(row, col,
                                                    input_data[input_idx]);
            }
        }
    }

    // Now, this can be transformed into any matrix format Ginkgo supports:
    auto example4_csr = csr::create(omp);
    example4_csr->read(example4_data);
    auto example4_coo = coo::create(omp);
    example4_coo->read(example4_data);

    std::cout << "Example 4 CSR matrix:\n";
    write_sparse(std::cout, gko::lend(example4_csr));
    std::cout << "Example 4 COO matrix:\n";
    write_sparse(std::cout, gko::lend(example4_coo));

    // Print the ginkgo version information.
    std::cout << gko::version_info::get() << std::endl;
}
