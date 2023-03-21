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

template <typename ValueType>
inline void batch_entry_tridiagonal_thomas_solve_impl(
    const size_t ibatch, const ValueType* const tridiag_mat_subdiags,
    ValueType* const tridiag_mat_maindiags,
    const ValueType* const tridiag_mat_superdiags,
    const gko::batch_dense::UniformBatch<ValueType>& rhs,
    const gko::batch_dense::UniformBatch<ValueType>& x,
    unsigned char* const local_space = nullptr)
{
    const auto nrows = rhs.num_rows;
    const auto nrhs = rhs.num_rhs;

    const gko::batch_dense::BatchEntry<ValueType> d_entry =
        gko::batch::batch_entry(rhs, ibatch);

    const gko::batch_dense::BatchEntry<ValueType> x_entry =
        gko::batch::batch_entry(x, ibatch);

    const auto a_entry = tridiag_mat_subdiags + ibatch * nrows;
    auto b_entry = tridiag_mat_maindiags + ibatch * nrows;
    const auto c_entry = tridiag_mat_superdiags + ibatch * nrows;

    // Thomas algorithm - simplified gaussian elimination

    // Forward elimination phase
    for (int row_idx = 1; row_idx < nrows; row_idx++) {
        const auto mult = a_entry[row_idx] / b_entry[row_idx - 1];

        b_entry[row_idx] -= mult * c_entry[row_idx - 1];

        for (int rhs_idx = 0; rhs_idx < nrhs; rhs_idx++) {
            d_entry.values[row_idx * d_entry.stride + rhs_idx] -=
                mult * d_entry.values[(row_idx - 1) * d_entry.stride + rhs_idx];
        }
    }

    // Backward substitution phase
    for (int rhs_idx = 0; rhs_idx < nrhs; rhs_idx++) {
        x_entry.values[(nrows - 1) * x_entry.stride + rhs_idx] =
            d_entry.values[(nrows - 1) * d_entry.stride + rhs_idx] /
            b_entry[nrows - 1];

        for (int row_idx = nrows - 2; row_idx >= 0; row_idx--) {
            x_entry.values[row_idx * x_entry.stride + rhs_idx] =
                (d_entry.values[row_idx * d_entry.stride + rhs_idx] -
                 c_entry[row_idx] *
                     x_entry.values[(row_idx + 1) * x_entry.stride + rhs_idx]) /
                b_entry[row_idx];
        }
    }
}