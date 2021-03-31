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

#include <ginkgo/core/factorization/par_bilu.hpp>


#include <memory>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/matrix/fbcsr.hpp>


#include "core/factorization/block_factorization_kernels.hpp"
#include "core/factorization/par_bilu_kernels.hpp"


namespace gko {
namespace factorization {
namespace par_bilu_factorization {


GKO_REGISTER_OPERATION(compute_bilu, bilu_factorization::compute_bilu);
GKO_REGISTER_OPERATION(add_diagonal_blocks, factorization::add_diagonal_blocks);
GKO_REGISTER_OPERATION(initialize_row_ptrs_BLU,
                       factorization::initialize_row_ptrs_BLU);
GKO_REGISTER_OPERATION(initialize_BLU, factorization::initialize_BLU);
GKO_REGISTER_OPERATION(fbcsr_transpose, fbcsr::transpose);


}  // namespace par_bilu_factorization


template <typename ValueType, typename IndexType>
std::unique_ptr<Composition<ValueType>>
Bilu<ValueType, IndexType>::generate_block_LU(
    const std::shared_ptr<const LinOp> system_matrix) const
{
    GKO_ASSERT_IS_SQUARE_MATRIX(system_matrix);

    const auto exec = this->get_executor();

    // Converts the system matrix to FBCSR.
    // Throws an exception if it is not convertible.
    auto c_system_matrix =
        std::dynamic_pointer_cast<matrix_type>(system_matrix);
    if (!c_system_matrix) throw NotSupported(__FILE__, __LINE__, __func__, "");

    const int blksz = c_system_matrix->get_block_size();

    // Add explicit diagonal nonsingular blocks if they are missing
    exec->run(par_bilu_factorization::make_add_diagonal_blocks(
        c_system_matrix.get(), false));

    // Separate L and U factors: nnz
    const auto matrix_size = c_system_matrix->get_size();
    const auto num_rows = matrix_size[0];
    Array<IndexType> l_row_ptrs{exec, num_rows + 1};
    Array<IndexType> u_row_ptrs{exec, num_rows + 1};
    exec->run(par_bilu_factorization::make_initialize_row_ptrs_BLU(
        c_system_matrix.get(), l_row_ptrs.get_data(), u_row_ptrs.get_data()));

    // Get nnz from device memory
    auto l_nnz = static_cast<size_type>(
        exec->copy_val_to_host(l_row_ptrs.get_data() + num_rows));
    auto u_nnz = static_cast<size_type>(
        exec->copy_val_to_host(u_row_ptrs.get_data() + num_rows));

    // Init arrays
    Array<IndexType> l_col_idxs{exec, l_nnz};
    Array<ValueType> l_vals{exec, l_nnz};
    std::shared_ptr<l_matrix_type> l_factor =
        matrix_type::create(exec, matrix_size, blksz, std::move(l_vals),
                            std::move(l_col_idxs), std::move(l_row_ptrs));
    Array<IndexType> u_col_idxs{exec, u_nnz};
    Array<ValueType> u_vals{exec, u_nnz};
    std::shared_ptr<u_matrix_type> u_factor =
        matrix_type::create(exec, matrix_size, blksz, std::move(u_vals),
                            std::move(u_col_idxs), std::move(u_row_ptrs));

    // Separate L and U: columns and values
    exec->run(bilu_factorization::make_initialize_BLU(
        c_system_matrix.get(), l_factor.get(), u_factor.get()));

    // We use `transpose()` here to convert the Csr format to Csc.
    auto u_factor_transpose_lin_op = u_factor->transpose();
    // Since `transpose()` returns an `std::unique_ptr<LinOp>`, we need to
    // convert it to `u_matrix_type *` in order to use it.
    auto u_factor_t =
        static_cast<u_matrix_type *>(u_factor_transpose_lin_op.get());

    // Compute LU factorization
    exec->run(par_bilu_factorization::make_compute_bilu(c_system_matrix.get()));

    // Transpose it again, which is basically a conversion from CSC back to CSR
    // Since the transposed version has the exact same non-zero positions
    // as `u_factor`, we can both skip the allocation and the `make_srow()`
    // call from CSR, leaving just the `transpose()` kernel call
    exec->run(par_bilu_factorization::make_fbcsr_transpose(u_factor_t,
                                                           u_factor.get()));

    return Composition<ValueType>::create(std::move(l_factor),
                                          std::move(u_factor));
}


#define GKO_DECLARE_BILU(ValueType, IndexType) class Bilu<ValueType, IndexType>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_BILU);


}  // namespace factorization
}  // namespace gko
