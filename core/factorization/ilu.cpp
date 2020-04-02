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

#include <ginkgo/core/factorization/ilu.hpp>


#include <memory>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>


#include "core/factorization/factorization_kernels.hpp"
#include "core/factorization/ilu_kernels.hpp"
#include "core/factorization/par_ilu_kernels.hpp"


namespace gko {
namespace factorization {
namespace ilu_factorization {


GKO_REGISTER_OPERATION(compute_ilu, ilu_factorization::compute_lu);
GKO_REGISTER_OPERATION(add_diagonal_elements,
                       factorization::add_diagonal_elements);
GKO_REGISTER_OPERATION(initialize_row_ptrs_l_u,
                       factorization::initialize_row_ptrs_l_u);
GKO_REGISTER_OPERATION(initialize_l_u, factorization::initialize_l_u);


}  // namespace ilu_factorization


template <typename ValueType, typename IndexType>
std::unique_ptr<Composition<ValueType>> Ilu<ValueType, IndexType>::generate_l_u(
    const std::shared_ptr<const LinOp> &system_matrix) const
{
    GKO_ASSERT_IS_SQUARE_MATRIX(system_matrix);

    const auto exec = this->get_executor();
    const auto host_exec = exec->get_master();

    // Converts the system matrix to CSR.
    // Throws an exception if it is not convertible.
    auto local_system_matrix = matrix_type::create(exec);
    as<ConvertibleTo<matrix_type>>(system_matrix.get())
        ->convert_to(local_system_matrix.get());

    // Add explicit diagonal zero elements if they are missing
    exec->run(ilu_factorization::make_add_diagonal_elements(
        local_system_matrix.get(), false));

    // Compute LU factorization
    exec->run(ilu_factorization::make_compute_ilu(local_system_matrix.get()));

    // Separate L and U factors: nnz
    const auto matrix_size = local_system_matrix->get_size();
    const auto num_rows = matrix_size[0];
    Array<IndexType> l_row_ptrs{exec, num_rows + 1};
    Array<IndexType> u_row_ptrs{exec, num_rows + 1};
    exec->run(ilu_factorization::make_initialize_row_ptrs_l_u(
        local_system_matrix.get(), l_row_ptrs.get_data(),
        u_row_ptrs.get_data()));

    IndexType l_nnz_it{};
    IndexType u_nnz_it{};
    // Get nnz from device memory
    host_exec->copy_from(exec.get(), 1, l_row_ptrs.get_data() + num_rows,
                         &l_nnz_it);
    host_exec->copy_from(exec.get(), 1, u_row_ptrs.get_data() + num_rows,
                         &u_nnz_it);
    auto l_nnz = static_cast<size_type>(l_nnz_it);
    auto u_nnz = static_cast<size_type>(u_nnz_it);

    // Init arrays
    Array<IndexType> l_col_idxs{exec, l_nnz};
    Array<ValueType> l_vals{exec, l_nnz};
    std::shared_ptr<matrix_type> l_factor = matrix_type::create(
        exec, matrix_size, std::move(l_vals), std::move(l_col_idxs),
        std::move(l_row_ptrs), parameters_.l_strategy);
    Array<IndexType> u_col_idxs{exec, u_nnz};
    Array<ValueType> u_vals{exec, u_nnz};
    std::shared_ptr<matrix_type> u_factor = matrix_type::create(
        exec, matrix_size, std::move(u_vals), std::move(u_col_idxs),
        std::move(u_row_ptrs), parameters_.u_strategy);

    // Separate L and U: columns and values
    exec->run(ilu_factorization::make_initialize_l_u(
        local_system_matrix.get(), l_factor.get(), u_factor.get()));

    return Composition<ValueType>::create(std::move(l_factor),
                                          std::move(u_factor));
}


#define GKO_DECLARE_ILU(ValueType, IndexType) class Ilu<ValueType, IndexType>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_ILU);


}  // namespace factorization
}  // namespace gko
