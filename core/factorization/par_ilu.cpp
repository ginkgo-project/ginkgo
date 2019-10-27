/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2019, the Ginkgo authors
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

#include <ginkgo/core/factorization/par_ilu.hpp>


#include <memory>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/polymorphic_object.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/csr.hpp>


#include "core/factorization/par_ilu_kernels.hpp"
#include "core/matrix/csr_kernels.hpp"


namespace gko {
namespace factorization {
namespace par_ilu_factorization {


GKO_REGISTER_OPERATION(initialize_row_ptrs_l_u,
                       par_ilu_factorization::initialize_row_ptrs_l_u);
GKO_REGISTER_OPERATION(initialize_l_u, par_ilu_factorization::initialize_l_u);
GKO_REGISTER_OPERATION(compute_l_u_factors,
                       par_ilu_factorization::compute_l_u_factors);
GKO_REGISTER_OPERATION(csr_transpose, csr::transpose);


}  // namespace par_ilu_factorization


template <typename ValueType, typename IndexType>
std::unique_ptr<Composition<ValueType>>
ParIlu<ValueType, IndexType>::generate_l_u(
    const std::shared_ptr<const LinOp> &system_matrix, bool skip_sorting) const
{
    using CsrMatrix = matrix::Csr<ValueType, IndexType>;
    using CooMatrix = matrix::Coo<ValueType, IndexType>;

    GKO_ASSERT_IS_SQUARE_MATRIX(system_matrix);

    const auto exec = this->get_executor();
    const auto host_exec = exec->get_master();

    // If required, it is also possible to make this a Factory parameter
    auto csr_strategy = std::make_shared<typename CsrMatrix::sparselib>();

    // Only copies the matrix if it is not on the same executor or was not in
    // the right format. Throws an exception if it is not convertable.
    std::unique_ptr<CsrMatrix> csr_system_matrix_unique_ptr{};
    auto csr_system_matrix =
        dynamic_cast<const CsrMatrix *>(system_matrix.get());
    if (csr_system_matrix == nullptr ||
        csr_system_matrix->get_executor() != exec) {
        csr_system_matrix_unique_ptr = CsrMatrix::create(exec);
        as<ConvertibleTo<CsrMatrix>>(system_matrix.get())
            ->convert_to(csr_system_matrix_unique_ptr.get());
        csr_system_matrix = csr_system_matrix_unique_ptr.get();
    }
    // If it needs to be sorted, copy it if necessary and sort it
    if (!skip_sorting) {
        if (csr_system_matrix_unique_ptr == nullptr) {
            csr_system_matrix_unique_ptr = CsrMatrix::create(exec);
            csr_system_matrix_unique_ptr->copy_from(csr_system_matrix);
        }
        csr_system_matrix_unique_ptr->sort_by_column_index();
        csr_system_matrix = csr_system_matrix_unique_ptr.get();
    }

    const auto matrix_size = csr_system_matrix->get_size();
    const auto number_rows = matrix_size[0];
    Array<IndexType> l_row_ptrs{exec, number_rows + 1};
    Array<IndexType> u_row_ptrs{exec, number_rows + 1};
    exec->run(par_ilu_factorization::make_initialize_row_ptrs_l_u(
        csr_system_matrix, l_row_ptrs.get_data(), u_row_ptrs.get_data()));

    IndexType l_nnz_it;
    IndexType u_nnz_it;
    // Since nnz is always at row_ptrs[m], it can be extracted easily
    host_exec->get_mem_space()->copy_from(exec->get_mem_space().get(), 1,
                                          l_row_ptrs.get_data() + number_rows,
                                          &l_nnz_it);
    host_exec->get_mem_space()->copy_from(exec->get_mem_space().get(), 1,
                                          u_row_ptrs.get_data() + number_rows,
                                          &u_nnz_it);
    auto l_nnz = static_cast<size_type>(l_nnz_it);
    auto u_nnz = static_cast<size_type>(u_nnz_it);

    // Since `row_ptrs` of L and U is already created, the matrix can be
    // directly created with it
    Array<IndexType> l_col_idxs{exec, l_nnz};
    Array<ValueType> l_vals{exec, l_nnz};
    std::shared_ptr<CsrMatrix> l_factor = l_matrix_type::create(
        exec, matrix_size, std::move(l_vals), std::move(l_col_idxs),
        std::move(l_row_ptrs), csr_strategy);
    Array<IndexType> u_col_idxs{exec, u_nnz};
    Array<ValueType> u_vals{exec, u_nnz};
    std::shared_ptr<CsrMatrix> u_factor = u_matrix_type::create(
        exec, matrix_size, std::move(u_vals), std::move(u_col_idxs),
        std::move(u_row_ptrs), csr_strategy);

    exec->run(par_ilu_factorization::make_initialize_l_u(
        csr_system_matrix, l_factor.get(), u_factor.get()));

    // We use `transpose()` here to convert the Csr format to Csc.
    auto u_factor_transpose_lin_op = u_factor->transpose();
    // Since `transpose()` returns an `std::unique_ptr<LinOp>`, we need to
    // convert it to `u_matrix_type *` in order to use it.
    auto u_factor_transpose =
        static_cast<u_matrix_type *>(u_factor_transpose_lin_op.get());

    // At first, test if the given system_matrix was already a Coo matrix,
    // so no conversion would be necessary.
    std::unique_ptr<CooMatrix> coo_system_matrix_unique_ptr{nullptr};
    auto coo_system_matrix_ptr =
        dynamic_cast<const CooMatrix *>(system_matrix.get());

    // If it was not, and we already own a CSR `system_matrix`,
    // we can move the Csr matrix to Coo, which has very little overhead.
    // Otherwise, we convert from the Csr matrix, since it is the conversion
    // with the least overhead.
    // We also have to convert / move from the CSR matrix if it was not already
    // sorted (in which case we definitively own a CSR `system_matrix`).
    if (!skip_sorting || coo_system_matrix_ptr == nullptr) {
        coo_system_matrix_unique_ptr = CooMatrix::create(exec);
        if (csr_system_matrix_unique_ptr == nullptr) {
            csr_system_matrix->convert_to(coo_system_matrix_unique_ptr.get());
        } else {
            csr_system_matrix_unique_ptr->move_to(
                coo_system_matrix_unique_ptr.get());
        }
        coo_system_matrix_ptr = coo_system_matrix_unique_ptr.get();
    }

    exec->run(par_ilu_factorization::make_compute_l_u_factors(
        parameters_.iterations, coo_system_matrix_ptr, l_factor.get(),
        u_factor_transpose));

    // Transpose it again, which is basically a conversion from CSC back to CSR
    // Since the transposed version has the exact same non-zero positions
    // as `u_factor`, we can both skip the allocation and the `make_srow()`
    // call from CSR, leaving just the `transpose()` kernel call
    exec->run(par_ilu_factorization::make_csr_transpose(u_factor.get(),
                                                        u_factor_transpose));

    return Composition<ValueType>::create(std::move(l_factor),
                                          std::move(u_factor));
}


#define GKO_DECLARE_PAR_ILU(ValueType, IndexType) \
    class ParIlu<ValueType, IndexType>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_PAR_ILU);


}  // namespace factorization
}  // namespace gko
