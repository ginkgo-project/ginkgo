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


#include <ginkgo/core/base/polymorphic_object.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/csr.hpp>


#include "core/factorization/par_ilu_kernels.hpp"
#include "core/matrix/csr_kernels.hpp"


namespace gko {
namespace factorization {
namespace par_ilu_factorization {


GKO_REGISTER_OPERATION(compute_nnz_l_u, par_ilu_factorization::compute_nnz_l_u);
GKO_REGISTER_OPERATION(initialize_l_u, par_ilu_factorization::initialize_l_u);
GKO_REGISTER_OPERATION(compute_l_u_factors,
                       par_ilu_factorization::compute_l_u_factors);
GKO_REGISTER_OPERATION(csr_transpose, csr::transpose);


}  // namespace par_ilu_factorization


template <typename ValueType, typename IndexType>
std::unique_ptr<Composition<ValueType>>
ParIlu<ValueType, IndexType>::generate_l_u(
    const std::shared_ptr<const LinOp> &system_matrix) const
{
    using CsrMatrix = matrix::Csr<ValueType, IndexType>;
    using CooMatrix = matrix::Coo<ValueType, IndexType>;

    const auto exec = this->get_executor();
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

    const auto matrix_size = csr_system_matrix->get_size();
    size_type l_nnz{};
    size_type u_nnz{};
    exec->run(par_ilu_factorization::make_compute_nnz_l_u(csr_system_matrix,
                                                          &l_nnz, &u_nnz));
    auto l_factor =
        l_matrix_type::create(exec, matrix_size, l_nnz /* TODO set strategy */);
    auto u_factor =
        u_matrix_type::create(exec, matrix_size, u_nnz /* TODO set strategy */);

    exec->run(par_ilu_factorization::make_initialize_l_u(
        csr_system_matrix, l_factor.get(), u_factor.get()));
    auto u_factor_transpose_lin_op = u_factor->transpose();

    // We use `transpose()` here to convert the Csr format to Csc.
    // Since `transpose()` returns an `std::unique_ptr<LinOp>`, we need to
    // convert it to `u_matrix_type *` in order to use it.
    auto u_factor_transpose =
        static_cast<u_matrix_type *>(u_factor_transpose_lin_op.get());

    // At first, test if the given system_matrix was already a Coo matrix,
    // so no conversion would be necessary.
    std::unique_ptr<CooMatrix> coo_system_matrix_unique_ptr{nullptr};
    auto coo_system_matrix_ptr =
        dynamic_cast<const CooMatrix *>(system_matrix.get());

    // If it was not, and we already converted the `system_matrix` to CSR,
    // we can move the Csr matrix to Coo, which has very little overhead.
    // Otherwise, we convert from the Csr matrix, since it is the conversion
    // with the least overhead.
    if (coo_system_matrix_ptr == nullptr) {
        coo_system_matrix_unique_ptr = CooMatrix::create(exec);
        if (csr_system_matrix_unique_ptr == nullptr) {
            csr_system_matrix->convert_to(coo_system_matrix_unique_ptr.get());
        } else {
            csr_system_matrix_unique_ptr->move_to(
                coo_system_matrix_unique_ptr.get());
        }
        coo_system_matrix_ptr = coo_system_matrix_unique_ptr.get();
    }

    // TODO: We probably need to make sure that both the COO matrix and both CSR
    // matrices are sorted first by row, then by column
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
