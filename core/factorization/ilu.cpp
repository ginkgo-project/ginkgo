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

#include <ginkgo/core/factorization/ilu.hpp>


#include <memory>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/polymorphic_object.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/csr.hpp>


#include "core/factorization/ilu_kernels.hpp"


namespace gko {
namespace factorization {
namespace par_ilu_factorization {

GKO_REGISTER_OPERATION(compute_nnz_l_u, par_ilu_factorization::compute_nnz_l_u);
GKO_REGISTER_OPERATION(initialize_l_u, par_ilu_factorization::initialize_l_u);
GKO_REGISTER_OPERATION(compute_l_u_factors,
                       par_ilu_factorization::compute_l_u_factors);


}  // namespace par_ilu_factorization


template <typename ValueType, typename IndexType>
void ParIluFactory<ValueType, IndexType>::generate_impl(
    std::shared_ptr<const LinOp> system)
{
    using CsrMatrix = matrix::Csr<ValueType, IndexType>;
    using CooMatrix = matrix::Coo<ValueType, IndexType>;

    const auto exec = this->get_executor();
    // only copies if it is not on the same executor (or was not in the right
    // format)
    auto csr_system_matrix = copy_and_convert_to<CsrMatrix>(system.get());

    const auto matrix_size = csr_system_matrix->get_size();
    size_t l_nnz{};
    size_t u_nnz{};
    exec->run(ilu_factorization::make_compute_nnz_l_u(csr_system_matrix.get(),
                                                      &l_nnz, &u_nnz));
    auto l_factor =
        l_matrix_type::create(exec, matrix_size, l_nnz /* TODO set strategy */);
    auto u_factor =
        u_matrix_type::create(exec, matrix_size, u_nnz /* TODO set strategy */);

    // TODO create a new kernel that does the fill-in
    exec->run(ilu_factorization::make_initialize_l_u(
        csr_system_matrix.get(), l_factor.get(), u_factor.get()));
    auto u_factor_trans = u_factor->transpose();
    // TODO compute l_factor and u_factor

    // Use copy_and_convert_to again in case it was originally a COO matrix, so
    // the conversion would not be necessary
    auto coo_system_matrix = copy_and_convert_to<CooMatrix>(system.get());

    exec->run(ilu_factorization::make_compute_l_u_factors(
        coo_system_matrix.get(), l_factor.get(), u_factor_transpose.get()));

    // TODO maybe directly call the csr kernel for transpose so there is one
    // less allocation and deletion!
    u_factor = u_factor_trans->transpose();

    return ComposedLinOp::create(std::move(l_factor), std::move(u_factor));
}


#define GKO_DECLARE_PAR_ILU_FACTORY(ValueType, IndexType) \
    class ParIluFactory<ValueType, IndexType>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_PAR_ILU_FACTORY);


}  // namespace factorization
}  // namespace gko
