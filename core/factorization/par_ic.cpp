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

#include <ginkgo/core/factorization/par_ic.hpp>


#include <map>
#include <memory>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/composition.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/polymorphic_object.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/csr.hpp>


#include "core/components/validation_helpers.hpp"
#include "core/factorization/factorization_kernels.hpp"
#include "core/factorization/par_ic_kernels.hpp"
#include "core/matrix/csr_kernels.hpp"


namespace gko {
namespace factorization {


template <typename ValueType, typename IndexType>
void ParIc<ValueType, IndexType>::validate_impl() const
{
    std::map<std::string, std::function<bool()>> constraints_map{
        {"is_finite",
         [this] {
             bool l_factor_is_finite =
                 ::gko::validate::is_finite(get_l_factor().get());
             return this->parameters_.both_factors
                        ? ::gko::validate::is_finite(get_lt_factor().get()) &&
                              l_factor_is_finite
                        : l_factor_is_finite;
         }},
        {"has_non_zero_diagonal", [this] {
             bool l_factor_has_non_zero_diagonal =
                 ::gko::validate::has_non_zero_diagonal(get_l_factor().get());
             return this->parameters_.both_factors
                        ? ::gko::validate::has_non_zero_diagonal(
                              get_lt_factor().get()) &&
                              l_factor_has_non_zero_diagonal
                        : l_factor_has_non_zero_diagonal;
         }}};

    for (auto const &x : constraints_map) {
        if (!x.second()) {
            throw gko::Invalid(__FILE__, __LINE__, "ParIc", x.first);
        };
    }
}

namespace par_ic_factorization {


GKO_REGISTER_OPERATION(add_diagonal_elements,
                       factorization::add_diagonal_elements);
GKO_REGISTER_OPERATION(initialize_row_ptrs_l,
                       factorization::initialize_row_ptrs_l);
GKO_REGISTER_OPERATION(initialize_l, factorization::initialize_l);
GKO_REGISTER_OPERATION(init_factor, par_ic_factorization::init_factor);
GKO_REGISTER_OPERATION(compute_factor, par_ic_factorization::compute_factor);
GKO_REGISTER_OPERATION(csr_transpose, csr::transpose);
GKO_REGISTER_OPERATION(convert_to_coo, csr::convert_to_coo);


}  // namespace par_ic_factorization


template <typename ValueType, typename IndexType>
std::unique_ptr<Composition<ValueType>> ParIc<ValueType, IndexType>::generate(
    const std::shared_ptr<const LinOp> &system_matrix, bool skip_sorting,
    bool both_factors) const
{
    using CsrMatrix = matrix::Csr<ValueType, IndexType>;
    using CooMatrix = matrix::Coo<ValueType, IndexType>;

    GKO_ASSERT_IS_SQUARE_MATRIX(system_matrix);

    const auto exec = this->get_executor();

    // Converts the system matrix to CSR.
    // Throws an exception if it is not convertible.
    auto csr_system_matrix = CsrMatrix::create(exec);
    as<ConvertibleTo<CsrMatrix>>(system_matrix.get())
        ->convert_to(csr_system_matrix.get());
    // If necessary, sort it
    if (!skip_sorting) {
        csr_system_matrix->sort_by_column_index();
    }

    // Add explicit diagonal zero elements if they are missing
    exec->run(par_ic_factorization::make_add_diagonal_elements(
        csr_system_matrix.get(), true));

    const auto matrix_size = csr_system_matrix->get_size();
    const auto number_rows = matrix_size[0];
    Array<IndexType> l_row_ptrs{exec, number_rows + 1};
    exec->run(par_ic_factorization::make_initialize_row_ptrs_l(
        csr_system_matrix.get(), l_row_ptrs.get_data()));

    // Get nnz from device memory
    auto l_nnz = static_cast<size_type>(
        exec->copy_val_to_host(l_row_ptrs.get_data() + number_rows));

    // Since `row_ptrs` of L is already created, the matrix can be
    // directly created with it
    Array<IndexType> l_col_idxs{exec, l_nnz};
    Array<ValueType> l_vals{exec, l_nnz};
    std::shared_ptr<CsrMatrix> l_factor = matrix_type::create(
        exec, matrix_size, std::move(l_vals), std::move(l_col_idxs),
        std::move(l_row_ptrs), parameters_.l_strategy);

    exec->run(par_ic_factorization::make_initialize_l(csr_system_matrix.get(),
                                                      l_factor.get(), false));

    // build COO representation of lower factor
    Array<IndexType> l_row_idxs{exec, l_nnz};
    // copy values from l_factor, which are the lower triangular values of A
    auto l_vals_view =
        Array<ValueType>::view(exec, l_nnz, l_factor->get_values());
    auto a_vals = Array<ValueType>{exec, l_vals_view};
    auto a_row_idxs =
        Array<IndexType>::view(exec, l_nnz, l_factor->get_col_idxs());
    auto a_col_idxs = Array<IndexType>{exec, l_nnz};
    auto a_lower_coo =
        CooMatrix::create(exec, matrix_size, std::move(a_vals),
                          std::move(a_row_idxs), std::move(a_col_idxs));
    exec->run(par_ic_factorization::make_convert_to_coo(l_factor.get(),
                                                        a_lower_coo.get()));

    // compute sqrt of diagonal entries
    exec->run(par_ic_factorization::make_init_factor(l_factor.get()));

    // execute sweeps
    exec->run(par_ic_factorization::make_compute_factor(
        parameters_.iterations, a_lower_coo.get(), l_factor.get()));

    if (both_factors) {
        auto lh_factor = l_factor->conj_transpose();
        return Composition<ValueType>::create(std::move(l_factor),
                                              std::move(lh_factor));
    } else {
        return Composition<ValueType>::create(std::move(l_factor));
    }
}


#define GKO_DECLARE_PAR_IC(ValueType, IndexType) \
    class ParIc<ValueType, IndexType>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_PAR_IC);


}  // namespace factorization
}  // namespace gko
