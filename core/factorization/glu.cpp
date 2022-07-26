/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2022, the Ginkgo authors
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

#include <ginkgo/core/factorization/glu.hpp>
#include <ginkgo/core/factorization/lu.hpp>

#include <memory>


#include <fstream>
#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/mtx_io.hpp>
#include <ginkgo/core/matrix/diagonal.hpp>


#include "third_party/glu/include/Timer.h"
#include "third_party/glu/include/numeric.h"
#include "third_party/glu/include/symbolic.h"
#include "third_party/glu/src/nicslu/include/nics_config.h"
#include "third_party/glu/src/nicslu/include/nicslu.h"
#include "third_party/glu/src/preprocess/preprocess.h"


#include "core/factorization/factorization_kernels.hpp"
#include "core/factorization/glu_kernels.hpp"
#include "core/factorization/par_ilu_kernels.hpp"


namespace gko {
namespace experimental {
namespace factorization {
namespace glu_factorization {
namespace {


GKO_REGISTER_OPERATION(compute_glu, glu_factorization::compute_lu);
GKO_REGISTER_OPERATION(add_diagonal_elements,
                       factorization::add_diagonal_elements);
GKO_REGISTER_OPERATION(initialize_row_ptrs_l_u,
                       factorization::initialize_row_ptrs_l_u);
GKO_REGISTER_OPERATION(initialize_l_u, factorization::initialize_l_u);


}  // anonymous namespace
}  // namespace glu_factorization


template <typename ValueType, typename IndexType>
void Glu<ValueType, IndexType>::ReusableFactory::symbolic_factorization(
    const LinOp* system_matrix) GKO_NOT_IMPLEMENTED;


template <>
void Glu<double, int>::ReusableFactory::symbolic_factorization(
    const LinOp* system_matrix)
{
    GKO_ASSERT_IS_SQUARE_MATRIX(system_matrix);
    const auto exec = this->get_executor();

    // Converts the system matrix to CSR.
    // Throws an exception if it is not convertible.
    auto local_system_matrix = matrix_type::create(exec->get_master());
    as<ConvertibleTo<matrix_type>>(system_matrix)
        ->convert_to(local_system_matrix.get());

    auto v = local_system_matrix->get_values();
    auto r = local_system_matrix->get_row_ptrs();
    auto c = local_system_matrix->get_col_idxs();

    // Transpose system matrix as GLU starts off with a CSC matrix.
    auto transp = as<matrix_type>(local_system_matrix->transpose());
    auto values = transp->get_values();
    auto col_ptrs = transp->get_row_ptrs();
    auto row_idxs = transp->get_col_idxs();

    const auto matrix_size = local_system_matrix->get_size();
    const auto num_rows = matrix_size[0];
    const auto nnz = local_system_matrix->get_num_stored_elements();

    // Convert index arrays to unsigned int as this is what GLU uses for
    // indices.
    unsigned int* u_col_ptrs = new unsigned int[num_rows + 1];
    unsigned int* u_row_idxs = new unsigned int[nnz];
    for (auto i = 0; i < nnz; i++) {
        u_row_idxs[i] = (unsigned int)(row_idxs[i]);
        if (i <= num_rows) {
            u_col_ptrs[i] = (unsigned int)(col_ptrs[i]);
        }
    }

    // Symbolic factorization on the CPU
    Symbolic_Matrix A_sym(num_rows, cout, cerr);
    A_sym.fill_in(u_row_idxs, u_col_ptrs);
    A_sym.csr();
    A_sym.leveling();

    gko::array<int> row_ptrs{exec->get_master(), num_rows + 1};
    gko::array<int> col_idxs{exec->get_master(), A_sym.nnz};

    for (auto i = 0; i < A_sym.nnz; i++) {
        col_idxs.get_data()[i] = (int)(A_sym.csr_c_idx[i]);
        if (i <= num_rows) {
            row_ptrs.get_data()[i] = (int)(A_sym.csr_r_ptr[i]);
        }
    }

    symbolic_ = gko::matrix::SparsityCsr<double, int>::create(
        exec, system_matrix->get_size(), col_idxs, row_ptrs);
}


template <typename ValueType, typename IndexType>
std::unique_ptr<Factorization<ValueType, IndexType>>
Glu<ValueType, IndexType>::generate_l_u(
    const std::shared_ptr<const LinOp>& system_matrix,
    const std::shared_ptr<gko::matrix::SparsityCsr<ValueType, IndexType>>
        symbolic) GKO_NOT_IMPLEMENTED;


template <>
std::unique_ptr<Factorization<double, int>> Glu<double, int>::generate_l_u(
    const std::shared_ptr<const LinOp>& system_matrix,
    const std::shared_ptr<gko::matrix::SparsityCsr<double, int>> symbolic)
{
    auto exec = system_matrix->get_executor();
    const auto matrix_size = system_matrix->get_size();
    const auto num_rows = matrix_size[0];

    auto fact = gko::experimental::factorization::Lu<double, int>::build()
                    .with_symbolic_factorization(symbolic)
                    .on(exec);
    auto lu = fact->generate(system_matrix);
    return lu;
}


#define GKO_DECLARE_GLU(ValueType, IndexType) class Glu<ValueType, IndexType>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_GLU);


}  // namespace factorization
}  // namespace experimental
}  // namespace gko
