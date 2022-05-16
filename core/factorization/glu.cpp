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

#include <ginkgo/core/factorization/glu.hpp>


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

    row_ptrs_ = A_sym.sym_c_ptr;
    col_idxs_ = A_sym.sym_r_idx;
    level_idx_ = A_sym.level_idx;
    level_ptr_ = A_sym.level_ptr;
    sym_nnz_ = A_sym.nnz;
    num_lev_ = A_sym.num_lev;
    csr_c_idx = A_sym.csr_c_idx;
    csr_r_ptr = A_sym.csr_r_ptr;
    csr_diag_ptr = A_sym.csr_diag_ptr;
    csr_val = A_sym.val;
    l_col_ptr = A_sym.l_col_ptr;
}


template <typename ValueType, typename IndexType>
std::unique_ptr<Composition<ValueType>> Glu<ValueType, IndexType>::generate_l_u(
    const std::shared_ptr<const LinOp>& system_matrix,
    const std::vector<unsigned>& row_ptrs,
    const std::vector<unsigned>& col_idxs, const std::vector<int>& level_idx,
    const std::vector<int>& level_ptr, const std::vector<unsigned>& csr_r_ptr,
    const std::vector<unsigned>& csr_c_idx,
    const std::vector<unsigned>& csr_diag_ptr,
    const std::vector<unsigned>& l_col_ptr,
    const std::vector<ValueType>& csr_val, unsigned sym_nnz, unsigned num_lev,
    bool skip_sorting) GKO_NOT_IMPLEMENTED;


template <>
std::unique_ptr<Composition<double>> Glu<double, int>::generate_l_u(
    const std::shared_ptr<const LinOp>& system_matrix,
    const std::vector<unsigned>& row_ptrs,
    const std::vector<unsigned>& col_idxs, const std::vector<int>& level_idx,
    const std::vector<int>& level_ptr, const std::vector<unsigned>& csr_r_ptr,
    const std::vector<unsigned>& csr_c_idx,
    const std::vector<unsigned>& csr_diag_ptr,
    const std::vector<unsigned>& l_col_ptr, const std::vector<double>& csr_val,
    unsigned sym_nnz, unsigned num_lev, bool skip_sorting)
{
    auto exec = system_matrix->get_executor();
    const auto matrix_size = system_matrix->get_size();
    const auto num_rows = matrix_size[0];

    Symbolic_Matrix A_sym{std::cout, std::cerr};
    A_sym.sym_c_ptr = row_ptrs;
    A_sym.sym_r_idx = col_idxs;
    A_sym.level_idx = level_idx;
    A_sym.level_ptr = level_ptr;
    A_sym.csr_r_ptr = csr_r_ptr;
    A_sym.csr_c_idx = csr_c_idx;
    A_sym.csr_diag_ptr = csr_diag_ptr;
    A_sym.nnz = sym_nnz;
    A_sym.num_lev = num_lev;
    A_sym.n = num_rows;
    A_sym.val = csr_val;
    A_sym.l_col_ptr = l_col_ptr;
    auto local_system_matrix = gko::share(matrix_type::create(exec));
    as<ConvertibleTo<matrix_type>>(system_matrix.get())
        ->convert_to(local_system_matrix.get());

    // Apply MC64 permutation and scaling and AMD reordering to system matrix
    auto transp = as<matrix_type>(local_system_matrix->transpose());

    auto host_transp = gko::clone(exec->get_master(), transp);
    auto vals = host_transp->get_values();
    auto col_ptrs = host_transp->get_row_ptrs();
    auto row_idxs = host_transp->get_col_idxs();
    const auto nnz = transp->get_num_stored_elements();
    unsigned int* u_col_ptrs = new unsigned int[num_rows + 1];
    unsigned int* u_row_idxs = new unsigned int[nnz];
    for (auto i = 0; i < nnz; i++) {
        u_row_idxs[i] = (unsigned int)(row_idxs[i]);
        if (i <= num_rows) {
            u_col_ptrs[i] = (unsigned int)(col_ptrs[i]);
        }
    }

    // Fill in system matrix values into factorization sparsity pattern
    A_sym.predictLU(u_row_idxs, u_col_ptrs, vals);

    // Numeric factorization on the GPU
    status_ = LUonDevice(A_sym, cout, cerr, true);
    // Convert unsigned indices back to int
    const auto res_nnz = A_sym.nnz;
    Array<int> res_row_ptrs{exec->get_master(), num_rows + 1};
    Array<int> res_col_idxs{exec->get_master(), res_nnz};
    for (auto i = 0; i < res_nnz; i++) {
        res_col_idxs.get_data()[i] = int(A_sym.sym_r_idx[i]);
        if (i <= num_rows) {
            res_row_ptrs.get_data()[i] = int(A_sym.sym_c_ptr[i]);
        }
    }
    auto res_values =
        Array<double>::view(exec->get_master(), res_nnz, &A_sym.val[0]);

    // Put factorized matrix into ginkgo CSR
    auto host_result =
        matrix_type::create(exec->get_master(), matrix_size, res_values,
                            res_col_idxs, res_row_ptrs);
    auto result = matrix_type::create(exec);
    result->copy_from(host_result->transpose().get());

    // Separate L and U factors: nnz
    Array<int> l_row_ptrs{exec, num_rows + 1};
    Array<int> u_row_ptrs{exec, num_rows + 1};
    exec->run(glu_factorization::make_initialize_row_ptrs_l_u(
        result.get(), l_row_ptrs.get_data(), u_row_ptrs.get_data()));

    // Get nnz from device memory
    auto l_nnz = static_cast<size_type>(
        exec->copy_val_to_host(l_row_ptrs.get_data() + num_rows));
    auto u_nnz = static_cast<size_type>(
        exec->copy_val_to_host(u_row_ptrs.get_data() + num_rows));

    // Init arrays
    Array<int> l_col_idxs{exec, l_nnz};
    Array<double> l_vals{exec, l_nnz};
    std::shared_ptr<matrix_type> l_factor = matrix_type::create(
        exec, matrix_size, std::move(l_vals), std::move(l_col_idxs),
        std::move(l_row_ptrs), parameters_.l_strategy);
    Array<int> u_col_idxs{exec, u_nnz};
    Array<double> u_vals{exec, u_nnz};
    std::shared_ptr<matrix_type> u_factor = matrix_type::create(
        exec, matrix_size, std::move(u_vals), std::move(u_col_idxs),
        std::move(u_row_ptrs), parameters_.u_strategy);

    // Separate L and U: columns and values
    exec->run(glu_factorization::make_initialize_l_u(
        result.get(), l_factor.get(), u_factor.get()));

    return Composition<double>::create(std::move(l_factor),
                                       std::move(u_factor));
}


#define GKO_DECLARE_GLU(ValueType, IndexType) class Glu<ValueType, IndexType>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_GLU);


}  // namespace factorization
}  // namespace gko
