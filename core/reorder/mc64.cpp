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

#include <ginkgo/core/reorder/mc64.hpp>


#include <memory>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/polymorphic_object.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/permutation.hpp>
#include <ginkgo/core/matrix/sparsity_csr.hpp>


#include "core/matrix/csr_kernels.hpp"
#include "third_party/glu/include/numeric.h"
#include "third_party/glu/include/symbolic.h"
#include "third_party/glu/src/nicslu/include/nics_config.h"
#include "third_party/glu/src/nicslu/include/nicslu.h"
#include "third_party/glu/src/preprocess/preprocess.h"


namespace gko {
namespace reorder {


template <typename ValueType, typename IndexType>
void Mc64<ValueType, IndexType>::generate(std::shared_ptr<const Executor>& exec,
                                          std::shared_ptr<LinOp> system_matrix)
    GKO_NOT_IMPLEMENTED;

template <>
void Mc64<double, int>::generate(std::shared_ptr<const Executor>& exec,
                                 std::shared_ptr<LinOp> system_matrix)
{
    const auto is_gpu_executor = exec != exec->get_master();
    auto cpu_exec = is_gpu_executor ? exec->get_master() : exec;

    // Converts the system matrix to CSR.
    // Throws an exception if it is not convertible.
    auto local_system_matrix = matrix_type::create(cpu_exec);
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

    // MC64 and AMD reorderings + scaling to make diagonal elements dominant
    // and reduce fill-in
    SNicsLU* nicslu = (SNicsLU*)malloc(sizeof(SNicsLU));
    NicsLU_Initialize(nicslu);
    NicsLU_CreateMatrix(nicslu, num_rows, nnz, values, u_row_idxs, u_col_ptrs);
    nicslu->cfgi[0] = 1;
    nicslu->cfgf[1] = 0;
    NicsLU_Analyze(nicslu);
    DumpA(nicslu, values, u_row_idxs, u_col_ptrs);

    // Store scalings and permutations to solve linear system later
    auto mc64_scale = nicslu->cfgi[1];
    auto rp = nicslu->col_perm;
    auto irp = nicslu->row_perm_inv;
    auto piv = nicslu->pivot;
    auto rs = nicslu->col_scale_perm;
    auto cs = nicslu->row_scale;

    Array<double> row_scaling_array{cpu_exec, num_rows};
    Array<double> col_scaling_array{cpu_exec, num_rows};
    index_array perm_array{cpu_exec, num_rows};
    index_array inv_perm_array{cpu_exec, num_rows};
    index_array pivot_array{cpu_exec, num_rows};
    for (auto i = 0; i < num_rows; i++) {
        perm_array.get_data()[i] = int(rp[i]);
        inv_perm_array.get_data()[i] = int(irp[i]);
        pivot_array.get_data()[i] = int(piv[i]);
        row_scaling_array.get_data()[i] = rs[i];
        col_scaling_array.get_data()[i] = cs[i];
    }

    row_scaling_array.set_executor(exec);
    col_scaling_array.set_executor(exec);
    perm_array.set_executor(exec);
    inv_perm_array.set_executor(exec);

    permutation_ = std::make_shared<index_array>(perm_array);
    inv_permutation_ = std::make_shared<index_array>(inv_perm_array);
    row_scaling_ = gko::share(
        ScalingMatrix::create(exec, num_rows, std::move(row_scaling_array)));
    col_scaling_ = gko::share(
        ScalingMatrix::create(exec, num_rows, std::move(col_scaling_array)));
}


#define GKO_DECLARE_MC64(ValueType, IndexType) class Mc64<ValueType, IndexType>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_MC64);


}  // namespace reorder
}  // namespace gko
