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


#ifndef GKO_CORE_TEST_FACTORIZATION_BLOCK_UTILS_HPP_
#define GKO_CORE_TEST_FACTORIZATION_BLOCK_UTILS_HPP_


#include <ginkgo/core/matrix/fbcsr.hpp>


#include "core/factorization/block_factorization_kernels.hpp"


namespace gko {
namespace test {


/**
 * Initialize L and U block-triangular factor matrices on reference.
 */
template <typename FbcsrType>
void initialize_bilu(const FbcsrType *const mat,
                     std::shared_ptr<FbcsrType> *const l_fact,
                     std::shared_ptr<FbcsrType> *const u_fact)
{
    const auto exec =
        std::dynamic_pointer_cast<const ReferenceExecutor>(mat->get_executor());
    if (!exec) {
        GKO_NOT_SUPPORTED(mat->get_executor());
    }
    using index_type = typename FbcsrType::index_type;
    using value_type = typename FbcsrType::value_type;
    const gko::size_type num_brows = mat->get_num_block_rows();
    const int bs = mat->get_block_size();
    gko::Array<index_type> l_row_ptrs{exec, num_brows + 1};
    gko::Array<index_type> u_row_ptrs{exec, num_brows + 1};
    gko::kernels::reference::factorization::initialize_row_ptrs_BLU(
        exec, mat, l_row_ptrs.get_data(), u_row_ptrs.get_data());
    const auto l_nbnz = l_row_ptrs.get_data()[num_brows];
    const auto u_nbnz = u_row_ptrs.get_data()[num_brows];
    gko::Array<index_type> l_col_idxs(exec, l_nbnz);
    gko::Array<value_type> l_vals(exec, l_nbnz * bs * bs);
    gko::Array<index_type> u_col_idxs(exec, u_nbnz);
    gko::Array<value_type> u_vals(exec, u_nbnz * bs * bs);
    *l_fact = FbcsrType::create(exec, mat->get_size(), bs, std::move(l_vals),
                                std::move(l_col_idxs), std::move(l_row_ptrs));
    *u_fact = FbcsrType::create(exec, mat->get_size(), bs, std::move(u_vals),
                                std::move(u_col_idxs), std::move(u_row_ptrs));
    gko::kernels::reference::factorization::initialize_BLU(
        exec, mat, l_fact->get(), u_fact->get());
}


}  // namespace test
}  // namespace gko


#endif
