/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2023, the Ginkgo authors
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

#include "core/matrix/sparsity_csr_kernels.hpp"


#include <algorithm>
#include <random>


#include <gtest/gtest.h>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/sparsity_csr.hpp>


#include "core/test/utils.hpp"
#include "core/test/utils/assertions.hpp"
#include "core/test/utils/matrix_generator.hpp"
#include "test/utils/executor.hpp"


namespace {


class SparsityCsr : public CommonTestFixture {
protected:
    using value_type = double;
    using index_type = int;
    using Mtx = gko::matrix::SparsityCsr<value_type, index_type>;

    SparsityCsr() : rng{9312}
    {
        auto data =
            gko::test::generate_random_matrix_data<value_type, index_type>(
                100, 100, std::uniform_int_distribution<index_type>(1, 10),
                std::uniform_real_distribution<value_type>(0.0, 1.0), rng);
        // make sure the matrix contains a few diagonal entries
        for (int i = 0; i < 10; i++) {
            data.nonzeros.emplace_back(i * 3, i * 3, 0.0);
        }
        data.sum_duplicates();
        mtx = Mtx::create(ref);
        mtx->read(data);
        dmtx = gko::clone(exec, mtx);
    }

    std::default_random_engine rng;
    std::unique_ptr<Mtx> mtx;
    std::unique_ptr<Mtx> dmtx;
};


TEST_F(SparsityCsr, KernelDiagonalElementPrefixSumIsEquivalentToRef)
{
    gko::array<index_type> prefix_sum{this->ref, this->mtx->get_size()[0] + 1};
    gko::array<index_type> dprefix_sum{this->exec,
                                       this->mtx->get_size()[0] + 1};

    gko::kernels::reference::sparsity_csr::diagonal_element_prefix_sum(
        ref, mtx.get(), prefix_sum.get_data());
    gko::kernels::EXEC_NAMESPACE::sparsity_csr::diagonal_element_prefix_sum(
        exec, dmtx.get(), dprefix_sum.get_data());

    GKO_ASSERT_ARRAY_EQ(prefix_sum, dprefix_sum);
}


TEST_F(SparsityCsr, KernelRemoveDiagonalElementsIsEquivalentToRef)
{
    const auto num_rows = this->mtx->get_size()[0];
    gko::array<index_type> prefix_sum{this->ref, num_rows + 1};
    gko::kernels::reference::sparsity_csr::diagonal_element_prefix_sum(
        ref, mtx.get(), prefix_sum.get_data());
    gko::array<index_type> dprefix_sum{this->exec, prefix_sum};
    const auto out_mtx = Mtx::create(
        ref, mtx->get_size(),
        mtx->get_num_nonzeros() - prefix_sum.get_const_data()[num_rows]);
    const auto dout_mtx = Mtx::create(
        exec, mtx->get_size(),
        mtx->get_num_nonzeros() - prefix_sum.get_const_data()[num_rows]);

    gko::kernels::reference::sparsity_csr::remove_diagonal_elements(
        ref, mtx->get_const_row_ptrs(), mtx->get_const_col_idxs(),
        prefix_sum.get_const_data(), out_mtx.get());
    gko::kernels::EXEC_NAMESPACE::sparsity_csr::remove_diagonal_elements(
        exec, dmtx->get_const_row_ptrs(), dmtx->get_const_col_idxs(),
        dprefix_sum.get_const_data(), dout_mtx.get());

    GKO_ASSERT_MTX_NEAR(out_mtx, dout_mtx, 0.0);
}


TEST_F(SparsityCsr, ToAdjacencyMatrixIsEquivalentToRef)
{
    const auto out_mtx = mtx->to_adjacency_matrix();
    const auto dout_mtx = dmtx->to_adjacency_matrix();

    GKO_ASSERT_MTX_NEAR(out_mtx, dout_mtx, 0.0);
}


TEST_F(SparsityCsr, ConvertToDenseIsEquivalentToRef)
{
    const auto out_dense = gko::matrix::Dense<value_type>::create(
        exec, mtx->get_size(), mtx->get_size()[1] + 2);
    const auto dout_dense = gko::matrix::Dense<value_type>::create(
        exec, mtx->get_size(), mtx->get_size()[1] + 2);

    mtx->convert_to(out_dense);
    dmtx->convert_to(dout_dense);

    GKO_ASSERT_MTX_NEAR(out_dense, dout_dense, 0.0);
}


}  // namespace
