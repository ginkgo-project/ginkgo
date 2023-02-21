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

#include "core/factorization/cholesky_kernels.hpp"


#include <algorithm>
#include <memory>


#include <gtest/gtest.h>


#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/csr.hpp>


#include "core/components/prefix_sum_kernels.hpp"
#include "core/factorization/elimination_forest.hpp"
#include "core/test/utils.hpp"
#include "core/test/utils/assertions.hpp"
#include "matrices/config.hpp"
#include "test/utils/executor.hpp"


namespace {


template <typename ValueIndexType>
class Cholesky : public CommonTestFixture {
protected:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using matrix_type = gko::matrix::Csr<value_type, index_type>;

    Cholesky() : tmp{ref}, dtmp{exec}
    {
        matrices.emplace_back(
            "example small",
            gko::initialize<matrix_type>(
                {{1, 0, 1, 0}, {0, 1, 0, 1}, {1, 0, 1, 0}, {0, 0, 0, 1}}, ref));
        matrices.emplace_back("example", gko::initialize<matrix_type>(
                                             {{1, 0, 1, 0, 0, 0, 0, 1, 0, 0},
                                              {0, 1, 0, 1, 0, 0, 0, 0, 0, 1},
                                              {1, 0, 1, 0, 0, 0, 0, 0, 0, 0},
                                              {0, 0, 0, 1, 0, 0, 0, 0, 1, 1},
                                              {0, 1, 0, 0, 1, 0, 0, 0, 1, 1},
                                              {0, 0, 0, 0, 0, 1, 0, 1, 0, 0},
                                              {0, 0, 1, 0, 0, 1, 1, 0, 0, 0},
                                              {1, 0, 0, 0, 0, 1, 0, 1, 1, 1},
                                              {0, 0, 0, 1, 1, 0, 0, 1, 1, 0},
                                              {0, 1, 0, 1, 1, 0, 0, 1, 0, 1}},
                                             ref));
        matrices.emplace_back("separable", gko::initialize<matrix_type>(
                                               {{1, 0, 1, 0, 0, 0, 0, 0, 0, 0},
                                                {0, 1, 1, 0, 0, 0, 0, 0, 0, 0},
                                                {1, 1, 1, 0, 0, 0, 0, 0, 0, 0},
                                                {0, 0, 0, 1, 1, 0, 0, 0, 0, 0},
                                                {0, 0, 0, 1, 1, 1, 0, 0, 0, 1},
                                                {0, 0, 0, 0, 1, 1, 0, 0, 0, 0},
                                                {0, 0, 0, 0, 0, 0, 1, 0, 0, 1},
                                                {0, 0, 0, 0, 0, 0, 1, 1, 0, 0},
                                                {0, 0, 0, 0, 0, 0, 0, 0, 1, 1},
                                                {0, 0, 0, 0, 1, 0, 1, 0, 1, 1}},
                                               ref));
        matrices.emplace_back(
            "missing diagonal",
            gko::initialize<matrix_type>({{1, 0, 1, 0, 0, 0, 0, 0, 0, 0},
                                          {0, 1, 1, 0, 0, 0, 0, 0, 0, 0},
                                          {1, 1, 0, 1, 0, 0, 0, 0, 0, 0},
                                          {0, 0, 1, 1, 1, 0, 0, 0, 0, 0},
                                          {0, 0, 0, 1, 0, 1, 0, 0, 0, 0},
                                          {0, 0, 0, 0, 1, 1, 1, 0, 0, 0},
                                          {0, 0, 0, 0, 0, 1, 1, 1, 0, 1},
                                          {0, 0, 0, 0, 0, 0, 1, 1, 0, 0},
                                          {0, 0, 0, 0, 0, 0, 0, 0, 1, 1},
                                          {0, 0, 0, 0, 0, 0, 1, 0, 1, 0}},
                                         ref));
        std::ifstream ani1_stream{gko::matrices::location_ani1_mtx};
        matrices.emplace_back("ani1", gko::read<matrix_type>(ani1_stream, ref));
        std::ifstream ani1_amd_stream{gko::matrices::location_ani1_amd_mtx};
        matrices.emplace_back("ani1_amd",
                              gko::read<matrix_type>(ani1_amd_stream, ref));
    }

    std::vector<std::pair<std::string, std::unique_ptr<const matrix_type>>>
        matrices;
    gko::array<index_type> tmp;
    gko::array<index_type> dtmp;
};

using Types = ::testing::Types<std::tuple<float, gko::int32>>;

TYPED_TEST_SUITE(Cholesky, Types, PairTypenameNameGenerator);


TYPED_TEST(Cholesky, KernelSymbolicCount)
{
    using matrix_type = typename TestFixture::matrix_type;
    using index_type = typename TestFixture::index_type;
    for (const auto& pair : this->matrices) {
        SCOPED_TRACE(pair.first);
        const auto& mtx = pair.second;
        const auto dmtx = gko::clone(this->exec, mtx);
        std::unique_ptr<gko::factorization::elimination_forest<index_type>>
            forest;
        std::unique_ptr<gko::factorization::elimination_forest<index_type>>
            dforest;
        gko::factorization::compute_elim_forest(mtx.get(), forest);
        gko::factorization::compute_elim_forest(dmtx.get(), dforest);
        gko::array<index_type> row_nnz{this->ref, mtx->get_size()[0]};
        gko::array<index_type> drow_nnz{this->exec, mtx->get_size()[0]};

        gko::kernels::reference::cholesky::cholesky_symbolic_count(
            this->ref, mtx.get(), *forest, row_nnz.get_data(), this->tmp);
        gko::kernels::EXEC_NAMESPACE::cholesky::cholesky_symbolic_count(
            this->exec, dmtx.get(), *dforest, drow_nnz.get_data(), this->dtmp);

        GKO_ASSERT_ARRAY_EQ(drow_nnz, row_nnz);
    }
}


TYPED_TEST(Cholesky, KernelSymbolicFactorize)
{
    using matrix_type = typename TestFixture::matrix_type;
    using index_type = typename TestFixture::index_type;
    using value_type = typename TestFixture::value_type;
    for (const auto& pair : this->matrices) {
        SCOPED_TRACE(pair.first);
        const auto& mtx = pair.second;
        const auto dmtx = gko::clone(this->exec, mtx);
        const auto num_rows = mtx->get_size()[0];
        std::unique_ptr<gko::factorization::elimination_forest<index_type>>
            forest;
        gko::factorization::compute_elim_forest(mtx.get(), forest);
        gko::array<index_type> row_ptrs{this->ref, num_rows + 1};
        gko::kernels::reference::cholesky::cholesky_symbolic_count(
            this->ref, mtx.get(), *forest, row_ptrs.get_data(), this->tmp);
        gko::kernels::reference::components::prefix_sum(
            this->ref, row_ptrs.get_data(), num_rows + 1);
        const auto nnz =
            static_cast<gko::size_type>(row_ptrs.get_const_data()[num_rows]);
        auto l_factor = matrix_type::create(
            this->ref, mtx->get_size(), gko::array<value_type>{this->ref, nnz},
            gko::array<index_type>{this->ref, nnz}, row_ptrs);
        auto dl_factor = matrix_type::create(
            this->exec, mtx->get_size(),
            gko::array<value_type>{this->exec, nnz},
            gko::array<index_type>{this->exec, nnz}, row_ptrs);
        // need to call the device kernels to initialize dtmp
        std::unique_ptr<gko::factorization::elimination_forest<index_type>>
            dforest;
        gko::factorization::compute_elim_forest(dmtx.get(), dforest);
        gko::array<index_type> dtmp_ptrs{this->exec, num_rows + 1};
        gko::kernels::EXEC_NAMESPACE::cholesky::cholesky_symbolic_count(
            this->exec, dmtx.get(), *dforest, dtmp_ptrs.get_data(), this->dtmp);

        gko::kernels::reference::cholesky::cholesky_symbolic_factorize(
            this->ref, mtx.get(), *forest, l_factor.get(), this->tmp);
        gko::kernels::EXEC_NAMESPACE::cholesky::cholesky_symbolic_factorize(
            this->exec, dmtx.get(), *dforest, dl_factor.get(), this->dtmp);

        GKO_ASSERT_MTX_EQ_SPARSITY(dl_factor, l_factor);
    }
}


}  // namespace
